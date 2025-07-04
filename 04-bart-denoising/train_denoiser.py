import torch
from torch.utils.data import DataLoader, random_split
from transformers import BartTokenizer, BartConfig
from tqdm import tqdm
from datasets import load_dataset
import wandb
import time
import os
import signal
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
import datetime

# Set CUDA memory allocation configuration for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set cache directory for datasets
CACHE_DIR = "/mnt/win/Users/markhenry/.cache/huggingface"
os.environ['HF_HOME'] = CACHE_DIR

from denoiser import (
    BartDiffusionLM, 
    demo_denoising_step,
    token_discrete_loss
)
from dataset_utils import calculate_optimal_examples, StreamingTextDataset, collect_validation_examples, ValidationDataset

# Model configurations optimized for specific training times
MODEL_CONFIGS = {
    "tiny": {
        "description": "Tiny (2.6M params)",
        "vocab_size": 15_000,
        "batch_size": 96,
        "d_model": 128,
        "encoder_layers": 1,
        "attention_heads": 2,
        "sequence_length": 32,
    },
    "small": {
        "description": "Small (3.6M params)",
        "vocab_size": 22_000,
        "batch_size": 128,
        "d_model": 160,
        "encoder_layers": 2,
        "attention_heads": 4,
        "sequence_length": 48,
    },
    "medium": {
        "description": "Medium (15.2M params)",
        "vocab_size": 30_000,
        "batch_size": 128,
        "d_model": 256,
        "encoder_layers": 3,
        "attention_heads": 4,
        "sequence_length": 64,
    }
}

# Global variable to hold training state for graceful interruption
training_state = {
    'model': None,
    'current_epoch': 0,
    'best_val_loss': float('inf'),
    'current_val_loss': None,
    'current_val_cosine_sim': None,
    'current_val_magnitude_ratio': None,
    'optimizer': None,
    'config_info': None,
}

def signal_handler(signum, frame):
    """Handle Ctrl-C interruption by saving current model state"""
    print(f"\n\n⚠️  Training interrupted (signal {signum})! Saving current state...")
    
    try:
        if training_state['model'] is not None:
            # Save current model as interrupted checkpoint
            current_checkpoint = {
                'model_state_dict': training_state['model'].state_dict(),
                'epoch': training_state['current_epoch'],
                'best_val_loss': training_state['best_val_loss'],
                'val_loss': training_state['current_val_loss'],
                'val_cosine_sim': training_state['current_val_cosine_sim'],
                'val_magnitude_ratio': training_state['current_val_magnitude_ratio'],
                'learning_rate': training_state['optimizer'].param_groups[0]['lr'] if training_state['optimizer'] else None,
                'interrupted': True,
                'config_info': training_state.get('config_info'),
            }
            
            # Save as interrupted checkpoint
            interrupted_path = f"interrupted_epoch_{training_state['current_epoch']}_diffusion_lm.pt"
            torch.save(current_checkpoint, interrupted_path)
            print(f"💾 Saved interrupted checkpoint to: {interrupted_path}")
            
            print(f"✅ Training state saved! You can resume with:")
            print(f"   python train_denoiser.py --checkpoint {interrupted_path}")
        else:
            print(f"❌ No model to save (training not started yet)")
    except Exception as e:
        print(f"⚠️ Error saving checkpoint: {e}")
        print(f"🚨 Forcing exit without saving...")
    
    print(f"👋 Exiting now...")
    
    # Force immediate exit
    try:
        if wandb.run is not None:
            wandb.finish()
    except:
        pass
    
    os._exit(1)  # Force exit without cleanup

# Register signal handler for Ctrl-C
signal.signal(signal.SIGINT, signal_handler)

def bart_config(model_config: dict) -> BartConfig:
    return BartConfig(
        vocab_size=model_config.get('vocab_size', BartConfig.from_pretrained("facebook/bart-base").vocab_size),
        d_model=model_config['d_model'],
        encoder_layers=model_config['encoder_layers'],
        decoder_layers=model_config['encoder_layers'],
        encoder_attention_heads=model_config['attention_heads'],
    )

def config_info(model):
    return {
        'vocab_size': model.bart_config.vocab_size,
        'd_model': model.bart_config.d_model,
        'encoder_layers': model.bart_config.encoder_layers,
        'decoder_layers': model.bart_config.decoder_layers,
        'max_length': model.max_length,
        'time_embed_dim': model.time_embed_dim,
        'num_timesteps': model.scheduler.num_timesteps,
        'dropout': model.dropout_prob,
        'bart_config': model.bart_config,
    }

def create_model(config_info: dict):
    model = BartDiffusionLM(
        bart_model_name="facebook/bart-base",  # Not used when custom_config provided
        max_length=config_info["sequence_length"],  # Use sequence length from config
        time_embed_dim=256,
        num_timesteps=2000,
        dropout=0.1,
        custom_config=bart_config(config_info)
    )
    
    return model


def compute_embedding_health_metrics(model, val_loader, device):
    """Compute comprehensive embedding health metrics."""
    model.eval()
    
    # Metrics to track
    embedding_magnitudes = []
    embedding_values = []
    nearest_neighbor_accuracies = []
    vocab_usage_counts = torch.zeros(model.bart_config.vocab_size, device=device)
    
    # Get vocabulary embeddings for comparison
    vocab_embeddings = model.bart_model.get_encoder().embed_tokens.weight  # [vocab_size, embed_dim]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = input_ids.shape[0]
            
            # Get clean embeddings
            clean_latents = model.embed_tokens(input_ids, attention_mask)
            
            # Add some noise and predict
            timesteps = torch.randint(0, model.scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, _ = model.scheduler.add_noise(clean_latents.transpose(1, 2), timesteps)
            noisy_latents = noisy_latents.transpose(1, 2)
            
            predicted_x0 = model(noisy_latents, timesteps)
            
            # Compute embedding magnitudes
            clean_magnitudes = torch.norm(clean_latents, p=2, dim=-1)  # [B, L]
            pred_magnitudes = torch.norm(predicted_x0, p=2, dim=-1)    # [B, L]
            
            # Apply attention mask
            if attention_mask is not None:
                clean_magnitudes = clean_magnitudes[attention_mask.bool()]
                pred_magnitudes = pred_magnitudes[attention_mask.bool()]
            
            embedding_magnitudes.extend([clean_magnitudes.cpu(), pred_magnitudes.cpu()])
            
            # Collect embedding values for distribution analysis (sample to reduce CPU overhead)
            if batch_idx % 100 == 0:
                clean_values = clean_latents[attention_mask.bool()] if attention_mask is not None else clean_latents.reshape(-1, clean_latents.shape[-1])
                pred_values = predicted_x0[attention_mask.bool()] if attention_mask is not None else predicted_x0.reshape(-1, predicted_x0.shape[-1])
                
                # Sample subset of values to reduce memory usage
                clean_sample = clean_values[::4]  # Every 4th value
                pred_sample = pred_values[::4]    # Every 4th value
                
                embedding_values.extend([clean_sample.cpu(), pred_sample.cpu()])
            
            # Compute nearest neighbor accuracy
            from denoiser import clamp_to_embeddings
            _, clamped_tokens = clamp_to_embeddings(predicted_x0, model, attention_mask)
            
            # Compare with ground truth tokens
            if attention_mask is not None:
                valid_positions = attention_mask.bool()
                correct_predictions = (clamped_tokens[valid_positions] == input_ids[valid_positions]).float()
            else:
                correct_predictions = (clamped_tokens == input_ids).float()
            
            nearest_neighbor_accuracies.append(correct_predictions.mean().cpu().item())
            
            # Track vocabulary usage
            unique_tokens = input_ids[attention_mask.bool()] if attention_mask is not None else input_ids.reshape(-1)
            for token in unique_tokens:
                if 0 <= token < model.bart_config.vocab_size:
                    vocab_usage_counts[token] += 1
    
    # Compute statistics
    all_magnitudes = torch.cat(embedding_magnitudes)
    all_values = torch.cat(embedding_values).reshape(-1)
    
    # Vocabulary usage statistics
    vocab_used = (vocab_usage_counts > 0).sum().item()
    vocab_total = model.bart_config.vocab_size
    vocab_coverage = vocab_used / vocab_total
    
    # Most and least used tokens
    most_used_tokens = torch.topk(vocab_usage_counts, k=10).indices.cpu().tolist()
    least_used_tokens = torch.topk(vocab_usage_counts, k=10, largest=False).indices.cpu().tolist()
    
    metrics = {
        'embedding_magnitude_mean': all_magnitudes.mean().item(),
        'embedding_magnitude_std': all_magnitudes.std().item(),
        'embedding_magnitude_min': all_magnitudes.min().item(),
        'embedding_magnitude_max': all_magnitudes.max().item(),
        'embedding_value_mean': all_values.mean().item(),
        'embedding_value_std': all_values.std().item(),
        'embedding_value_min': all_values.min().item(),
        'embedding_value_max': all_values.max().item(),
        'nearest_neighbor_accuracy': sum(nearest_neighbor_accuracies) / len(nearest_neighbor_accuracies),
        'vocab_coverage': vocab_coverage,
        'vocab_used': vocab_used,
        'vocab_total': vocab_total,
        'most_used_tokens': most_used_tokens,
        'least_used_tokens': least_used_tokens,
        'embedding_value_histogram': torch.histc(all_values, bins=50, min=-3, max=3).cpu().numpy().tolist(),
        'embedding_magnitude_histogram': torch.histc(all_magnitudes, bins=50, min=0, max=5).cpu().numpy().tolist(),
    }
    
    return metrics


def validate_model(model, val_loader, loss_fn, device, demo_example=None, tokenizer=None):
    """Validate the model with the new Diffusion-LM loss."""
    model.eval()
    val_total_loss = 0
    val_cosine_sim = 0
    val_magnitude_ratio = 0
    
    # Track individual loss components for corrected loss function
    val_diffusion_loss = 0
    val_embedding_loss = 0
    val_reconstruction_loss = 0
    val_diffusion_loss_weighted = 0
    val_embedding_loss_weighted = 0
    val_reconstruction_loss_weighted = 0
    val_t0_samples = 0
    
    num_batches = 0
    
    # For loss vs timestep and cosine similarity vs timestep analysis
    timestep_losses = {}
    timestep_cos_sims = {}
    timestep_counts = {}
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = input_ids.shape[0]
            
            # Compute clean latents from token IDs
            latents = model.embed_tokens(input_ids, attention_mask)
            
            timesteps = torch.randint(0, model.scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = model.scheduler.add_noise(latents.transpose(1, 2), timesteps)
            noisy_latents = noisy_latents.transpose(1, 2)  # Back to [B, L, C]
            
            predicted_x0 = model(noisy_latents, timesteps)
            
            loss_dict = loss_fn(predicted_x0, latents, noisy_latents, input_ids, attention_mask, timesteps)
            
            val_total_loss += loss_dict['total_loss'].item()
            val_cosine_sim += loss_dict['cosine_sim'].item()
            val_magnitude_ratio += loss_dict['magnitude_ratio'].item()
            
            # Track individual loss components (corrected for new loss function)
            val_diffusion_loss += loss_dict['diffusion_loss'].item()
            val_embedding_loss += loss_dict['embedding_loss'].item()
            val_reconstruction_loss += loss_dict['reconstruction_loss'].item()
            val_diffusion_loss_weighted += loss_dict['diffusion_loss_weighted'].item()
            val_embedding_loss_weighted += loss_dict['embedding_loss_weighted'].item()
            val_reconstruction_loss_weighted += loss_dict['reconstruction_loss_weighted'].item()
            
            # Track t=0 metrics
            val_t0_samples += loss_dict['t0_samples']
            
            num_batches += 1
            
            # Collect loss vs timestep and cosine similarity vs timestep data (sample every few batches to avoid too much data)
            if num_batches % 100 == 0:
                # Compute per-sample losses and cosine similarities for timestep analysis
                for i in range(batch_size):
                    t = timesteps[i].item()
                    # Compute loss for this single sample
                    sample_pred = predicted_x0[i:i+1]
                    sample_target = latents[i:i+1]
                    sample_loss = F.mse_loss(sample_pred, sample_target).item()
                    
                    # Compute cosine similarity for this single sample
                    pred_flat = sample_pred.reshape(1, -1)
                    target_flat = sample_target.reshape(1, -1)
                    sample_cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).item()
                    
                    # Group timesteps into bins for cleaner visualization
                    t_bin = (t // 100) * 100  # Bin into groups of 100
                    
                    if t_bin not in timestep_losses:
                        timestep_losses[t_bin] = 0
                        timestep_cos_sims[t_bin] = 0
                        timestep_counts[t_bin] = 0
                    
                    timestep_losses[t_bin] += sample_loss
                    timestep_cos_sims[t_bin] += sample_cos_sim
                    timestep_counts[t_bin] += 1
    
    # Log loss vs timestep and cosine similarity vs timestep data to wandb
    if timestep_losses:
        timestep_loss_data = []
        timestep_cosine_data = []
        for t_bin in sorted(timestep_losses.keys()):
            avg_loss = timestep_losses[t_bin] / timestep_counts[t_bin]
            avg_cos_sim = timestep_cos_sims[t_bin] / timestep_counts[t_bin]
            timestep_loss_data.append([t_bin, avg_loss])
            timestep_cosine_data.append([t_bin, avg_cos_sim])
        
        # Create wandb tables for loss vs timestep and cosine similarity vs timestep
        import wandb
        loss_table = wandb.Table(data=timestep_loss_data, columns=["timestep", "loss"])
        cosine_table = wandb.Table(data=timestep_cosine_data, columns=["timestep", "cosine_similarity"])
        wandb.log({
            "val/loss_vs_timestep": wandb.plot.line(loss_table, "timestep", "loss", 
                                                         title="Validation Loss vs Timestep"),
            "val/cosine_similarity_vs_timestep": wandb.plot.line(cosine_table, "timestep", "cosine_similarity", 
                                                                       title="Validation Cosine Similarity vs Timestep")
        })
    
    # Calculate averages
    avg_total_loss = val_total_loss / num_batches
    avg_cosine_sim = val_cosine_sim / num_batches
    avg_magnitude_ratio = val_magnitude_ratio / num_batches
    
    # Calculate average loss components (corrected)
    avg_diffusion_loss = val_diffusion_loss / num_batches
    avg_embedding_loss = val_embedding_loss / num_batches
    avg_reconstruction_loss = val_reconstruction_loss / num_batches
    avg_diffusion_loss_weighted = val_diffusion_loss_weighted / num_batches
    avg_embedding_loss_weighted = val_embedding_loss_weighted / num_batches
    avg_reconstruction_loss_weighted = val_reconstruction_loss_weighted / num_batches
    avg_t0_samples = val_t0_samples / num_batches
    
    # Calculate loss component percentages (corrected)
    total_weighted = (avg_diffusion_loss_weighted + avg_embedding_loss_weighted + avg_reconstruction_loss_weighted)
    
    if total_weighted > 0:
        diffusion_pct = (avg_diffusion_loss_weighted / total_weighted) * 100
        embedding_pct = (avg_embedding_loss_weighted / total_weighted) * 100
        reconstruction_pct = (avg_reconstruction_loss_weighted / total_weighted) * 100
    else:
        diffusion_pct = embedding_pct = reconstruction_pct = 0
    
    # Print detailed loss component analysis (corrected)
    print(f"\n🔍 VALIDATION LOSS COMPONENT ANALYSIS (t=0 Corrected):")
    print(f"   📊 Total Loss: {avg_total_loss:.6f}")
    print(f"   🎯 t=0 Samples per batch: {avg_t0_samples:.1f}")
    print(f"   ⚖️  Loss Component Breakdown:")
    print(f"      🎯 Diffusion (w/ t=0): {avg_diffusion_loss:.6f} → {avg_diffusion_loss_weighted:.6f} ({diffusion_pct:.1f}%)")
    print(f"      🔗 Embedding (t=0):    {avg_embedding_loss:.6f} → {avg_embedding_loss_weighted:.6f} ({embedding_pct:.1f}%)")
    print(f"      📝 Reconstruction:     {avg_reconstruction_loss:.6f} → {avg_reconstruction_loss_weighted:.6f} ({reconstruction_pct:.1f}%)")
    
    # Compute embedding health metrics
    print(f"\n🔬 Computing embedding health metrics...")
    embedding_metrics = compute_embedding_health_metrics(model, val_loader, device)
    
    print(f"📊 EMBEDDING HEALTH METRICS:")
    print(f"   📏 Magnitude: μ={embedding_metrics['embedding_magnitude_mean']:.3f} ± {embedding_metrics['embedding_magnitude_std']:.3f}")
    print(f"   📏 Range: [{embedding_metrics['embedding_magnitude_min']:.3f}, {embedding_metrics['embedding_magnitude_max']:.3f}]")
    print(f"   📈 Values: μ={embedding_metrics['embedding_value_mean']:.3f} ± {embedding_metrics['embedding_value_std']:.3f}")
    print(f"   📈 Range: [{embedding_metrics['embedding_value_min']:.3f}, {embedding_metrics['embedding_value_max']:.3f}]")
    print(f"   🎯 Nearest Neighbor Accuracy: {embedding_metrics['nearest_neighbor_accuracy']:.3f}")
    print(f"   📚 Vocab Coverage: {embedding_metrics['vocab_used']}/{embedding_metrics['vocab_total']} ({embedding_metrics['vocab_coverage']:.3f})")
    
    # Log component analysis to wandb (corrected)
    import wandb
    wandb.log({
        "val/loss_percentages/diffusion": diffusion_pct,
        "val/loss_percentages/embedding": embedding_pct,
        "val/loss_percentages/reconstruction": reconstruction_pct,
        "val/loss/diffusion_raw": avg_diffusion_loss,
        "val/loss/embedding_raw": avg_embedding_loss,
        "val/loss/reconstruction_raw": avg_reconstruction_loss,
        "val/loss/diffusion_weighted": avg_diffusion_loss_weighted,
        "val/loss/embedding_weighted": avg_embedding_loss_weighted,
        "val/loss/reconstruction_weighted": avg_reconstruction_loss_weighted,
        "val/t0_samples_per_batch": avg_t0_samples,
        
        # Embedding health metrics
        "embeddings/magnitude_mean": embedding_metrics['embedding_magnitude_mean'],
        "embeddings/magnitude_std": embedding_metrics['embedding_magnitude_std'],
        "embeddings/magnitude_min": embedding_metrics['embedding_magnitude_min'],
        "embeddings/magnitude_max": embedding_metrics['embedding_magnitude_max'],
        "embeddings/value_mean": embedding_metrics['embedding_value_mean'],
        "embeddings/value_std": embedding_metrics['embedding_value_std'],
        "embeddings/value_min": embedding_metrics['embedding_value_min'],
        "embeddings/value_max": embedding_metrics['embedding_value_max'],
        "embeddings/nearest_neighbor_accuracy": embedding_metrics['nearest_neighbor_accuracy'],
        "embeddings/vocab_coverage": embedding_metrics['vocab_coverage'],
        "embeddings/vocab_used": embedding_metrics['vocab_used'],
        "embeddings/vocab_total": embedding_metrics['vocab_total'],
        
        # Histograms
        "embeddings/value_histogram": wandb.Histogram(embedding_metrics['embedding_value_histogram']),
        "embeddings/magnitude_histogram": wandb.Histogram(embedding_metrics['embedding_magnitude_histogram']),
        
        # Token usage analysis
        "embeddings/most_used_tokens": embedding_metrics['most_used_tokens'],
        "embeddings/least_used_tokens": embedding_metrics['least_used_tokens'],
    })
    
    # Demo denoising during validation
    if demo_example and tokenizer:
        try:
            print(f"\n🎭 VALIDATION DEMO - Denoising at timestep 1:")
            demo_result = demo_denoising_step(
                demo_example, model, tokenizer, device, timestep=1
            )
            
            print(f"   📝 Original: {demo_result['original_text']}")
            print(f"   🟢 Denoised: {demo_result['denoised_text']}")
            print(f"   📊 Cosine Similarity: {demo_result['cosine_similarity']:.4f}")
            
            # Log demo to wandb
            wandb.log({
                "demo/original_text": demo_result['original_text'],
                "demo/noisy_text": demo_result['noisy_text'], 
                "demo/denoised_text": demo_result['denoised_text'],
                "demo/noise_percentage": demo_result['noise_percentage'],
                "demo/cosine_similarity": demo_result['cosine_similarity'],
                "demo/timestep": demo_result['timestep']
            })
            
        except Exception as e:
            print(f"⚠️ Demo failed: {e}")
    
    return avg_total_loss, avg_cosine_sim, avg_magnitude_ratio, embedding_metrics

def train_denoiser(
    model: BartDiffusionLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config_info: dict,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    start_epoch: int = 0,
    initial_best_val_loss: float = float('inf'),
    tokenizer = None
) -> Dict[str, List[float]]:
    """
    Train the denoising model using Diffusion-LM approach.
    Key change: predict clean latents x_0 instead of noise.
    Returns:
        Dictionary containing training history.
    """
    
    def diffusion_lm_loss(predicted_x0, target_x0, noisy_latents, input_ids, attention_mask, timesteps):
        """
        Li et al. 2022 three-component loss function
        
        L_e2e_simple(w) = E_q[L_simple(x_0) + ||EMB(w) - μ_θ(x_1, 1)||² - log p_θ(w|x_0)]
        
        Components:
        1. L_simple: Standard diffusion loss - MSE between predicted and target clean latents
        2. Embedding loss: ||EMB(w) - μ_θ(x_1, 1)||² - prediction from least noisy state vs learnable embedding
        3. Reconstruction loss: -log p_θ(w|x_0) - cross-entropy for rounding back to vocabulary
        
        Args:
            predicted_x0: Model predictions [B, L, C] - predicted clean latents
            target_x0: Ground truth clean latents [B, L, C]  
            noisy_latents: Current noisy latents [B, L, C] (for metrics)
            input_ids: Original token IDs [B, L] for reconstruction loss
            attention_mask: Attention mask [B, L] for masking padded tokens
            timesteps: Timestep indices [B] for embedding loss computation
        """
        batch_size = predicted_x0.shape[0]
        
        # Flatten for computing similarities and magnitudes  
        pred_flat_metrics = predicted_x0.reshape(batch_size, -1)
        target_flat_metrics = target_x0.reshape(batch_size, -1)
        noisy_flat_metrics = noisy_latents.reshape(batch_size, -1)
        
        # COMPONENT 1: Standard diffusion loss
        # Standard diffusion loss for all samples initially
        standard_mse = F.mse_loss(predicted_x0, target_x0, reduction='none')  # [B, L, C]
        
        # COMPONENT 2: Embedding alignment loss
        # From Li et al. 2022: Only apply embedding alignment at t=0
        t0_mask = (timesteps == 0)
        # Get target embeddings for all samples (will be masked appropriately)
        target_embeddings = model.embed_tokens(input_ids, attention_mask)
        
        # Compute embedding alignment loss for t=0 samples
        t0_embedding_loss = F.mse_loss(
            target_embeddings * attention_mask.unsqueeze(-1),
            predicted_x0 * attention_mask.unsqueeze(-1),
            reduction='none'
        )
            
        # Use embedding alignment for t=0 samples, standard MSE for others
        corrected_loss = torch.where(
            t0_mask.unsqueeze(-1).unsqueeze(-1),
            t0_embedding_loss,
            standard_mse
        )
        
        # Track losses (embedding loss is zero when no t=0 samples)
        diffusion_loss = corrected_loss.mean()
        embedding_loss = t0_embedding_loss[t0_mask].mean() if t0_mask.any() else torch.tensor(0.0, device=predicted_x0.device)
        
        # COMPONENT 3: Simplified discrete token reconstruction loss
        reconstruction_loss, reconstruction_accuracy = token_discrete_loss(
            predicted_x0, model, input_ids, attention_mask
        )
        
        # Combine components
        lambda_recon = 1.0  # Weight for reconstruction loss
        total_loss = diffusion_loss + lambda_recon * reconstruction_loss
        
        # Compute additional metrics for monitoring
        pred_magnitudes = torch.norm(pred_flat_metrics, p=2, dim=1)
        target_magnitudes = torch.norm(target_flat_metrics, p=2, dim=1)
        
        cosine_similarities = F.cosine_similarity(pred_flat_metrics, target_flat_metrics, dim=1)
        magnitude_ratios = pred_magnitudes / (target_magnitudes + 1e-8)
        
        # Also compute denoising quality (how well we recover from noise)
        denoising_similarity = F.cosine_similarity(pred_flat_metrics, target_flat_metrics, dim=1)
        noise_similarity = F.cosine_similarity(noisy_flat_metrics, target_flat_metrics, dim=1)
        denoising_improvement = denoising_similarity - noise_similarity
        
        # Reconstruction accuracy is now computed in token_discrete_loss
        
        # Return loss and metrics for logging
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,  # Now includes embedding alignment for t=0
            'embedding_loss': embedding_loss,  # For tracking t=0 alignment
            'reconstruction_loss': reconstruction_loss,
            'l2_loss': diffusion_loss,  # For backward compatibility
            'l1_loss': F.l1_loss(predicted_x0, target_x0),
            'cosine_sim': cosine_similarities.mean(),
            'magnitude_ratio': magnitude_ratios.mean(),
            'pred_magnitude_mean': pred_magnitudes.mean(),
            'target_magnitude_mean': target_magnitudes.mean(),
            'magnitude_ratio_std': magnitude_ratios.std(),
            'cosine_sim_std': cosine_similarities.std(),
            'denoising_improvement': denoising_improvement.mean(),
            'noise_similarity': noise_similarity.mean(),
            'reconstruction_accuracy': reconstruction_accuracy,
            # Individual loss component contributions for analysis
            'diffusion_loss_weighted': diffusion_loss,
            'embedding_loss_weighted': embedding_loss,
            'reconstruction_loss_weighted': lambda_recon * reconstruction_loss,
            # t=0 tracking metrics
            't0_samples': t0_mask.sum().item(),
            'total_samples': batch_size,
        }
    
    # Adaptive learning rate based on model size
    base_lr = 2e-5 if "vocab_size" in config_info and config_info["vocab_size"] <= 15000 else 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    
    # Use ReduceLROnPlateau with validation loss (lower is better)
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=2, min_lr=1e-6
    )
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_cosine_sim": [],
        "val_cosine_sim": [],
        "val_magnitude_ratio": []
    }
    
    best_val_loss = initial_best_val_loss
    patience = 2
    patience_counter = 0
    
    # Update global training state for graceful interruption
    global training_state
    training_state.update({
        'model': model,
        'current_epoch': start_epoch,
        'best_val_loss': best_val_loss,
        'optimizer': optimizer,
        'config_info': config_info,
    })
    
    print(f"🚀 Starting Diffusion-LM training from epoch {start_epoch} (patience {patience})")
    
    # select demo example from train loader
    demo_batch = next(iter(train_loader))
    demo_example = {
        'input_ids': demo_batch['input_ids'][1],  # Take an example from batch
        'attention_mask': demo_batch['attention_mask'][1]
    }
    
    epoch = start_epoch
    while True:
        # Update training state for current epoch
        training_state['current_epoch'] = epoch
        
        model.train()
        epoch_losses = {
            'total': 0, 'l2': 0, 'l1': 0
        }
        epoch_cosine_sim = 0
        epoch_magnitude_ratio = 0
        epoch_pred_magnitude = 0
        epoch_target_magnitude = 0
        epoch_denoising_improvement = 0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} (patience: {patience_counter}/{patience})")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = input_ids.shape[0]
            
            # Compute clean latents from token IDs
            latents = model.embed_tokens(input_ids, attention_mask)
            
            timesteps = torch.randint(0, model.scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = model.scheduler.add_noise(latents.transpose(1, 2), timesteps)
            noisy_latents = noisy_latents.transpose(1, 2)  # Back to [B, L, C]
            
            # Forward pass - predict clean latents x_0
            predicted_x0 = model(noisy_latents, timesteps)  # type: ignore
            
            loss_dict = diffusion_lm_loss(predicted_x0, latents, noisy_latents, input_ids, attention_mask, timesteps)
            
            loss = loss_dict['total_loss']
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate metrics
            epoch_losses['total'] += loss.item()
            epoch_losses['l2'] += loss_dict['l2_loss'].item()
            epoch_losses['l1'] += loss_dict['l1_loss'].item()
            epoch_cosine_sim += loss_dict['cosine_sim'].item()
            epoch_magnitude_ratio += loss_dict['magnitude_ratio'].item()
            epoch_pred_magnitude += loss_dict['pred_magnitude_mean'].item()
            epoch_target_magnitude += loss_dict['target_magnitude_mean'].item()
            epoch_denoising_improvement += loss_dict['denoising_improvement'].item()
            
            # Calculate noise level (signal-to-noise ratio)
            with torch.no_grad():
                alphas_cumprod = model.scheduler.alphas_cumprod.to(device)[timesteps]
                signal_level = torch.sqrt(alphas_cumprod)
                noise_level = torch.sqrt(1 - alphas_cumprod)
                snr = (signal_level / noise_level).mean()
            
            # Log batch metrics less frequently for speed
            if batch_idx % 100 == 0:
                # Calculate weighted component percentages for training batches
                total_weighted_batch = (loss_dict['diffusion_loss_weighted'].item() + 
                                      loss_dict['embedding_loss_weighted'].item() + 
                                      loss_dict['reconstruction_loss_weighted'].item())

                wandb.log({
                    # Main training metrics
                    "train/batch_total_loss": loss.item(),
                    "train/batch_reconstruction_accuracy": loss_dict['reconstruction_accuracy'].item(),
                    "train/batch_cosine_sim": loss_dict['cosine_sim'].item(),
                    "train/batch_magnitude_ratio": loss_dict['magnitude_ratio'].item(),
                    "train/batch_denoising_improvement": loss_dict['denoising_improvement'].item(),
                    "train/batch_noise_similarity": loss_dict['noise_similarity'].item(),

                    # Training parameters
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/grad_norm": grad_norm.item(),
                    "train/batch": batch_idx,
                    "train/epoch": epoch,
                    "train/signal_to_noise_ratio": snr.item(),
                    
                    # t=0 tracking metrics (new)
                    "train/batch_t0_samples": loss_dict['t0_samples'],
                    "train/batch_embedding_loss": loss_dict['embedding_loss'].item(),
                    
                    # Legacy individual metrics (for backward compatibility)
                    "train/batch_l2_loss": loss_dict['l2_loss'].item(),
                    "train/batch_l1_loss": loss_dict['l1_loss'].item(),
                })
        
        # Validation
        val_loss, val_cosine_sim, val_magnitude_ratio, embedding_metrics = validate_model(
            model, val_loader, diffusion_lm_loss, device, 
            demo_example=demo_example, tokenizer=tokenizer
        )
        
        # Update training state with current validation metrics
        training_state.update({
            'current_val_loss': val_loss,
            'current_val_cosine_sim': val_cosine_sim,
            'current_val_magnitude_ratio': val_magnitude_ratio
        })
        
        # Calculate epoch metrics
        num_batches = len(train_loader)
        avg_train_loss = epoch_losses['total'] / num_batches
        avg_train_cosine_sim = epoch_cosine_sim / num_batches
        avg_train_magnitude_ratio = epoch_magnitude_ratio / num_batches
        avg_train_pred_magnitude = epoch_pred_magnitude / num_batches
        avg_train_target_magnitude = epoch_target_magnitude / num_batches
        avg_denoising_improvement = epoch_denoising_improvement / num_batches
        epoch_time = time.time() - epoch_start_time
        
        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["train_cosine_sim"].append(avg_train_cosine_sim)
        history["val_cosine_sim"].append(val_cosine_sim)
        history["val_magnitude_ratio"].append(val_magnitude_ratio)
        
        # Add new metrics to history
        if "train_magnitude_ratio" not in history:
            history["train_magnitude_ratio"] = []
        if "train_l2_loss" not in history:
            history["train_l2_loss"] = []
        if "train_l1_loss" not in history:
            history["train_l1_loss"] = []
        if "denoising_improvement" not in history:
            history["denoising_improvement"] = []
        if "epoch_times" not in history:
            history["epoch_times"] = []
            
        history["train_magnitude_ratio"].append(avg_train_magnitude_ratio)
        history["train_l2_loss"].append(epoch_losses['l2'] / num_batches)
        history["train_l1_loss"].append(epoch_losses['l1'] / num_batches)
        history["denoising_improvement"].append(avg_denoising_improvement)
        history["epoch_times"].append(epoch_time)
        
        # Step the learning rate scheduler based on validation loss
        scheduler_lr.step(val_loss)
        
        # Log comprehensive epoch metrics
        wandb.log({
            # Loss metrics
            "train/epoch_total_loss": avg_train_loss,
            "train/epoch_l2_loss": avg_train_loss,  # Same as total loss for Diffusion-LM
            "train/epoch_l1_loss": epoch_losses['l1'] / num_batches,
            "val/epoch_loss": val_loss,
            
            # Performance metrics
            "train/epoch_cosine_sim": avg_train_cosine_sim,
            "val/epoch_cosine_sim": val_cosine_sim,
            "train/epoch_magnitude_ratio": avg_train_magnitude_ratio,
            "val/magnitude_ratio": val_magnitude_ratio,
            "train/denoising_improvement": avg_denoising_improvement,
            
            # Magnitude analysis
            "train/epoch_pred_magnitude": avg_train_pred_magnitude,
            "train/epoch_target_magnitude": avg_train_target_magnitude,
            "train/magnitude_ratio": avg_train_pred_magnitude / (avg_train_target_magnitude + 1e-8),
            
            # Training efficiency
            "train/epoch_time": epoch_time,
            "train/batches_per_second": num_batches / epoch_time,
            "train/samples_per_second": (num_batches * (train_loader.batch_size or 1)) / epoch_time,
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            
            # Progress tracking
            "epoch": epoch,
            "training_progress": patience_counter / patience,  # Progress based on patience
            "absolute_epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            
            # Embedding health metrics (epoch-level summary)
            "epoch/embedding_magnitude_mean": embedding_metrics['embedding_magnitude_mean'],
            "epoch/embedding_nearest_neighbor_accuracy": embedding_metrics['nearest_neighbor_accuracy'],
            "epoch/embedding_vocab_coverage": embedding_metrics['vocab_coverage']
        })
        
        print(f"\n{'='*80}")
        current_epoch_in_session = epoch + 1 - start_epoch
        print(f"Epoch {epoch + 1} (session epoch {current_epoch_in_session}) - Time: {epoch_time:.2f}s")
        print(f"{'='*80}")
        
        # Loss metrics
        print(f"🎯 Diffusion-LM Loss (x_0 prediction):")
        print(f"   📉 Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"   🏆 Best Val Loss: {best_val_loss:.6f}")
        
        # Secondary metrics
        print(f"📊 Quality Metrics:")
        print(f"   Cosine Similarity - Train: {avg_train_cosine_sim:.4f} | Val: {val_cosine_sim:.4f}")
        print(f"   Magnitude Ratio - Train: {avg_train_magnitude_ratio:.4f} | Val: {val_magnitude_ratio:.4f}")
        print(f"   Denoising Improvement: {avg_denoising_improvement:.4f}")
        
        # Add embedding health summary  
        print(f"🔬 Embedding Health Summary:")
        print(f"   Magnitude: {embedding_metrics['embedding_magnitude_mean']:.3f} ± {embedding_metrics['embedding_magnitude_std']:.3f}")
        print(f"   Nearest Neighbor Accuracy: {embedding_metrics['nearest_neighbor_accuracy']:.3f}")
        print(f"   Vocab Coverage: {embedding_metrics['vocab_coverage']:.3f}")
        
        # Training efficiency
        print(f"⚡ Training Speed:")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Progress tracking based on LOSS
        if patience_counter > 0:
            print(f"⏳ Early Stopping: {patience_counter}/{patience} epochs without loss improvement")
        
        # Save best model based on VALIDATION LOSS (primary metric)
        if val_loss < best_val_loss:
            print(f"💾 NEW BEST MODEL! Val Loss: {val_loss:.6f} (prev: {best_val_loss:.6f})")
            best_val_loss = val_loss
            patience_counter = 0
            
            # Update training state with new best
            training_state['best_val_loss'] = best_val_loss
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'val_loss': val_loss,
                'val_cosine_sim': val_cosine_sim,
                'val_magnitude_ratio': val_magnitude_ratio,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'config_info': config_info,  # Save config for demos
                'checkpoint_version': '1',
                'save_timestamp': time.time(),
            }
            torch.save(checkpoint, "best_diffusion_lm_denoiser.pt")
        else:
            patience_counter += 1
            print(f"⏸️  No val loss improvement for {patience_counter}/{patience} epochs")
            
        # Early stopping based on validation loss plateau
        if patience_counter >= patience:
            print(f"⏹️  Early stopping triggered after {epoch + 1} epochs (no loss improvement)")
            break
            
        # Increment epoch for next iteration
        epoch += 1
    
    # Training summary
    print(f"\n{'='*80}")
    print(f"🎉 DIFFUSION-LM TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"📈 Training Summary:")
    print(f"   Total Epochs: {len(history['train_loss'])}")
    print(f"   Total Time: {sum(history.get('epoch_times', [0])):.2f}s")
    print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"   Best Val Loss: {min(history['val_loss']):.6f}")
    print(f"   Final Train Cosine Sim: {history['train_cosine_sim'][-1]:.4f}")
    print(f"   Final Val Cosine Sim: {history['val_cosine_sim'][-1]:.4f}")
    print(f"   Final Magnitude Ratio: {history['val_magnitude_ratio'][-1]:.4f}")
    print(f"   Average Denoising Improvement: {sum(history.get('denoising_improvement', [0]))/max(len(history.get('denoising_improvement', [1])), 1):.4f}")
    
    # Log final training summary to wandb
    wandb.log({
        "training_summary/total_epochs": len(history['train_loss']),
        "training_summary/total_time": sum(history.get('epoch_times', [0])),
        "training_summary/final_train_loss": history['train_loss'][-1],
        "training_summary/final_val_loss": history['val_loss'][-1],
        "training_summary/best_val_loss": min(history['val_loss']),
        "training_summary/final_train_cosine_sim": history['train_cosine_sim'][-1],
        "training_summary/final_val_cosine_sim": history['val_cosine_sim'][-1],
        "training_summary/final_magnitude_ratio": history['val_magnitude_ratio'][-1],
        "training_summary/improvement_cosine_sim": max(history['val_cosine_sim']) - history['val_cosine_sim'][0],
        "training_summary/improvement_magnitude_ratio": abs(1.0 - history['val_magnitude_ratio'][-1]) - abs(1.0 - history['val_magnitude_ratio'][0])
    })
    
    return history

def count_model_parameters(model: torch.nn.Module, verbose: bool = True) -> int:
    """
    Count the total number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model to count parameters for
        verbose: Whether to print detailed breakdown
        
    Returns:
        Total number of parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print(f"🔢 Parameter Count:")
        print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        
        if total_params != trainable_params:
            frozen_params = total_params - trainable_params
            print(f"   Frozen parameters: {frozen_params:,} ({frozen_params/1e6:.1f}M)")
    
    return total_params

def load_checkpoint(checkpoint_path: str, device: Optional[str] = None) -> Tuple[Optional[BartDiffusionLM], Optional[Dict], bool]:
    """
    Robustly load a diffusion model by detecting configuration from the checkpoint.
    
    This function reads the saved configuration metadata from the checkpoint and 
    automatically creates the model with the correct parameters.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load on (auto-detect if None)
        
    Returns:
        Tuple of (model, checkpoint_metadata, success)
        checkpoint_metadata contains epoch, loss, and other training info
    """
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
        
    try:
        print(f"🔍 Loading model configuration from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device_obj)
        
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary with model_state_dict")
            
        # Try to detect configuration from saved metadata (new format)
        config_info = None
        
        if 'config_info' in checkpoint and checkpoint['config_info'] is not None:
            print("✅ Found config_info in checkpoint")
            config_info = checkpoint['config_info']
        
        if config_info is None:
            raise ValueError("Cannot detect model configuration, no model configuration present")
        
        print(f"🤖 Using model with config: d_model={config_info['d_model']}, layers={config_info['encoder_layers']}, vocab={config_info['vocab_size']}")
        
        # Create the model - use defaults for missing keys
        model = BartDiffusionLM(
            bart_model_name="facebook/bart-base",  # Not used when custom_config provided
            max_length=config_info.get('max_length', 64),
            time_embed_dim=config_info.get('time_embed_dim', 256),
            num_timesteps=config_info.get('num_timesteps', 2000),
            dropout=config_info.get('dropout', 0.1),
            custom_config=bart_config(config_info)
        ).to(device_obj)
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Prepare metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'val_loss': checkpoint.get('val_loss', None),
            'val_cosine_sim': checkpoint.get('val_cosine_sim', None),
            'val_magnitude_ratio': checkpoint.get('val_magnitude_ratio', None),
            'config_info': config_info,
            'checkpoint_version': checkpoint.get('checkpoint_version', 'legacy'),
            'save_timestamp': checkpoint.get('save_timestamp', None)
        }
        
        print(f"✅ Successfully loaded model")
        print(f"📊 Model stats: {count_model_parameters(model, verbose=False)/1e6:.1f}M parameters")
        if metadata['epoch'] > 0:
            print(f"🎯 Training info: Epoch {metadata['epoch']}, Best Val Loss: {metadata['best_val_loss']:.6f}")
        
        return model, metadata, True
        
    except Exception as e:
        print(f"❌ Failed to load model with configuration: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return None, None, False


def main(model_type: str = "tiny", checkpoint_path=None):
    """
    Main training function with configurable model types.
    
    Args:
        model_type: from MODEL_CONFIGS
        checkpoint_path: Path to checkpoint for resuming training
    """
    print(f"=== BART Diffusion Language Model - {model_type.upper()} ===\n")
    
    # Load tokenizer first
    print("Loading BART tokenizer...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", use_fast=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. CREATE MODEL AND COUNT PARAMETERS
    print(f"\n🏗️  Creating {model_type} BART diffusion model...")
    if checkpoint_path:
        model, metadata, success = load_checkpoint(checkpoint_path, str(device))
        if not success or model is None or metadata is None:
            raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")
        start_epoch = metadata['epoch']
        config_info = metadata['config_info']
        model_type = 'loaded'
        print(f"🔄 Continuing Diffusion-LM training from epoch {start_epoch + 1}")
    else:
        config_info = MODEL_CONFIGS[model_type]
        model = create_model(config_info)
        start_epoch = 0
    
    # Count parameters and calculate optimal dataset size  
    param_count = count_model_parameters(model, verbose=True)
    target_examples = calculate_optimal_examples(param_count, tokens_per_example=64)
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   🎯 Target examples: {target_examples:,}")
    
    # 2. init wandb
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"diffusion-lm-{model_type}-v{timestamp}"
    wandb.init(
        project="text-diffusion",
        name=run_name,
        config={
            "model_type": f"BartDiffusionLM_{model_type.title()}",
            "approach": f"Diffusion-LM (Li et al. 2022) - {model_type.title()}",
            "encoder": config_info["description"],
            "embeddings": "Custom BART embeddings (trainable)",
            "architecture": "Custom_BART_encoder + time_embeddings",
            "model_params": f"{param_count/1e6:.1f}M",
            "training_examples": target_examples,
            "batch_size": config_info["batch_size"],
            "learning_rate": 5e-5,
            "patience": 2,
            "sequence_length": config_info["sequence_length"],
            "vocab_size": model.bart_config.vocab_size,
            "dataset": "OpenWebText",
            "noise_scheduler": "sqrt",
            "num_timesteps": 2000,
            "s": 0.008,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "dropout": 0.1,
            "loss_function": "li2022",
            "objective": "L_e2e_simple = E_q[L_simple(x_0) + ||EMB(w) - μ_θ(x_1, 1)||² - log p_θ(w|x_0)]",
            "configuration": f"{model_type}",
            "time_encoding": "sinusoidal_embeddings",
            "time_embed_dim": 256,
            "learned_time_scaling": False,
            "trainable_embeddings": True,
            "attention_mechanisms": "BART_bidirectional",
            "mixed_precision": True,
        }
    )
    
    # 3. Create streaming dataset
    print(f"\n🌊 Initializing streaming dataset...")
    
    # Get sequence length from config
    sequence_length = config_info["sequence_length"]
    
    # Create streaming training dataset
    train_dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        chunk_size=sequence_length,
        vocab_size=config_info.get("vocab_size", tokenizer.vocab_size),
        num_examples=target_examples + 5000,
    )
    
    print(f"   🗣️ Using vocab_size: {config_info.get("vocab_size", tokenizer.vocab_size)}")
    
    # First, collect validation examples from the beginning of the stream
    validation_examples = collect_validation_examples(
        dataset=train_dataset,
        num_examples=5000,
    )
    
    # Create validation dataset from collected examples
    val_dataset = ValidationDataset(validation_examples)

    # Optimized DataLoaders with configured batch size
    batch_size = config_info["batch_size"]
    num_workers = min(8, os.cpu_count() or 1)  # Use multiple workers for data loading
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=4 if num_workers > 0 else 2,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,  # Match training batch size
        shuffle=False, 
        pin_memory=True if device == "cuda" else False,
    )
    
    # Log dataset info
    wandb.log({
        "dataset/train_size": "streaming",
        "dataset/val_size": len(val_dataset),
        "dataset/train_batches": "infinite",
        "dataset/val_batches": len(val_loader),
        "dataset/streaming": True,
        "dataset/same_distribution": True,
    })
    
    # Move model to device
    model = model.to(device)
    print(f"BART Diffusion model moved to device: {device}")
    
    if checkpoint_path:
        print(f"🔄 Continuing Diffusion-LM training from epoch {start_epoch + 1}...")
        initial_best_val_loss = metadata['best_val_loss']
    else:
        print(f"🚀 Training {model_type} Diffusion-LM model...")
        initial_best_val_loss = float('inf')
    
    start_time = time.time()
    history = train_denoiser(
        model, train_loader, val_loader, 
        config_info=config_info, device=device,
        start_epoch=start_epoch, 
        initial_best_val_loss=initial_best_val_loss,
        tokenizer=tokenizer
    )
    total_time = time.time() - start_time
    
    # Log final metrics
    wandb.log({
        "training/total_time": total_time,
        "training/final_train_loss": history["train_loss"][-1],
        "training/final_val_loss": history["val_loss"][-1],
        "training/best_val_loss": min(history["val_loss"])
    })
    
    # Save final model
    torch.save(model.state_dict(), f"final_{model_type}_diffusion_lm_model.pt")
    
    # Save best model as wandb artifact
    try:
        artifact = wandb.Artifact(f"diffusion-lm-{model_type}-best", type="model")
        artifact.add_file("best_diffusion_lm_denoiser.pt")
        wandb.log_artifact(artifact)
    except FileNotFoundError:
        print("Warning: best_diffusion_lm_denoiser.pt not found, skipping artifact upload")
    
    print(f"Diffusion-LM {model_type} training complete!")
    wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BART Diffusion Model")
    parser.add_argument("--model", type=str, default="tiny", choices=["tiny", "small", "medium"], 
                       help="Model type to train")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint path to resume training from")
    parser.add_argument("--continue", action="store_true", default=False,
                       help="Continue training from best checkpoint")
    
    args = parser.parse_args()
    
    # For quick continuation, check for existing checkpoints
    best_checkpoint = "best_diffusion_lm_denoiser.pt"
    
    if args.checkpoint:
        # Explicit checkpoint specified
        main(model_type=args.model, checkpoint_path=args.checkpoint)
    elif getattr(args, 'continue', False):
        main(model_type=args.model, checkpoint_path=best_checkpoint)
    else:
        main(model_type=args.model)