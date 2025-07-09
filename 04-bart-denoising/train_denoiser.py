import torch
from torch.utils.data import DataLoader, random_split
from transformers import BartTokenizer, BartConfig, BertTokenizer, BertConfig
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
    DiffusionLM, 
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
        "description": "Medium (10-15.2M params)",
        "vocab_size": 30_000,
        "batch_size": 128,
        "d_model": 256,
        "encoder_layers": 3,
        "attention_heads": 4,
        "sequence_length": 64,
    },
    "large": {
        "description": "Large (bert-base 'actual size')",
        # "vocab_size": 30_000,
        "batch_size": 128,
        "d_model": 768,
        "encoder_layers": 12,
        "attention_heads": 12,
        "sequence_length": 64,
    }
}

# Encoder type configurations
ENCODER_CONFIGS = {
    "bart": {
        "tokenizer_class": BartTokenizer,
        "config_class": BartConfig,
        "model_name": "facebook/bart-base",
        "config_mapping": {
            "d_model": "d_model",
            "encoder_layers": "encoder_layers", 
            "attention_heads": "encoder_attention_heads",
        }
    },
    "bert": {
        "tokenizer_class": BertTokenizer,
        "config_class": BertConfig,
        "model_name": "bert-base-uncased",
        "config_mapping": {
            "d_model": "hidden_size",
            "encoder_layers": "num_hidden_layers",
            "attention_heads": "num_attention_heads",
        }
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

def create_encoder_config(model_config: dict, encoder_type: str):
    """Create encoder-specific config from unified model config."""
    encoder_config = ENCODER_CONFIGS[encoder_type]
    config_class = encoder_config["config_class"]
    config_mapping = encoder_config["config_mapping"]
    
    # Map unified config to encoder-specific config
    encoder_specific_config = {}
    for unified_key, encoder_key in config_mapping.items():
        if unified_key in model_config:
            encoder_specific_config[encoder_key] = model_config[unified_key]
    
    # Add vocab_size
    encoder_specific_config['vocab_size'] = model_config.get('vocab_size', 30000)
    
    # Add encoder-specific parameters
    if encoder_type == "bart":
        encoder_specific_config['decoder_layers'] = model_config['encoder_layers']
        encoder_specific_config['decoder_attention_heads'] = model_config['attention_heads']
    elif encoder_type == "bert":
        encoder_specific_config['intermediate_size'] = model_config['d_model'] * 4
    
    return config_class(**encoder_specific_config)

def config_info(model):
    return {
        'vocab_size': model.config.vocab_size,
        'd_model': model.encoder.get_latent_dim(),
        'encoder_layers': getattr(model.config, 'encoder_layers', getattr(model.config, 'num_hidden_layers', 1)),
        'max_length': model.max_length,
        'time_embed_dim': model.time_embed_dim,
        'num_timesteps': model.scheduler.num_timesteps,
        'dropout': model.dropout_prob,
        'encoder_type': model.encoder_type,
        'encoder_config': model.config,
    }

def create_model(config_info: dict, encoder_type: str = "bert"):
    """Create a DiffusionLM model with the specified encoder type."""
    encoder_config = create_encoder_config(config_info, encoder_type)
    
    model = DiffusionLM(
        encoder_type=encoder_type,
        model_name=None,  # Use default for encoder type
        max_length=config_info["sequence_length"],
        time_embed_dim=256,
        num_timesteps=2000,
        dropout=0.1,
        custom_config=encoder_config
    )
    
    return model


def compute_embedding_health_metrics(model, val_loader, device):
    """Compute comprehensive embedding health metrics."""
    model.eval()
    
    # Metrics to track
    embedding_magnitudes = []
    embedding_values = []
    nearest_neighbor_accuracies = []
    vocab_usage_counts = torch.zeros(model.encoder.get_vocab_size(), device=device)
    
    # Get vocabulary embeddings for comparison
    vocab_embeddings = model.encoder.get_embedding_weights()  # [vocab_size, embed_dim]
    
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
                if 0 <= token < model.encoder.get_vocab_size():
                    vocab_usage_counts[token] += 1
    
    # Compute statistics
    all_magnitudes = torch.cat(embedding_magnitudes)
    all_values = torch.cat(embedding_values).reshape(-1)
    
    # Vocabulary usage statistics
    vocab_used = (vocab_usage_counts > 0).sum().item()
    vocab_total = model.encoder.get_vocab_size()
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
    val_mse_loss = 0
    val_tT_loss = 0
    val_decoder_nll = 0
    val_mse_loss_weighted = 0
    val_tT_loss_weighted = 0
    val_decoder_nll_weighted = 0
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
            val_mse_loss += loss_dict['mse_loss'].item()
            val_tT_loss += loss_dict['tT_loss'].item()
            val_decoder_nll += loss_dict['decoder_nll'].item()
            val_mse_loss_weighted += loss_dict['mse_loss_weighted'].item()
            val_tT_loss_weighted += loss_dict['tT_loss_weighted'].item()
            val_decoder_nll_weighted += loss_dict['decoder_nll_weighted'].item()
            
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
    avg_mse_loss = val_mse_loss / num_batches
    avg_tT_loss = val_tT_loss / num_batches
    avg_decoder_nll = val_decoder_nll / num_batches
    avg_mse_loss_weighted = val_mse_loss_weighted / num_batches
    avg_tT_loss_weighted = val_tT_loss_weighted / num_batches
    avg_decoder_nll_weighted = val_decoder_nll_weighted / num_batches
    avg_t0_samples = val_t0_samples / num_batches
    
    # Calculate loss component percentages (corrected)
    total_weighted = (avg_mse_loss_weighted + avg_tT_loss_weighted + avg_decoder_nll_weighted)
    
    if total_weighted > 0:
        mse_pct = (avg_mse_loss_weighted / total_weighted) * 100
        tT_pct = (avg_tT_loss_weighted / total_weighted) * 100
        decoder_pct = (avg_decoder_nll_weighted / total_weighted) * 100
    else:
        mse_pct = tT_pct = decoder_pct = 0
    
    # Print detailed loss component analysis (corrected)
    print(f"\n🔍 VALIDATION LOSS COMPONENT ANALYSIS (Corrected Implementation):")
    print(f"   📊 Total Loss: {avg_total_loss:.6f}")
    print(f"   🎯 t=0 Samples per batch: {avg_t0_samples:.1f}")
    print(f"   ⚖️  Loss Component Breakdown:")
    print(f"      🎯 MSE (w/ t=0):       {avg_mse_loss:.6f} → {avg_mse_loss_weighted:.6f} ({mse_pct:.1f}%)")
    print(f"      🌊 tT_loss (prior):    {avg_tT_loss:.6f} → {avg_tT_loss_weighted:.6f} ({tT_pct:.1f}%)")
    print(f"      📝 Decoder NLL:       {avg_decoder_nll:.6f} → {avg_decoder_nll_weighted:.6f} ({decoder_pct:.1f}%)")
    
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
        "val/loss_percentages/mse": mse_pct,
        "val/loss_percentages/tT_loss": tT_pct,
        "val/loss_percentages/decoder_nll": decoder_pct,
        "val/loss/mse_raw": avg_mse_loss,
        "val/loss/tT_loss_raw": avg_tT_loss,
        "val/loss/decoder_nll_raw": avg_decoder_nll,
        "val/loss/mse_weighted": avg_mse_loss_weighted,
        "val/loss/tT_loss_weighted": avg_tT_loss_weighted,
        "val/loss/decoder_nll_weighted": avg_decoder_nll_weighted,
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
    model: DiffusionLM,
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
        Li et al. 2022 three-component loss function (corrected implementation)
        
        L_e2e_simple(w) = E_q[L_simple(x_0) + ||EMB(w) - μ_θ(x_1, 1)||² - log p_θ(w|x_0)]
        
        Components:
        1. MSE loss: Standard diffusion loss with special t=0 handling
        2. tT_loss: Prior matching loss at final timestep (signal fully corrupted)
        3. Decoder NLL: Discrete token reconstruction loss
        
        Args:
            predicted_x0: Model predictions [B, L, C] - predicted clean latents
            target_x0: Ground truth clean latents [B, L, C]  
            noisy_latents: Current noisy latents [B, L, C] (for metrics)
            input_ids: Original token IDs [B, L] for reconstruction loss
            attention_mask: Attention mask [B, L] for masking padded tokens
            timesteps: Timestep indices [B] for loss computation
        """
        batch_size = predicted_x0.shape[0]
        
        # Helper function for mean over non-batch dimensions
        def mean_flat(tensor):
            """Take mean over all non-batch dimensions."""
            return tensor.view(batch_size, -1).mean(dim=1)
        
        # COMPONENT 1: MSE Loss with t=0 special handling
        # Standard MSE loss for all timesteps
        standard_mse = F.mse_loss(predicted_x0, target_x0, reduction='none')  # [B, L, C]
        
        # For t=0 samples, use embedding alignment instead
        t0_mask = (timesteps == 0)
        if t0_mask.any():
            # Get target embeddings for t=0 samples
            target_embeddings = model.embed_tokens(input_ids, attention_mask)
            t0_embedding_loss = F.mse_loss(target_embeddings, predicted_x0, reduction='none')
            
            # Replace MSE with embedding loss for t=0 samples
            corrected_mse = torch.where(
                t0_mask.unsqueeze(-1).unsqueeze(-1),
                t0_embedding_loss,
                standard_mse
            )
        else:
            corrected_mse = standard_mse
        
        # Apply attention mask and compute mean
        attention_mask_3d = attention_mask.unsqueeze(-1).expand_as(corrected_mse)
        masked_mse = corrected_mse * attention_mask_3d.float()
        mse_loss = masked_mse.sum() / attention_mask_3d.sum()
        
        # COMPONENT 2: tT_loss - Prior matching at final timestep
        # This ensures the forward process fully corrupts signal to pure noise
        final_timestep = torch.tensor([model.scheduler.num_timesteps - 1], device=target_x0.device)
        
        # Compute q_mean_variance at final timestep: out_mean = sqrt_alphas_cumprod[T-1] * x_start
        sqrt_alphas_cumprod_final = torch.sqrt(model.scheduler.alphas_cumprod[final_timestep[0]])
        out_mean = sqrt_alphas_cumprod_final * target_x0  # Signal remaining at final timestep
        
        # tT_loss = mean_flat(out_mean ** 2) - measures residual signal strength
        tT_loss_per_sample = mean_flat(out_mean ** 2)
        tT_loss = tT_loss_per_sample.mean()  # Average over batch
        
        # COMPONENT 3: Decoder NLL - Discrete token reconstruction loss
        decoder_nll, reconstruction_accuracy = token_discrete_loss(
            predicted_x0, model, input_ids, attention_mask
        )
        
        # Total loss: MSE + tT_loss + decoder_nll (matching reference implementation)
        total_loss = mse_loss + tT_loss + decoder_nll
        
        # Compute additional metrics for monitoring (only over valid tokens)
        # Apply attention mask to flatten only valid tokens
        valid_mask = attention_mask.bool()
        
        # Get valid tokens only for more accurate metrics
        pred_valid = predicted_x0[valid_mask]  # [num_valid_tokens, C]
        target_valid = target_x0[valid_mask]   # [num_valid_tokens, C]
        noisy_valid = noisy_latents[valid_mask] # [num_valid_tokens, C]
        
        if pred_valid.numel() > 0:
            # Compute magnitudes over valid tokens only
            pred_magnitudes = torch.norm(pred_valid, p=2, dim=1)
            target_magnitudes = torch.norm(target_valid, p=2, dim=1)
            
            # Compute similarities over valid tokens only
            cosine_similarities = F.cosine_similarity(pred_valid, target_valid, dim=1)
            magnitude_ratios = pred_magnitudes / (target_magnitudes + 1e-8)
            
            # Also compute denoising quality (how well we recover from noise)
            denoising_similarity = F.cosine_similarity(pred_valid, target_valid, dim=1)
            noise_similarity = F.cosine_similarity(noisy_valid, target_valid, dim=1)
            denoising_improvement = denoising_similarity - noise_similarity
            
            # Average across valid tokens
            cosine_sim_mean = cosine_similarities.mean()
            magnitude_ratio_mean = magnitude_ratios.mean()
            magnitude_ratio_std = magnitude_ratios.std()
            cosine_sim_std = cosine_similarities.std()
            denoising_improvement_mean = denoising_improvement.mean()
            noise_similarity_mean = noise_similarity.mean()
            pred_magnitude_mean = pred_magnitudes.mean()
            target_magnitude_mean = target_magnitudes.mean()
        else:
            # Fallback if no valid tokens (shouldn't happen in practice)
            cosine_sim_mean = torch.tensor(0.0, device=predicted_x0.device)
            magnitude_ratio_mean = torch.tensor(1.0, device=predicted_x0.device)
            magnitude_ratio_std = torch.tensor(0.0, device=predicted_x0.device)
            cosine_sim_std = torch.tensor(0.0, device=predicted_x0.device)
            denoising_improvement_mean = torch.tensor(0.0, device=predicted_x0.device)
            noise_similarity_mean = torch.tensor(0.0, device=predicted_x0.device)
            pred_magnitude_mean = torch.tensor(0.0, device=predicted_x0.device)
            target_magnitude_mean = torch.tensor(0.0, device=predicted_x0.device)
        
        # Reconstruction accuracy is now computed in token_discrete_loss
        
        # Return loss and metrics for logging
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,  # Main diffusion loss component
            'tT_loss': tT_loss,  # Prior matching loss at final timestep
            'decoder_nll': decoder_nll,  # Discrete token reconstruction loss
            'l2_loss': mse_loss,  # For backward compatibility
            'l1_loss': F.l1_loss(predicted_x0 * attention_mask.unsqueeze(-1), target_x0 * attention_mask.unsqueeze(-1)),
            'cosine_sim': cosine_sim_mean,
            'magnitude_ratio': magnitude_ratio_mean,
            'pred_magnitude_mean': pred_magnitude_mean,
            'target_magnitude_mean': target_magnitude_mean,
            'magnitude_ratio_std': magnitude_ratio_std,
            'cosine_sim_std': cosine_sim_std,
            'denoising_improvement': denoising_improvement_mean,
            'noise_similarity': noise_similarity_mean,
            'reconstruction_accuracy': reconstruction_accuracy,
            # Individual loss component contributions for analysis
            'mse_loss_weighted': mse_loss,
            'tT_loss_weighted': tT_loss,
            'decoder_nll_weighted': decoder_nll,
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
                total_weighted_batch = (loss_dict['mse_loss_weighted'].item() + 
                                      loss_dict['tT_loss_weighted'].item() + 
                                      loss_dict['decoder_nll_weighted'].item())

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
                    "train/batch_mse_loss": loss_dict['mse_loss'].item(),
                    "train/batch_tT_loss": loss_dict['tT_loss'].item(),
                    "train/batch_decoder_nll": loss_dict['decoder_nll'].item(),
                    
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

def load_checkpoint(checkpoint_path: str, device: Optional[str] = None) -> Tuple[Optional[DiffusionLM], Optional[Dict], bool]:
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
        
        # Create the model - detect encoder type from state_dict if not in config
        encoder_type = config_info.get('encoder_type')
        if encoder_type is None:
            # Auto-detect encoder type from state_dict keys for backward compatibility
            state_dict = checkpoint['model_state_dict']
            if 'encoder.model.embeddings.word_embeddings.weight' in state_dict:
                encoder_type = 'bert'
                print("🔍 Auto-detected BERT encoder from checkpoint keys")
            elif 'encoder.model.encoder.embed_tokens.weight' in state_dict:
                encoder_type = 'bart'
                print("🔍 Auto-detected BART encoder from checkpoint keys")
            else:
                encoder_type = 'bert'  # Default fallback for legacy checkpoints
                print("⚠️ Could not auto-detect encoder type, defaulting to BERT")
        
        # Update config_info with detected encoder_type for future saves
        config_info['encoder_type'] = encoder_type
        
        encoder_config = create_encoder_config(config_info, encoder_type)
        
        model = DiffusionLM(
            encoder_type=encoder_type,
            model_name=None,  # Use default for encoder type
            max_length=config_info.get('max_length', 64),
            time_embed_dim=config_info.get('time_embed_dim', 256),
            num_timesteps=config_info.get('num_timesteps', 2000),
            dropout=config_info.get('dropout', 0.1),
            custom_config=encoder_config
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


def main(model_type: str = "tiny", encoder_type: str = "bert", checkpoint_path=None):
    """
    Main training function with configurable model types.
    
    Args:
        model_type: from MODEL_CONFIGS
        encoder_type: "bart" or "bert"
        checkpoint_path: Path to checkpoint for resuming training
    """
    print(f"=== {encoder_type.upper()} Diffusion Language Model - {model_type.upper()} ===\n")
    
    # Load tokenizer first
    encoder_config = ENCODER_CONFIGS[encoder_type]
    print(f"Loading {encoder_type.upper()} tokenizer...")
    tokenizer = encoder_config["tokenizer_class"].from_pretrained(encoder_config["model_name"], use_fast=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. CREATE MODEL AND COUNT PARAMETERS
    print(f"\n🏗️  Creating {model_type} {encoder_type.upper()} diffusion model...")
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
        model = create_model(config_info, encoder_type)
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
            "model_type": f"DiffusionLM_{encoder_type.upper()}_{model_type.title()}",
            "approach": f"Diffusion-LM (Li et al. 2022) - {model_type.title()}",
            "encoder": config_info["description"],
            "embeddings": "Custom embeddings (trainable)",
            "architecture": f"{encoder_type.upper()} encoder + time_embeddings",
            "model_params": f"{param_count/1e6:.1f}M",
            "training_examples": target_examples,
            "batch_size": config_info["batch_size"],
            "learning_rate": 5e-5,
            "patience": 2,
            "sequence_length": config_info["sequence_length"],
            "vocab_size": model.encoder.get_vocab_size(),
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
    
    print(f"   🗣️ Using vocab_size: {config_info.get('vocab_size', getattr(tokenizer, 'vocab_size', 30000))}")
    
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
    print(f"Model moved to device: {device}")
    
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
    
    parser = argparse.ArgumentParser(description="Train Diffusion Language Model")
    parser.add_argument("--model", type=str, default="tiny", choices=["tiny", "small", "medium"], 
                       help="Model type to train")
    parser.add_argument("--encoder", type=str, default="bert", choices=["bart", "bert"],
                       help="Encoder type to use")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint path to resume training from")
    parser.add_argument("--continue", action="store_true", default=False,
                       help="Continue training from best checkpoint")
    
    args = parser.parse_args()
    
    # For quick continuation, check for existing checkpoints
    best_checkpoint = "best_diffusion_lm_denoiser.pt"
    
    if args.checkpoint:
        # Explicit checkpoint specified
        main(model_type=args.model, encoder_type=args.encoder, checkpoint_path=args.checkpoint)
    elif getattr(args, 'continue', False):
        main(model_type=args.model, encoder_type=args.encoder, checkpoint_path=best_checkpoint)
    else:
        main(model_type=args.model, encoder_type=args.encoder)