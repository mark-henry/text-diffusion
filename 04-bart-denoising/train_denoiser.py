import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BartTokenizer
from tqdm import tqdm
from datasets import load_dataset
import wandb
import time
import os
from typing import List, Tuple, Dict
import torch.nn.functional as F

# Set CUDA memory allocation configuration for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from denoiser import (
    BartDiffusionLM, 
    SqrtNoiseScheduler, 
    TextDataset, 
    load_checkpoint,
    get_substantial_texts_from_dataset,
    demo_denoising_step,
    token_discrete_loss
)

def validate_model(model, val_loader, loss_fn, device, demo_text=None, tokenizer=None):
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
            if num_batches % 20 == 0:  # Sample every 5th batch
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
    })
    
    # Demo denoising during validation
    if demo_text and tokenizer:
        try:
            print(f"\n🎭 VALIDATION DEMO - Denoising at timestep 1:")
            demo_result = demo_denoising_step(
                demo_text, model, tokenizer, device, timestep=1
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
    
    return avg_total_loss, avg_cosine_sim, avg_magnitude_ratio

def train_denoiser(
    model: BartDiffusionLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    start_epoch: int = 0,
    initial_best_val_loss: float = float('inf'),
    demo_text = None,
    bart_model = None,
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
        Li et al. 2022 three-component loss function: L_e2e_simple(w)
        
        L_e2e_simple(w) = E_q[L_simple(x_0) + ||EMB(w) - μ_θ(x_1, 1)||² - log p_θ(w|x_0)]
        
        Components:
        1. L_simple: Standard diffusion loss - MSE between predicted and target clean latents
        2. Embedding loss: ||EMB(w) - μ_θ(x_1, 1)||² - prediction from noisiest state vs learnable embedding
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
        # Get learnable embeddings for all samples (will be masked appropriately)
        learnable_embeddings = model.embed_tokens(input_ids, attention_mask)  # type: ignore
        
        # Compute embedding alignment loss for t=0 samples
        t0_embedding_loss = F.mse_loss(
            learnable_embeddings * attention_mask.unsqueeze(-1),
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
    
    # Adaptive learning rate based on performance
    base_lr = 5e-5  # Lower learning rate for pretrained BART
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
    patience = 4
    patience_counter = 0
    
    print(f"🚀 Starting Diffusion-LM training from epoch {start_epoch} to {start_epoch + num_epochs}")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
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
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")):
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
        val_loss, val_cosine_sim, val_magnitude_ratio = validate_model(
            model, val_loader, diffusion_lm_loss, device, 
            demo_text=demo_text, tokenizer=tokenizer
        )
        
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
            "training_progress": (epoch + 1 - start_epoch) / num_epochs,
            "absolute_epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter
        })
        
        print(f"\n{'='*80}")
        current_epoch_in_session = epoch + 1 - start_epoch
        print(f"Epoch {epoch + 1} ({current_epoch_in_session}/{num_epochs} in session - {current_epoch_in_session/num_epochs*100:.1f}%) - Time: {epoch_time:.2f}s")
        print(f"{'='*80}")
        
        # Loss metrics - FOREFRONT
        print(f"🎯 Diffusion-LM Loss (x_0 prediction):")
        print(f"   📉 Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"   🏆 Best Val Loss: {best_val_loss:.6f}")
        
        # Secondary metrics
        print(f"📊 Quality Metrics:")
        print(f"   Cosine Similarity - Train: {avg_train_cosine_sim:.4f} | Val: {val_cosine_sim:.4f}")
        print(f"   Magnitude Ratio - Train: {avg_train_magnitude_ratio:.4f} | Val: {val_magnitude_ratio:.4f}")
        print(f"   Denoising Improvement: {avg_denoising_improvement:.4f}")
        print(f"   Loss Breakdown - L2: {epoch_losses['l2']/num_batches:.4f} | L1: {epoch_losses['l1']/num_batches:.4f}")
        
        # Magnitude analysis
        magnitude_scale = avg_train_pred_magnitude / (avg_train_target_magnitude + 1e-8)
        print(f"🔍 Magnitude Analysis:")
        print(f"   Predicted: {avg_train_pred_magnitude:.3f} | Target: {avg_train_target_magnitude:.3f} | Scale Factor: {magnitude_scale:.3f}")
        
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
            
            # Save comprehensive checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'val_loss': val_loss,
                'val_cosine_sim': val_cosine_sim,
                'val_magnitude_ratio': val_magnitude_ratio,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            torch.save(checkpoint, "best_diffusion_lm_denoiser.pt")
        else:
            patience_counter += 1
            print(f"⏸️  No val loss improvement for {patience_counter}/{patience} epochs")
            
        # Early stopping based on validation loss plateau
        if patience_counter >= patience:
            print(f"⏹️  Early stopping triggered after {epoch + 1} epochs (no loss improvement)")
            break
    
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

def main(checkpoint_path=None, continue_training=False):
    # Determine run name based on whether we're continuing training
    run_name = "diffusion-lm-bart-v4"
    
    # Initialize wandb
    wandb.init(
        project="text-diffusion",
        name=run_name,
        config={
            "model_type": "BartDiffusionLM",
            "approach": "Diffusion-LM (Li et al. 2022)",
            "encoder": "BART-base (fully trainable)",
            "embeddings": "BART embeddings (trainable)",
            "dataset": "WikiText-2",
            "batch_size": 96,
            "learning_rate": 5e-5,  # Lower for pretrained BART
            "num_epochs": 40,
            "max_length": 64,
            "noise_scheduler": "sqrt",
            "num_timesteps": 2000,
            "s": 0.008,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "dropout": 0.1,
            "loss_function": "li2022",
            "objective": "L_e2e_simple = E_q[L_simple(x_0) + ||EMB(w) - μ_θ(x_1, 1)||² - log p_θ(w|x_0)]",
            "improvements": "dropout + weight tying fix",
            "time_encoding": "sinusoidal_embeddings",
            "architecture": "bart_encoder",
            "time_embed_dim": 256,
            "learned_time_scaling": False,
            "trainable_embeddings": True,
            "attention_mechanisms": "BART_bidirectional",
            "mixed_precision": True,
        }
    )
    
    # Load tokenizer
    print("Loading BART tokenizer...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load BART model for demo during validation
    print("Loading BART model for validation demo...")
    from transformers import BartForConditionalGeneration
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    bart_model = bart_model.to(device)  # type: ignore
    bart_model.eval()  # Keep in eval mode for demo
    
    # Load dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Create text dataset for dynamic latent computation
    print("Creating text dataset...")
    train_texts = []
    for item in dataset['train']:  # type: ignore  # Hugging Face datasets support iteration
        if isinstance(item, dict) and 'text' in item and item['text'].strip():
            train_texts.append(item['text'])
        elif isinstance(item, str) and item.strip():
            train_texts.append(item)
    
    # Load demo text for validation
    print("Loading demo text for validation...")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    demo_texts = get_substantial_texts_from_dataset(test_dataset, min_length=150, max_samples=5)
    demo_text = demo_texts[0] if demo_texts else "The quick brown fox jumps over the lazy dog. This is a simple test sentence for denoising demonstration."
    print(f"📝 Demo text: {demo_text[:100]}...")
    
    full_dataset = TextDataset(tokenizer, train_texts, max_length=64)
    
    # Train/validation split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Optimized DataLoaders with larger batch size
    # Note: num_workers=0 and pin_memory=False since data is already on GPU from BART inference
    train_loader = DataLoader(
        train_dataset, 
        batch_size=96,
        shuffle=True, 
        num_workers=0,  # Disable multiprocessing to avoid CUDA issues
        pin_memory=False,  # Data already on GPU, pin_memory not needed
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=96,  # Match training batch size
        shuffle=False, 
        num_workers=0,  # Disable multiprocessing to avoid CUDA issues
        pin_memory=False,  # Data already on GPU, pin_memory not needed
    )
    
    # Log dataset info
    wandb.log({
        "dataset/train_size": len(train_dataset),
        "dataset/val_size": len(val_dataset),
        "dataset/train_batches": len(train_loader),
        "dataset/val_batches": len(val_loader)
    })
    
    # Create text-specific BART diffusion model with dropout for regularization
    print("Creating BartDiffusionLM model for BART semantic latents...")
    model = BartDiffusionLM(bart_model_name="facebook/bart-base", max_length=64, time_embed_dim=256, num_timesteps=2000, dropout=0.1)
    
    # Move model to device
    model = model.to(device)
    print(f"BART Diffusion model moved to device: {device}")
    
    # Load checkpoint if specified
    start_epoch = 0
    initial_best_val_loss = float('inf')
    if continue_training and checkpoint_path:
        success, loaded_epoch, loaded_best_val_loss = load_checkpoint(model, checkpoint_path, device)
        if success:
            start_epoch = loaded_epoch
            initial_best_val_loss = loaded_best_val_loss
            print(f"🔄 Continuing Diffusion-LM training from epoch {start_epoch + 1}")
        else:
            print(f"⚠️  Failed to load checkpoint, starting from scratch")
            continue_training = False
    
    if continue_training:
        print(f"🔄 Continuing Diffusion-LM training from epoch {start_epoch + 1}...")
    else:
        print("🚀 Training Diffusion-LM model...")
    start_time = time.time()
    history = train_denoiser(
        model, train_loader, val_loader, 
        num_epochs=40, device=device,
        start_epoch=start_epoch, 
        initial_best_val_loss=initial_best_val_loss,
        demo_text=demo_text,
        bart_model=bart_model,
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
    torch.save(model.state_dict(), "final_diffusion_lm_model.pt")
    
    # Save best model as wandb artifact
    try:
        artifact = wandb.Artifact("diffusion-lm-best", type="model")
        artifact.add_file("best_diffusion_lm_denoiser.pt")
        wandb.log_artifact(artifact)
    except FileNotFoundError:
        print("Warning: best_diffusion_lm_denoiser.pt not found, skipping artifact upload")
    
    print("Diffusion-LM training complete!")
    wandb.finish()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments for checkpoint loading
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print(f"🔍 Checkpoint specified: {checkpoint_path}")
        main(checkpoint_path=checkpoint_path, continue_training=True)
    else:
        # For quick continuation, check if best checkpoint exists
        import os
        best_checkpoint = "best_diffusion_lm_denoiser.pt"
        if os.path.exists(best_checkpoint):
            print(f"🔍 Found existing Diffusion-LM checkpoint: {best_checkpoint}")
            response = input("Continue training from best checkpoint? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                main(checkpoint_path=best_checkpoint, continue_training=True)
            else:
                main()
        else:
            main() 