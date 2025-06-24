import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
from datasets import load_dataset
import wandb
import time
from typing import List, Tuple, Dict
import torch.nn.functional as F

class CosineNoiseScheduler:
    def __init__(self, num_timesteps: int = 2000, s: float = 0.008):
        self.num_timesteps = num_timesteps
        self.s = s
        
        # Compute alphas_cumprod using cosine schedule: α¯t = cos(((t/T) + s) / (1 + s) * π/2)²
        t = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1
        
        # Ensure no negative values and proper bounds
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8, max=1.0)
        
        # Compute betas from alphas_cumprod
        self.alphas_cumprod = alphas_cumprod
        self.alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        self.betas = 1 - self.alphas
        
    def add_noise(self, latents: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to latents according to timesteps."""
        noise = torch.randn_like(latents)
        # Move alphas_cumprod to same device as timesteps
        alphas_cumprod = self.alphas_cumprod.to(timesteps.device)[timesteps]
        
        # Reshape for broadcasting with [B, C, L] format
        alphas_cumprod = alphas_cumprod.view(-1, 1, 1)
        noisy_latents = torch.sqrt(alphas_cumprod) * latents + torch.sqrt(1 - alphas_cumprod) * noise
        return noisy_latents, noise

def pad_tensor(tensor: torch.Tensor, max_length: int) -> torch.Tensor:
    """Pad tensor to max_length."""
    current_length = tensor.shape[0]
    if current_length >= max_length:
        return tensor[:max_length]
    
    # Create padding on the same device as the input tensor
    padding = torch.zeros((max_length - current_length, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=0)

class LatentDataset(Dataset):
    def __init__(self, model: BartForConditionalGeneration, tokenizer: BartTokenizer, 
                 dataset: List[str], max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = [text for text in dataset if text.strip()]  # Filter empty texts
        self.max_length = max_length
        
        # Pre-compute statistics for normalization
        print("Computing dataset statistics for normalization...")
        self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean and std for latent normalization."""
        sample_size = min(500, len(self.dataset))  # Smaller sample for efficiency
        latents_sample = []
        
        for i in range(0, sample_size, 20):  # Every 20th sample
            text = self.dataset[i]
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)
            # Move inputs to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                encoder_outputs = self.model.get_encoder()(inputs["input_ids"])
                latents = encoder_outputs.last_hidden_state.squeeze(0)
                latents = pad_tensor(latents, self.max_length)
                latents_sample.append(latents)
        
        if latents_sample:
            all_latents = torch.stack(latents_sample)
            self.latent_mean = all_latents.mean(dim=(0, 1), keepdim=True)
            self.latent_std = all_latents.std(dim=(0, 1), keepdim=True) + 1e-8
        else:
            self.latent_mean = torch.zeros(1, 768)
            self.latent_std = torch.ones(1, 768)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.dataset[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)
        # Move inputs to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(inputs["input_ids"])
            latents = encoder_outputs.last_hidden_state.squeeze(0)
            latents = pad_tensor(latents, self.max_length)
            # Normalize latents
            latents = (latents - self.latent_mean.squeeze()) / self.latent_std.squeeze()
        return latents

class TextUNet1D(nn.Module):
    """
    UNet specifically designed for BART semantic latent representations.
    
    Input: [batch_size, 768, 128] where:
    - 768 channels = BART-base semantic embedding dimensions (not frequency bands)
    - 128 sequence length = token positions (not time steps)
    - Values are normalized semantic embeddings (not raw audio/signals)
    
    Design Principles:
    - Gentle downsampling to preserve text structure
    - Skip connections to maintain semantic information across scales
    - No attention mechanisms (simpler is better for semantic embeddings)
    - Moderate parameter count (~5-10M) to avoid overfitting
    - Time embedding integrated throughout the network
    """
    def __init__(self, in_channels=768, out_channels=768, time_embed_dim=256, num_timesteps=2000):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Sinusoidal time embedding for diffusion timesteps
        # Much better than learned embedding for timestep conditioning
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Encoder (downsampling path)
        # We use gentle downsampling since 128 tokens isn't that long
        # Each block reduces sequence length by 2x while increasing channels
        
        # Block 1: 768 -> 256 channels, 128 -> 64 sequence length
        self.down1 = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),  # GroupNorm works well with varying batch sizes
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        self.down1_pool = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)
        
        # Time projection for down1
        self.time_proj1 = nn.Linear(time_embed_dim, 256)
        
        # Block 2: 256 -> 384 channels, 64 -> 32 sequence length
        self.down2 = nn.Sequential(
            nn.Conv1d(256, 384, kernel_size=3, padding=1),
            nn.GroupNorm(8, 384),
            nn.SiLU(),
            nn.Conv1d(384, 384, kernel_size=3, padding=1),
            nn.GroupNorm(8, 384),
            nn.SiLU(),
        )
        self.down2_pool = nn.Conv1d(384, 384, kernel_size=3, stride=2, padding=1)
        
        # Time projection for down2
        self.time_proj2 = nn.Linear(time_embed_dim, 384)
        
        # Bottleneck: 384 -> 512 channels, 32 sequence length (no downsampling)
        # This is where the model processes the most compressed representation
        self.bottleneck = nn.Sequential(
            nn.Conv1d(384, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv1d(512, 384, kernel_size=3, padding=1),  # Back to 384 for skip connection
            nn.GroupNorm(8, 384),
            nn.SiLU(),
        )
        
        # Time projection for bottleneck
        self.time_proj_bottleneck = nn.Linear(time_embed_dim, 384)
        
        # Decoder (upsampling path with skip connections)
        # Skip connections help preserve semantic details lost during downsampling
        
        # Upsampling layers (applied before concatenation)
        self.up2_upsample = nn.ConvTranspose1d(384, 384, kernel_size=4, stride=2, padding=1)
        self.up1_upsample = nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1)
        
        # Block 3: (384 + 384) -> 256 channels, after upsampling to 64 sequence length
        self.up2 = nn.Sequential(
            nn.Conv1d(384 + 384, 384, kernel_size=3, padding=1),  # +384 from skip connection
            nn.GroupNorm(8, 384),
            nn.SiLU(),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        
        # Time projection for up2
        self.time_proj3 = nn.Linear(time_embed_dim, 256)
        
        # Block 4: (256 + 256) -> 256 channels, after upsampling to 128 sequence length
        self.up1 = nn.Sequential(
            nn.Conv1d(256 + 256, 256, kernel_size=3, padding=1),  # +256 from skip connection
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        
        # Time projection for up1
        self.time_proj4 = nn.Linear(time_embed_dim, 256)
        
        # Final output layer: back to original 768 channels
        # This maps from processed features back to BART embedding space
        self.final = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, out_channels, kernel_size=1),  # 1x1 conv for final projection
        )

    def get_sinusoidal_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal timestep embeddings like in original Transformer paper.
        This gives much better timestep conditioning than learned embeddings.
        
        Args:
            timesteps: [batch_size] tensor of timestep values
            embedding_dim: dimension of output embeddings
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:  # If odd embedding dim, add one more dimension
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=-1)
        return emb
        

    def forward(self, x, timesteps):
        """
        Forward pass through the text-specific UNet.
        
        Args:
            x: [batch_size, 768, 128] - noisy BART latents
            timesteps: [batch_size] - diffusion timesteps
            
        Returns:
            [batch_size, 768, 128] - predicted noise
        """
        batch_size = x.shape[0]
        
        # Embed timesteps using sinusoidal embedding
        # Fixed: Don't normalize timesteps, use them directly for sinusoidal encoding
        t_emb = self.get_sinusoidal_embedding(timesteps.float(), self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # [batch_size, time_embed_dim]
        
        # Encoder path with skip connection storage
        # Down1: 768 -> 256 channels, 128 -> 64 length
        skip1 = self.down1(x)  # Store for skip connection
        skip1 = skip1 + self.time_proj1(t_emb).unsqueeze(-1)  # Add time information
        x1 = self.down1_pool(skip1)
        
        # Down2: 256 -> 384 channels, 64 -> 32 length  
        skip2 = self.down2(x1)  # Store for skip connection
        skip2 = skip2 + self.time_proj2(t_emb).unsqueeze(-1)  # Add time information
        x2 = self.down2_pool(skip2)
        
        # Bottleneck: process the most compressed representation
        x3 = self.bottleneck(x2)
        x3 = x3 + self.time_proj_bottleneck(t_emb).unsqueeze(-1)  # Add time information
        
        # Decoder path with skip connections
        # Up2: 32 -> 64 length, use skip connection from down2
        x4 = self.up2_upsample(x3)  # First upsample, then concatenate
        # Ensure dimensions match for concatenation
        if x4.shape[-1] != skip2.shape[-1]:
            # Adjust skip2 to match x4's sequence length
            skip2 = F.interpolate(skip2, size=x4.shape[-1], mode='linear', align_corners=False)
        x4 = torch.cat([x4, skip2], dim=1)  # Concatenate skip connection
        x4 = self.up2(x4)
        x4 = x4 + self.time_proj3(t_emb).unsqueeze(-1)  # Add time information
        
        # Up1: 64 -> 128 length, use skip connection from down1
        x5 = self.up1_upsample(x4)  # First upsample, then concatenate
        # Ensure dimensions match for concatenation
        if x5.shape[-1] != skip1.shape[-1]:
            # Adjust skip1 to match x5's sequence length
            skip1 = F.interpolate(skip1, size=x5.shape[-1], mode='linear', align_corners=False)
        x5 = torch.cat([x5, skip1], dim=1)  # Concatenate skip connection
        x5 = self.up1(x5)
        x5 = x5 + self.time_proj4(t_emb).unsqueeze(-1)  # Add time information
        
        # Final output: back to 768 channels (BART embedding space)
        output = self.final(x5)
        
        return output

def load_checkpoint(model, checkpoint_path, device):
    """
    Load a model checkpoint and return the loaded state.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        bool: True if checkpoint loaded successfully, False otherwise
    """
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint with training state
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"✅ Loaded checkpoint from epoch {epoch} with best val loss: {best_val_loss:.6f}")
            return True, epoch, best_val_loss
        else:
            # Simple state dict
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights from checkpoint")
            return True, 0, float('inf')
            
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False, 0, float('inf')

def validate_model(model, val_loader, scheduler, loss_fn, device):
    """Validate the model with the given loss."""
    model.eval()
    val_total_loss = 0
    val_cosine_sim = 0
    val_magnitude_ratio = 0
    num_batches = 0
    
    with torch.no_grad():
        for latents in val_loader:
            latents = latents.to(device)
            batch_size = latents.shape[0]
            latents = latents.transpose(1, 2)  # [B, L, C] -> [B, C, L]
            
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = scheduler.add_noise(latents, timesteps)
            
            predicted_noise = model(noisy_latents, timesteps)
            
            loss_dict = loss_fn(predicted_noise, noise)
            
            val_total_loss += loss_dict['total_loss'].item()
            val_cosine_sim += loss_dict['cosine_sim'].item()
            val_magnitude_ratio += loss_dict['magnitude_ratio'].item()
            num_batches += 1
    
    return val_total_loss / num_batches, val_cosine_sim / num_batches, val_magnitude_ratio / num_batches

def train_denoiser(
    model: nn.Module,
    scheduler: CosineNoiseScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    start_epoch: int = 0,
    initial_best_val_loss: float = float('inf')
) -> Dict[str, List[float]]:
    """
    Train the denoising model.
    Returns:
        Dictionary containing training history.
    """
    
    def simple_l1_loss(predicted_noise, target_noise):
        """
        Simple L1 (mean absolute error) loss function.
        
        L1 loss is much better than MSE for magnitude preservation.
        It doesn't penalize large errors as harshly, leading to better magnitude scaling.
        
        Args:
            predicted_noise: Model predictions [B, C, L]
            target_noise: Ground truth noise [B, C, L] 
        """
        batch_size = predicted_noise.shape[0]
        
        # Flatten for computing similarities and magnitudes  
        pred_flat = predicted_noise.reshape(batch_size, -1)
        target_flat = target_noise.reshape(batch_size, -1)
        
        # L1 loss - mean absolute error
        l1_loss = F.l1_loss(predicted_noise, target_noise)
        
        # Compute additional metrics for monitoring (but don't use in loss)
        cosine_similarities = F.cosine_similarity(pred_flat, target_flat, dim=1)
        pred_magnitudes = torch.norm(pred_flat, dim=1)
        target_magnitudes = torch.norm(target_flat, dim=1)
        magnitude_ratios = pred_magnitudes / (target_magnitudes + 1e-8)
        
        # Also compute other losses for comparison
        mse_loss = F.mse_loss(predicted_noise, target_noise)
        l2_loss = torch.norm(predicted_noise - target_noise, dim=(1, 2)).mean()
        
        # Return loss and metrics for logging
        return {
            'total_loss': l1_loss,
            'l1_loss': l1_loss,
            'cosine_sim': cosine_similarities.mean(),
            'magnitude_ratio': magnitude_ratios.mean(),
            # Additional metrics for comparison
            'mse_loss': mse_loss,
            'l2_loss': l2_loss,
            'pred_magnitude_mean': pred_magnitudes.mean(),
            'target_magnitude_mean': target_magnitudes.mean(),
            'magnitude_ratio_std': magnitude_ratios.std(),
            'cosine_sim_std': cosine_similarities.std()
        }
    
    # Adaptive learning rate based on performance
    base_lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    
    # Use ReduceLROnPlateau with validation loss (lower is better)
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=2, min_lr=1e-6
    )  # Mode 'min' because we're tracking validation loss (lower is better)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_cosine_sim": [],
        "val_cosine_sim": [],
        "val_magnitude_ratio": []
    }
    
    best_val_loss = float('inf')
    initial_best_val_loss = float('inf')
    patience = 4
    patience_counter = 0
    
    print(f"🚀 Starting training from epoch {start_epoch} to {start_epoch + num_epochs}")
    print(f"📊 Tracking validation loss for early stopping and checkpointing")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_losses = {
            'total': 0, 'mse': 0, 'l1': 0
        }
        epoch_cosine_sim = 0
        epoch_magnitude_ratio = 0
        epoch_pred_magnitude = 0
        epoch_target_magnitude = 0
        epoch_start_time = time.time()
        
        for batch_idx, latents in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")):
            latents = latents.to(device)
            batch_size = latents.shape[0]
            # latents should be [B, L, C] -> transpose to [B, C, L] for conv1d
            latents = latents.transpose(1, 2)
            
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = scheduler.add_noise(latents, timesteps)
            
            # Forward pass
            predicted_noise = model(noisy_latents, timesteps)
            
            loss_dict = simple_l1_loss(predicted_noise, noise)
            
            loss = loss_dict['total_loss']
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Reduced for stability
            optimizer.step()
            
            # Accumulate metrics
            epoch_losses['total'] += loss.item()
            epoch_losses['mse'] += loss_dict['mse_loss'].item()
            epoch_losses['l1'] += loss_dict['l1_loss'].item()
            epoch_cosine_sim += loss_dict['cosine_sim'].item()
            epoch_magnitude_ratio += loss_dict['magnitude_ratio'].item()
            epoch_pred_magnitude += loss_dict['pred_magnitude_mean'].item()
            epoch_target_magnitude += loss_dict['target_magnitude_mean'].item()
            
            # Calculate noise level (signal-to-noise ratio)
            with torch.no_grad():
                alphas_cumprod = scheduler.alphas_cumprod.to(device)[timesteps]
                signal_level = torch.sqrt(alphas_cumprod)
                noise_level = torch.sqrt(1 - alphas_cumprod)
                snr = (signal_level / noise_level).mean()
            
            # Log batch metrics less frequently for speed
            if batch_idx % 50 == 0:
                wandb.log({
                    "train/batch_total_loss": loss.item(),
                    "train/batch_l1_loss": loss_dict['l1_loss'].item(),
                    "train/batch_cosine_sim": loss_dict['cosine_sim'].item(),
                    "train/batch_magnitude_ratio": loss_dict['magnitude_ratio'].item(),
                    # Additional metrics for comparison
                    "train/batch_mse_loss": loss_dict['mse_loss'].item(),
                    "train/batch_l2_loss": loss_dict['l2_loss'].item(),
                    "train/batch_pred_magnitude": loss_dict['pred_magnitude_mean'].item(),
                    "train/batch_target_magnitude": loss_dict['target_magnitude_mean'].item(),
                    "train/batch_magnitude_ratio_std": loss_dict['magnitude_ratio_std'].item(),
                    "train/batch_cosine_sim_std": loss_dict['cosine_sim_std'].item(),
                    # Training parameters
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/grad_norm": grad_norm.item(),
                    "train/batch": batch_idx,
                    "train/epoch": epoch,
                    "train/signal_to_noise_ratio": snr.item(),
                })
        
        # Validation
        val_loss, val_cosine_sim, val_magnitude_ratio = validate_model(model, val_loader, scheduler, simple_l1_loss, device)
        
        # Calculate epoch metrics
        num_batches = len(train_loader)
        avg_train_loss = epoch_losses['total'] / num_batches
        avg_train_cosine_sim = epoch_cosine_sim / num_batches
        avg_train_magnitude_ratio = epoch_magnitude_ratio / num_batches
        avg_train_pred_magnitude = epoch_pred_magnitude / num_batches
        avg_train_target_magnitude = epoch_target_magnitude / num_batches
        epoch_time = time.time() - epoch_start_time
        
        # Update history with comprehensive metrics
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["train_cosine_sim"].append(avg_train_cosine_sim)
        history["val_cosine_sim"].append(val_cosine_sim)
        history["val_magnitude_ratio"].append(val_magnitude_ratio)
        
        # Add new metrics to history
        if "train_magnitude_ratio" not in history:
            history["train_magnitude_ratio"] = []
        if "train_mse_loss" not in history:
            history["train_mse_loss"] = []
        if "train_l1_loss" not in history:
            history["train_l1_loss"] = []
        if "epoch_times" not in history:
            history["epoch_times"] = []
            
        history["train_magnitude_ratio"].append(avg_train_magnitude_ratio)
        history["train_mse_loss"].append(epoch_losses['mse'] / num_batches)
        history["train_l1_loss"].append(epoch_losses['l1'] / num_batches)
        history["epoch_times"].append(epoch_time)
        
        # Step the learning rate scheduler based on validation loss (we want to minimize this)
        scheduler_lr.step(val_loss)
        
        # Log comprehensive epoch metrics
        wandb.log({
            # Loss metrics
            "train/epoch_total_loss": avg_train_loss,
            "train/epoch_l1_loss": avg_train_loss,  # Same as total loss for L1
            "train/epoch_mse_loss": epoch_losses['mse'] / num_batches,
            "train/epoch_l2_loss": epoch_losses['l1'] / num_batches,  # Note: this was swapped
            "val/epoch_loss": val_loss,
            
            # Performance metrics
            "train/epoch_cosine_sim": avg_train_cosine_sim,
            "val/epoch_cosine_sim": val_cosine_sim,
            "train/epoch_magnitude_ratio": avg_train_magnitude_ratio,
            "val/magnitude_ratio": val_magnitude_ratio,
            
            # Magnitude analysis
            "train/epoch_pred_magnitude": avg_train_pred_magnitude,
            "train/epoch_target_magnitude": avg_train_target_magnitude,
            "train/magnitude_ratio": avg_train_pred_magnitude / (avg_train_target_magnitude + 1e-8),
            
            # Training efficiency
            "train/epoch_time": epoch_time,
            "train/batches_per_second": num_batches / epoch_time,
            "train/samples_per_second": (num_batches * train_loader.batch_size) / epoch_time,
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
        print(f"🎯 Loss metrics:")
        print(f"   📉 Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"   🏆 Best Val Loss: {best_val_loss:.6f}")
        
        # Secondary metrics
        print(f"📊 Secondary Metrics:")
        print(f"   Cosine Similarity - Train: {avg_train_cosine_sim:.4f} | Val: {val_cosine_sim:.4f}")
        print(f"   Magnitude Ratio - Train: {avg_train_magnitude_ratio:.4f} | Val: {val_magnitude_ratio:.4f}")
        print(f"   Loss Breakdown - L1: {epoch_losses['l1']/num_batches:.4f} | MSE: {epoch_losses['mse']/num_batches:.4f}")
        
        # Magnitude analysis
        magnitude_scale = avg_train_pred_magnitude / (avg_train_target_magnitude + 1e-8)
        print(f"🔍 Magnitude Analysis:")
        print(f"   Predicted: {avg_train_pred_magnitude:.3f} | Target: {avg_train_target_magnitude:.3f} | Scale Factor: {magnitude_scale:.3f}")
        
        # Training efficiency
        batches_per_sec = num_batches / epoch_time
        samples_per_sec = (num_batches * train_loader.batch_size) / epoch_time
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
            torch.save(checkpoint, "best_multi_objective_denoiser.pt")
        else:
            patience_counter += 1
            print(f"⏸️  No val loss improvement for {patience_counter}/{patience} epochs")
            
        # Early stopping based on validation loss plateau
        if patience_counter >= patience:
            print(f"⏹️  Early stopping triggered after {epoch + 1} epochs (no loss improvement)")
            break
    
    # Training summary
    print(f"\n{'='*80}")
    print(f"🎉 TRAINING COMPLETED!")
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
    run_name = "bart-denoiser-v14-l1-sinusoidal-embedding"
    
    # Initialize wandb
    wandb.init(
        project="text-diffusion",
        name=run_name,
        config={
            "model_type": "TextUNet1D",
            "encoder": "BART-base",
            "dataset": "WikiText-2",
            "batch_size": 96,
            "learning_rate": 1e-3,
            "num_epochs": 40,
            "max_length": 128,
            "noise_scheduler": "cosine",
            "num_timesteps": 2000,
            "s": 0.008,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,  # Reduced for stability
            "loss_function": "l1_loss",
            "adaptive_loss_weights": "cosine_1.0->3.0_magnitude_0.5->2.0",
            "early_stopping_metric": "loss",
            "normalization": "latent_normalization",
            "architecture": "custom_text_unet_with_skip_connections",
            "time_embed_dim": 256,
            "downsampling_strategy": "gentle_2x_per_block",
            "skip_connections": True,
            "attention_mechanisms": False,
            "mixed_precision": True,
        }
    )
    
    # Load BART model and tokenizer
    print("Loading BART model and tokenizer...")
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Move BART to GPU for fast latent computation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bart_model = bart_model.to(device)
    print(f"BART model moved to device: {device}")
    
    # Load dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Create dataset with normalization
    print("Creating latent dataset...")
    full_dataset = LatentDataset(bart_model, tokenizer, dataset['train']['text'], max_length=128)
    
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
    
    # Create text-specific UNet model
    print("Creating TextUNet1D model for BART semantic latents...")
    model = TextUNet1D(in_channels=768, out_channels=768, time_embed_dim=256, num_timesteps=2000)
    
    # Move UNet to device
    model = model.to(device)
    print(f"UNet model moved to device: {device}")
    
    # Load checkpoint if specified
    start_epoch = 0
    initial_best_val_loss = float('inf')
    if continue_training and checkpoint_path:
        success, loaded_epoch, loaded_best_val_loss = load_checkpoint(model, checkpoint_path, device)
        if success:
            start_epoch = loaded_epoch
            initial_best_val_loss = loaded_best_val_loss
            print(f"🔄 Continuing training from epoch {start_epoch + 1}")
        else:
            print(f"⚠️  Failed to load checkpoint, starting from scratch")
            continue_training = False
    
    # Create noise scheduler with cosine schedule
    scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
    
    # Train model with improved settings
    if continue_training:
        print(f"🔄 Continuing training denoising model from epoch {start_epoch + 1}...")
    else:
        print("🚀 Training denoising model with magnitude scaling fixes...")
    start_time = time.time()
    history = train_denoiser(
        model, scheduler, train_loader, val_loader, 
        num_epochs=40, device=device,
        start_epoch=start_epoch, 
        initial_best_val_loss=initial_best_val_loss
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
    torch.save(model.state_dict(), "final_text_unet_denoiser.pt")
    
    # Save models as wandb artifacts
    for model_name in ["best_multi_objective_denoiser.pt", "final_text_unet_denoiser.pt"]:
        try:
            artifact = wandb.Artifact(f"text-unet-denoiser-{model_name.split('_')[0]}", type="model")
            artifact.add_file(model_name)
            wandb.log_artifact(artifact)
        except FileNotFoundError:
            print(f"Warning: {model_name} not found, skipping artifact upload")
    
    print("Training complete!")
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
        best_checkpoint = "best_multi_objective_denoiser.pt"
        if os.path.exists(best_checkpoint):
            print(f"🔍 Found existing checkpoint: {best_checkpoint}")
            response = input("Continue training from best checkpoint? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                main(checkpoint_path=best_checkpoint, continue_training=True)
            else:
                main()
        else:
            main() 