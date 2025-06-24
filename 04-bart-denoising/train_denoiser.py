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

class BartDiffusionLM(nn.Module):
    """
    BART-based diffusion language model following Li et al. 2022 approach.
    
    Architecture:
    - Uses BART's embedding layers (FROZEN) to handle latent -> embedding conversion
    - Uses BART's transformer layers (TRAINABLE) for processing
    - Predicts clean latents x_0 instead of noise
    - Incorporates learned time scaling function a(t)
    """
    def __init__(self, bart_model_name="facebook/bart-base", max_length=64, time_embed_dim=256, num_timesteps=2000):
        super().__init__()
        self.max_length = max_length
        self.num_timesteps = num_timesteps
        self.time_embed_dim = time_embed_dim
        
        # Load BART model and extract components
        from transformers import BartModel, BartConfig
        self.bart_config = BartConfig.from_pretrained(bart_model_name)
        bart_model = BartModel.from_pretrained(bart_model_name)
        
        # CRITICAL: Ensure max_length doesn't exceed BART's position embedding limit
        max_pos_embeds = self.bart_config.max_position_embeddings
        if max_length > max_pos_embeds:
            print(f"Warning: max_length {max_length} exceeds BART's max_position_embeddings {max_pos_embeds}")
            self.max_length = min(max_length, max_pos_embeds - 2)  # -2 for safety margin
            print(f"Adjusted max_length to {self.max_length}")
        
        # Extract BART components
        encoder = bart_model.encoder
        
        # FROZEN: BART's embedding layers (for converting latents to BART embedding space)
        self.embed_tokens = encoder.embed_tokens
        self.embed_positions = encoder.embed_positions  
        self.layernorm_embedding = encoder.layernorm_embedding
        
        # Freeze embedding layers
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.embed_positions.parameters():
            param.requires_grad = False
        for param in self.layernorm_embedding.parameters():
            param.requires_grad = False
            
        print(f"✅ Froze BART embedding layers ({sum(p.numel() for p in self.embed_tokens.parameters()):,} + {sum(p.numel() for p in self.embed_positions.parameters()):,} + {sum(p.numel() for p in self.layernorm_embedding.parameters()):,} params)")
        
        # TRAINABLE: BART's transformer layers
        self.transformer_layers = encoder.layers
        
        # Unfreeze transformer layers  
        for param in self.transformer_layers.parameters():
            param.requires_grad = True
            
        trainable_params = sum(p.numel() for p in self.transformer_layers.parameters() if p.requires_grad)
        print(f"✅ Unfroze BART transformer layers ({trainable_params:,} trainable params)")
        
        # Time embedding layers (sinusoidal encoding + MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        
        # Learned scaling function a(t) - maps time embedding to scaling factor
        self.time_scale = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim // 2, 1),
            nn.Sigmoid()  # Ensures positive scaling
        )
        
        # Project time embedding into BART's embedding space for injection
        self.time_proj = nn.Linear(time_embed_dim, self.bart_config.d_model)
        
        # Input projection layer to handle noisy latents -> BART embedding space
        self.input_proj = nn.Linear(self.bart_config.d_model, self.bart_config.d_model)
        
        # Final projection to predict clean latents (same dim as input)
        self.output_proj = nn.Sequential(
            nn.Linear(self.bart_config.d_model, self.bart_config.d_model),
            nn.LayerNorm(self.bart_config.d_model),
            nn.SiLU(),
            nn.Linear(self.bart_config.d_model, self.bart_config.d_model)  # 768 for BART-base
        )
        
    def get_sinusoidal_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal timestep embeddings like in original Transformer paper.
        Following Diffusion-LM approach for time encoding.
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=-1)
        return emb
    
    def forward(self, noisy_latents, timesteps):
        """
        Forward pass to predict clean latents x_0.
        
        Args:
            noisy_latents: [batch_size, seq_len, hidden_dim] - noisy BART latents at timestep t
            timesteps: [batch_size] - diffusion timesteps
            
        Returns:
            [batch_size, seq_len, hidden_dim] - predicted clean latents x_0
        """
        batch_size, seq_len, hidden_dim = noisy_latents.shape
        
        # Ensure sequence length doesn't exceed BART's limits
        if seq_len > self.max_length:
            print(f"Warning: input sequence length {seq_len} exceeds max_length {self.max_length}")
            noisy_latents = noisy_latents[:, :self.max_length, :]
            seq_len = self.max_length
        
        # Get sinusoidal time embeddings
        t_emb = self.get_sinusoidal_embedding(timesteps, self.time_embed_dim)
        t_emb = self.time_embed(t_emb)  # [batch_size, time_embed_dim]
        
        # Compute learned scaling function a(t)
        time_scale = self.time_scale(t_emb)  # [batch_size, 1]
        
        # Project time embedding to BART hidden dimension
        time_proj = self.time_proj(t_emb)  # [batch_size, d_model]
        time_proj = time_proj.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, d_model]
        
        # Apply learned time scaling to noisy latents
        scaled_noisy_latents = noisy_latents * time_scale.unsqueeze(-1)  # [96, 1] -> [96, 1, 1] for broadcasting
        
        # Project noisy latents and add time information
        projected_latents = self.input_proj(scaled_noisy_latents)
        embeddings = projected_latents + time_proj
        
        # Add BART's positional embeddings (using frozen embed_positions)
        # Position IDs: 0, 1, 2, ..., seq_len-1
        position_ids = torch.arange(seq_len, device=noisy_latents.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.embed_positions(position_ids)
        
        # Combine embeddings
        embeddings = embeddings + position_embeddings
        
        # Apply BART's embedding normalization (frozen)
        embeddings = self.layernorm_embedding(embeddings)
        
        # Create attention mask (all positions are valid)
        # BART expects 4D attention mask: [batch_size, 1, seq_len, seq_len]
        attention_mask = torch.ones(batch_size, seq_len, device=noisy_latents.device, dtype=torch.bool)
        # Convert to 4D causal mask format that BART expects
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
        attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
        
        # Pass through BART's transformer layers (trainable)
        hidden_states = embeddings
        for layer in self.transformer_layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,  # No head masking needed
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]  # Get hidden states from layer output
        
        # Project to predict clean latents x_0
        predicted_x0 = self.output_proj(hidden_states)
        
        return predicted_x0

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
    """Validate the model with the new Diffusion-LM loss."""
    model.eval()
    val_total_loss = 0
    val_cosine_sim = 0
    val_magnitude_ratio = 0
    num_batches = 0
    
    with torch.no_grad():
        for latents in val_loader:
            latents = latents.to(device)
            batch_size = latents.shape[0]
            # latents are [B, L, C] - keep this format for BART
            
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = scheduler.add_noise(latents.transpose(1, 2), timesteps)
            noisy_latents = noisy_latents.transpose(1, 2)  # Back to [B, L, C]
            
            predicted_x0 = model(noisy_latents, timesteps)
            
            loss_dict = loss_fn(predicted_x0, latents, noisy_latents)
            
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
    Train the denoising model using Diffusion-LM approach.
    Key change: predict clean latents x_0 instead of noise.
    Returns:
        Dictionary containing training history.
    """
    
    def diffusion_lm_loss(predicted_x0, target_x0, noisy_latents):
        """
        Diffusion-LM loss function: predict clean latents x_0 directly.
        
        L = ||f_θ(x_t, t) - x_0||²
        
        Args:
            predicted_x0: Model predictions [B, L, C] - predicted clean latents
            target_x0: Ground truth clean latents [B, L, C]
            noisy_latents: Current noisy latents [B, L, C] (for metrics)
        """
        batch_size = predicted_x0.shape[0]
        
        # Flatten for computing similarities and magnitudes  
        pred_flat = predicted_x0.reshape(batch_size, -1)
        target_flat = target_x0.reshape(batch_size, -1)
        noisy_flat = noisy_latents.reshape(batch_size, -1)
        
        # Main loss: L2 distance to clean latents (Diffusion-LM objective)
        l2_loss = F.mse_loss(predicted_x0, target_x0)
        
        # Compute additional metrics for monitoring
        cosine_similarities = F.cosine_similarity(pred_flat, target_flat, dim=1)
        pred_magnitudes = torch.norm(pred_flat, dim=1)
        target_magnitudes = torch.norm(target_flat, dim=1)
        magnitude_ratios = pred_magnitudes / (target_magnitudes + 1e-8)
        
        # Also compute denoising quality (how well we recover from noise)
        denoising_similarity = F.cosine_similarity(pred_flat, target_flat, dim=1)
        noise_similarity = F.cosine_similarity(noisy_flat, target_flat, dim=1)
        denoising_improvement = denoising_similarity - noise_similarity
        
        # Also compute other losses for comparison
        l1_loss = F.l1_loss(predicted_x0, target_x0)
        
        # Return loss and metrics for logging
        return {
            'total_loss': l2_loss,
            'l2_loss': l2_loss,
            'l1_loss': l1_loss,
            'cosine_sim': cosine_similarities.mean(),
            'magnitude_ratio': magnitude_ratios.mean(),
            'pred_magnitude_mean': pred_magnitudes.mean(),
            'target_magnitude_mean': target_magnitudes.mean(),
            'magnitude_ratio_std': magnitude_ratios.std(),
            'cosine_sim_std': cosine_similarities.std(),
            'denoising_improvement': denoising_improvement.mean(),
            'noise_similarity': noise_similarity.mean()
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
    print(f"📊 Using x_0 prediction objective (Diffusion-LM approach)")
    
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
        
        for batch_idx, latents in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")):
            latents = latents.to(device)
            batch_size = latents.shape[0]
            # latents are [B, L, C] - keep this format for BART
            
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = scheduler.add_noise(latents.transpose(1, 2), timesteps)
            noisy_latents = noisy_latents.transpose(1, 2)  # Back to [B, L, C]
            
            # Forward pass - predict clean latents x_0
            predicted_x0 = model(noisy_latents, timesteps)
            
            loss_dict = diffusion_lm_loss(predicted_x0, latents, noisy_latents)
            
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
                alphas_cumprod = scheduler.alphas_cumprod.to(device)[timesteps]
                signal_level = torch.sqrt(alphas_cumprod)
                noise_level = torch.sqrt(1 - alphas_cumprod)
                snr = (signal_level / noise_level).mean()
            
            # Log batch metrics less frequently for speed
            if batch_idx % 50 == 0:
                wandb.log({
                    "train/batch_total_loss": loss.item(),
                    "train/batch_l2_loss": loss_dict['l2_loss'].item(),
                    "train/batch_l1_loss": loss_dict['l1_loss'].item(),
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
                })
        
        # Validation
        val_loss, val_cosine_sim, val_magnitude_ratio = validate_model(model, val_loader, scheduler, diffusion_lm_loss, device)
        
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
    run_name = "diffusion-lm-bart-v1-x0-prediction"
    
    # Initialize wandb
    wandb.init(
        project="text-diffusion",
        name=run_name,
        config={
            "model_type": "BartDiffusionLM",
            "approach": "Diffusion-LM (Li et al. 2022)",
            "encoder": "BART-base (trainable)",
            "dataset": "WikiText-2",
            "batch_size": 96,
            "learning_rate": 5e-5,  # Lower for pretrained BART
            "num_epochs": 40,
            "max_length": 64,  # Reduced from 128
            "noise_scheduler": "cosine",
            "num_timesteps": 2000,
            "s": 0.008,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "loss_function": "x0_prediction",  # Key difference: predict clean latents
            "objective": "L_x0 = ||f_θ(x_t, t) - x_0||²",
            "time_encoding": "sinusoidal_with_learned_scaling",
            "architecture": "bart_encoder_with_time_injection",
            "time_embed_dim": 256,
            "learned_time_scaling": True,
            "attention_mechanisms": "BART_bidirectional",
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
    try:
        # Try normal indexing first
        train_texts = dataset['train']['text']
    except (KeyError, TypeError):
        # If that fails, try iteration
        train_texts = [item['text'] for item in dataset['train']]
    
    full_dataset = LatentDataset(bart_model, tokenizer, train_texts, max_length=64)
    
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
    
    # Create text-specific BART diffusion model
    print("Creating BartDiffusionLM model for BART semantic latents...")
    model = BartDiffusionLM(bart_model_name="facebook/bart-base", max_length=64, time_embed_dim=256, num_timesteps=2000)
    
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
    
    # Create noise scheduler with cosine schedule
    scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
    
    # Train model with Diffusion-LM approach
    if continue_training:
        print(f"🔄 Continuing Diffusion-LM training from epoch {start_epoch + 1}...")
    else:
        print("🚀 Training Diffusion-LM model with x_0 prediction objective...")
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
    torch.save(model.state_dict(), "final_diffusion_lm_model.pt")
    
    # Save models as wandb artifacts
    for model_name in ["best_diffusion_lm_denoiser.pt", "final_diffusion_lm_model.pt"]:
        try:
            artifact = wandb.Artifact(f"diffusion-lm-{model_name.split('_')[0]}", type="model")
            artifact.add_file(model_name)
            wandb.log_artifact(artifact)
        except FileNotFoundError:
            print(f"Warning: {model_name} not found, skipping artifact upload")
    
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