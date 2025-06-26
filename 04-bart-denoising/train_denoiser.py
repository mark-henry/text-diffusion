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

class TextDataset(Dataset):
    """Dataset that returns raw text and token IDs for dynamic latent computation."""
    def __init__(self, tokenizer: BartTokenizer, dataset: List[str], max_length: int):
        self.tokenizer = tokenizer
        self.dataset = [text for text in dataset if text.strip()]  # Filter empty texts
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.dataset[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        return {
            'input_ids': inputs.input_ids.squeeze(0),  # Remove batch dimension
            'attention_mask': inputs.attention_mask.squeeze(0)
        }

class BartDiffusionLM(nn.Module):
    """
    BART-based diffusion language model following Li et al. 2022 approach.
    
    Architecture:
    - Uses BART's embedding layers (TRAINABLE) to handle latent -> embedding conversion
    - Uses BART's transformer layers (TRAINABLE) for processing
    - Predicts clean latents x_0 instead of noise
    - Uses sinusoidal time embeddings for timestep conditioning
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
        
        # TRAINABLE: BART's embedding layers (now learnable)
        self.embed_tokens = encoder.embed_tokens
        self.embed_positions = encoder.embed_positions  
        self.layernorm_embedding = encoder.layernorm_embedding
        
        # Unfreeze embedding layers - make them trainable
        for param in self.embed_tokens.parameters():
            param.requires_grad = True
        for param in self.embed_positions.parameters():
            param.requires_grad = True
        for param in self.layernorm_embedding.parameters():
            param.requires_grad = True
            
        embedding_params = sum(p.numel() for p in self.embed_tokens.parameters()) + \
                          sum(p.numel() for p in self.embed_positions.parameters()) + \
                          sum(p.numel() for p in self.layernorm_embedding.parameters())
        print(f"✅ Made BART embedding layers trainable ({embedding_params:,} trainable params)")
        
        # TRAINABLE: BART's transformer layers
        self.transformer_layers = encoder.layers
        
        # Unfreeze transformer layers  
        for param in self.transformer_layers.parameters():
            param.requires_grad = True
            
        transformer_params = sum(p.numel() for p in self.transformer_layers.parameters() if p.requires_grad)
        print(f"✅ Made BART transformer layers trainable ({transformer_params:,} trainable params)")
        
        # Time embedding layers (sinusoidal encoding + MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
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
        
        # Note: We compute reconstruction loss directly using embedding similarities
        # instead of a separate linear layer, which better enforces embedding separation
    
    def compute_clean_latents(self, input_ids, attention_mask=None):
        """Compute clean latents from token IDs using BART's embedding layers."""
        batch_size, seq_len = input_ids.shape
        
        # Ensure sequence length doesn't exceed BART's limits
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length
        
        # Token embeddings (now trainable!)
        token_embeddings = self.embed_tokens(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.embed_positions(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Apply embedding normalization
        embeddings = self.layernorm_embedding(embeddings)
        
        # Create attention mask for BART transformer layers
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device, dtype=torch.bool)
        else:
            # Ensure attention mask is bool type
            attention_mask = attention_mask.bool()
        
        # Convert to 4D format for BART
        attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len).bool()
        
        # Pass through transformer layers to get latent representations
        hidden_states = embeddings
        for layer in self.transformer_layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                layer_head_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
        
        return hidden_states

    def get_learnable_embeddings(self, input_ids, attention_mask=None):
        """
        Get the learnable embeddings EMB(w) for the embedding loss component.
        This is the trainable BART embedding representation.
        """
        batch_size, seq_len = input_ids.shape
        
        # Ensure sequence length doesn't exceed BART's limits
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length
        
        # Get trainable token embeddings (this is EMB(w) in the paper)
        token_embeddings = self.embed_tokens(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.embed_positions(position_ids)
        
        # Combine embeddings and normalize
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layernorm_embedding(embeddings)
        
        return embeddings

    # predict_vocabulary_logits method removed - we now compute reconstruction loss
    # directly using cosine similarities to vocabulary embeddings

    def get_sinusoidal_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal timestep embeddings like in original Transformer paper.
        Following Diffusion-LM approach for time encoding.
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=timesteps.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # Zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, noisy_latents, timesteps):
        """
        Forward pass: predict clean latents x_0 from noisy latents x_t and timestep t.
        
        Architecture:
        1. Time embedding: sinusoidal encoding of timestep
        2. Input projection: noisy latents -> BART embedding space
        3. Time injection: add time embeddings to latent representations
        4. BART transformer: process time-conditioned latents
        5. Output projection: predict clean latents x_0
        
        Args:
            noisy_latents: [B, L, C] noisy latents at timestep t
            timesteps: [B] timestep indices
            
        Returns:
            predicted_x0: [B, L, C] predicted clean latents
        """
        batch_size, seq_len, embed_dim = noisy_latents.shape
        
        # 1. Create time embeddings using sinusoidal encoding
        time_emb = self.get_sinusoidal_embedding(timesteps, self.time_embed_dim).to(noisy_latents.device)
        time_emb = self.time_embed(time_emb)  # [B, time_embed_dim]
        
        # 2. Project time embedding to BART's embedding dimension
        time_proj = self.time_proj(time_emb)  # [B, d_model]
        time_proj = time_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, d_model]
        
        # 3. Project noisy latents to BART embedding space
        latent_proj = self.input_proj(noisy_latents)  # [B, L, d_model]
        
        # 4. Add time information to latent representations
        time_conditioned_latents = latent_proj + time_proj  # [B, L, d_model]
        
        # 5. Create attention mask (assume all positions are valid)
        attention_mask = torch.ones(batch_size, seq_len, device=noisy_latents.device, dtype=torch.bool)
        attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len).bool()
        
        # 6. Process through BART transformer layers
        hidden_states = time_conditioned_latents
        for layer in self.transformer_layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                layer_head_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
        
        # 7. Final projection to predict clean latents x_0
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
            latents = model.compute_clean_latents(input_ids, attention_mask)
            
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = scheduler.add_noise(latents.transpose(1, 2), timesteps)
            noisy_latents = noisy_latents.transpose(1, 2)  # Back to [B, L, C]
            
            predicted_x0 = model(noisy_latents, timesteps)
            
            loss_dict = loss_fn(predicted_x0, latents, noisy_latents, input_ids, attention_mask, timesteps)
            
            val_total_loss += loss_dict['total_loss'].item()
            val_cosine_sim += loss_dict['cosine_sim'].item()
            val_magnitude_ratio += loss_dict['magnitude_ratio'].item()
            num_batches += 1
            
            # Collect loss vs timestep and cosine similarity vs timestep data (sample every few batches to avoid too much data)
            if num_batches % 5 == 0:  # Sample every 5th batch
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
            "validation/loss_vs_timestep": wandb.plot.line(loss_table, "timestep", "loss", 
                                                         title="Validation Loss vs Timestep"),
            "validation/cosine_similarity_vs_timestep": wandb.plot.line(cosine_table, "timestep", "cosine_similarity", 
                                                                       title="Validation Cosine Similarity vs Timestep")
        })
    
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
        
        # COMPONENT 1: Standard diffusion loss L_simple
        # L_simple = ||f_θ(x_t, t) - x_0||²
        diffusion_loss = F.mse_loss(predicted_x0, target_x0)
        
        # COMPONENT 2: Embedding loss ||EMB(w) - μ_θ(x_1, 1)||²
        # Get learnable embeddings EMB(w) - the trainable BART embeddings
        learnable_embeddings = model.get_learnable_embeddings(input_ids, attention_mask)
        
        # For the embedding loss, we need the model's prediction from timestep 1 for timestep 0
        # This is μ_θ(x_1, 1) - the predicted x_0 from the previous step x_1
        # This helps reduce rounding errors by drawing predictions close to actual embedding positions
        timestep_1 = torch.full((batch_size,), 1, device=timesteps.device)
        x1_noisy, _ = scheduler.add_noise(target_x0.transpose(1, 2), timestep_1)
        x1_noisy = x1_noisy.transpose(1, 2)  # Back to [B, L, C]
        
        # Get model prediction from timestep 1: μ_θ(x_1, 1)
        with torch.no_grad():  # Don't backprop through this forward pass
            mu_theta_x1 = model(x1_noisy, timestep_1)
        
        # Compute embedding loss: ||EMB(w) - μ_θ(x_1, 1)||²
        # Apply attention mask to ignore padded tokens
        embedding_loss = F.mse_loss(learnable_embeddings * attention_mask.unsqueeze(-1), 
                                   mu_theta_x1.detach() * attention_mask.unsqueeze(-1))
        
        # COMPONENT 3: Reconstruction loss -log p_θ(w|x_0)
        # This enforces unambiguous prediction, which is the combination of
        # the embeddings being separated in space and each prediction being close to the correct embedding.
        # We compute distances from predicted vectors to all vocabulary embeddings
        # and use cross-entropy between the resulting probability distribution and one-hot target
        
        # Get all vocabulary embeddings [vocab_size, embed_dim]
        vocab_embeddings = model.embed_tokens.weight  # [vocab_size, 768]
        
        # Compute distances from predicted latents to all vocabulary embeddings
        # predicted_x0: [B, L, 768], vocab_embeddings: [vocab_size, 768]
        # We want to compute similarity between each predicted position and each vocab embedding
        
        batch_size, seq_len, embed_dim = predicted_x0.shape
        vocab_size = vocab_embeddings.shape[0]
        
        # Reshape predicted latents for batch computation: [B*L, embed_dim]
        pred_flat = predicted_x0.view(-1, embed_dim)  # [B*L, 768]
        
        # Creative memory-efficient approach: Combine direction (cosine) + magnitude information
        # This enforces both correct direction AND magnitude while avoiding OOM
        
        # 1. Compute cosine similarities (direction component) - memory efficient
        pred_normalized = F.normalize(pred_flat, p=2, dim=1)  # [B*L, 768]
        vocab_normalized = F.normalize(vocab_embeddings, p=2, dim=1)  # [vocab_size, 768]
        cosine_similarities = torch.mm(pred_normalized, vocab_normalized.t())  # [B*L, vocab_size]
        
        # 2. Compute magnitude factors - also memory efficient
        pred_magnitudes = torch.norm(pred_flat, p=2, dim=1, keepdim=True)  # [B*L, 1]
        vocab_magnitudes = torch.norm(vocab_embeddings, p=2, dim=1).unsqueeze(0)  # [1, vocab_size]
        
        # Clear normalized tensors from memory since we have cosine_similarities
        del pred_normalized, vocab_normalized
        
        # 3. Magnitude similarity: penalize magnitude differences using broadcasting  
        # Use negative squared ratio difference (handles different scales better than log)
        magnitude_ratios = pred_magnitudes / (vocab_magnitudes + 1e-8)  # [B*L, vocab_size]
        # Penalty for being too far from magnitude 1.0 (perfect match) - IN-PLACE
        magnitude_ratios.sub_(1.0)  # magnitude_ratios -= 1.0
        magnitude_ratios.pow_(2)    # magnitude_ratios = magnitude_ratios ** 2
        magnitude_ratios.mul_(-0.5) # magnitude_ratios *= -0.5 (now magnitude_penalty)
        
        # 4. Combine direction and magnitude: weighted sum - IN-PLACE
        # Higher weight on cosine similarity since direction is more important
        direction_weight = 0.85
        magnitude_weight = 0.15
        # Reuse cosine_similarities tensor for combined result
        cosine_similarities.mul_(direction_weight)  # cosine_similarities *= direction_weight
        magnitude_ratios.mul_(magnitude_weight)     # magnitude_ratios *= magnitude_weight  
        cosine_similarities.add_(magnitude_ratios)  # cosine_similarities += magnitude_ratios (now combined_similarities)
        
        # Apply temperature scaling to control sharpness of the distribution - IN-PLACE
        temperature = 0.07  # Lower temperature for sharper distributions
        vocab_logits = cosine_similarities.div_(temperature)  # Reuse tensor, now vocab_logits
        
        # Create one-hot target distribution
        input_ids_flat = input_ids.view(-1)  # [B*L]
        attention_mask_flat = attention_mask.view(-1)  # [B*L]
        
        # Compute cross-entropy loss between predicted distribution and one-hot target
        # This rewards being close to correct embedding and penalizes being close to incorrect ones
        reconstruction_loss = F.cross_entropy(vocab_logits, input_ids_flat, reduction='none')
        reconstruction_loss.mul_(attention_mask_flat.float())  # IN-PLACE: *= attention_mask
        reconstruction_loss = reconstruction_loss.sum() / attention_mask_flat.sum()  # Average over valid tokens
        
        # Combine all three components
        # Weight the components (following Li et al. 2022 approach)
        lambda_embed = 1.0  # Weight for embedding loss
        lambda_recon = 1.0  # Weight for reconstruction loss
        
        total_loss = diffusion_loss + lambda_embed * embedding_loss + lambda_recon * reconstruction_loss
        
        # Compute additional metrics for monitoring
        cosine_similarities = F.cosine_similarity(pred_flat_metrics, target_flat_metrics, dim=1)
        pred_magnitudes = torch.norm(pred_flat_metrics, dim=1)
        target_magnitudes = torch.norm(target_flat_metrics, dim=1)
        magnitude_ratios = pred_magnitudes / (target_magnitudes + 1e-8)
        
        # Also compute denoising quality (how well we recover from noise)
        denoising_similarity = F.cosine_similarity(pred_flat_metrics, target_flat_metrics, dim=1)
        noise_similarity = F.cosine_similarity(noisy_flat_metrics, target_flat_metrics, dim=1)
        denoising_improvement = denoising_similarity - noise_similarity
        
        # Compute reconstruction accuracy
        with torch.no_grad():
            predicted_tokens = torch.argmax(vocab_logits, dim=-1)
            correct_predictions = (predicted_tokens == input_ids_flat) & attention_mask_flat.bool()
            reconstruction_accuracy = correct_predictions.sum().float() / attention_mask_flat.sum()
        
        # Return loss and metrics for logging
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'embedding_loss': embedding_loss,
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
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = input_ids.shape[0]
            
            # Compute clean latents from token IDs (now with gradients!)
            latents = model.compute_clean_latents(input_ids, attention_mask)
            
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = scheduler.add_noise(latents.transpose(1, 2), timesteps)
            noisy_latents = noisy_latents.transpose(1, 2)  # Back to [B, L, C]
            
            # Forward pass - predict clean latents x_0
            predicted_x0 = model(noisy_latents, timesteps)
            
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
                alphas_cumprod = scheduler.alphas_cumprod.to(device)[timesteps]
                signal_level = torch.sqrt(alphas_cumprod)
                noise_level = torch.sqrt(1 - alphas_cumprod)
                snr = (signal_level / noise_level).mean()
            
            # Log batch metrics less frequently for speed
            if batch_idx % 50 == 0:
                wandb.log({
                    "train/batch_total_loss": loss.item(),
                    "train/batch_diffusion_loss": loss_dict['diffusion_loss'].item(),
                    "train/batch_embedding_loss": loss_dict['embedding_loss'].item(),
                    "train/batch_reconstruction_loss": loss_dict['reconstruction_loss'].item(),
                    "train/batch_reconstruction_accuracy": loss_dict['reconstruction_accuracy'].item(),
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
            "encoder": "BART-base (fully trainable)",
            "embeddings": "BART embeddings (trainable)",
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
            "loss_function": "li_et_al_2022_three_component",  # Full Li et al. 2022 approach
            "objective": "L_e2e_simple(w) = L_simple + ||EMB(w) - μ_θ(x_1, 1)||² - log p_θ(w|x_0)",
            "time_encoding": "sinusoidal_embeddings",
            "architecture": "bart_encoder_with_time_injection",
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
    
    # Load dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Create text dataset for dynamic latent computation
    print("Creating text dataset...")
    train_texts = []
    for item in dataset['train']:
        if isinstance(item, dict) and 'text' in item and item['text'].strip():
            train_texts.append(item['text'])
        elif isinstance(item, str) and item.strip():
            train_texts.append(item)
    
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