import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Union
import torch.nn.functional as F

# Import encoder abstractions
from encoders import BaseEncoder, BartEncoder, BertEncoder


class CosineNoiseScheduler:
    def __init__(self, num_timesteps: int = 2000, s: float = 1e-4):
        self.num_timesteps = num_timesteps
        self.s = s
        
        # Compute alphas_cumprod using cosine schedule: α¯t = cos(((t/T) + s) / (1 + s) * π/2)²
        t = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1
        
        # Ensure no negative values and proper bounds
        alphas_cumprod = torch.clamp(alphas_cumprod[1:], min=1e-8, max=1.0)  # Remove first element
        
        # Compute posterior quantities following DDPM
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # Compute alphas and betas
        self.alphas = alphas_cumprod / self.alphas_cumprod_prev
        self.betas = 1 - self.alphas
        
        # Compute posterior variance (clipped to prevent numerical issues)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clip the first value to avoid log(0)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        
    def add_noise(self, latents: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to latents according to timesteps."""
        noise = torch.randn_like(latents)
        # Move alphas_cumprod to same device as timesteps
        alphas_cumprod = self.alphas_cumprod.to(timesteps.device)[timesteps]
        
        # Reshape for broadcasting with [B, C, L] format
        alphas_cumprod = alphas_cumprod.view(-1, 1, 1)
        noisy_latents = torch.sqrt(alphas_cumprod) * latents + torch.sqrt(1 - alphas_cumprod) * noise
        return noisy_latents, noise


class SqrtNoiseScheduler:
    """
    Square root noise schedule from Diffusion-LM paper.
    
    Defined by: α̅_t = 1 - √(t/T + s)
    
    This schedule:
    - Starts with higher noise level than cosine/linear schedules
    - Increases noise rapidly for the first ~50 steps
    - Then slows down to avoid spending too many steps on very high-noise problems
    - Better suited for discrete text data where small noise doesn't change nearest neighbors
    """
    def __init__(self, num_timesteps: int = 2000, s: float = 1e-4):
        self.num_timesteps = num_timesteps
        self.s = s
        
        # Compute alphas_cumprod using sqrt schedule: α̅_t = 1 - √(t/T + s)
        # Note: t goes from 0 to num_timesteps-1 for the actual timesteps
        t = torch.arange(0, num_timesteps, dtype=torch.float32)
        alphas_cumprod = 1 - torch.sqrt((t / num_timesteps) + s)
        
        # Ensure no negative values and proper bounds
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8, max=1.0)
        
        # At t=0: α̅_0 = 1 - √(s) ≈ 1 - 0.01 = 0.99 (with s=1e-4)
        # This gives initial std dev of √(1-0.99) = 0.1 as intended
        
        # Compute posterior quantities following DDPM
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # Compute alphas and betas
        self.alphas = alphas_cumprod / self.alphas_cumprod_prev
        self.betas = 1 - self.alphas
        
        # Compute posterior variance (clipped to prevent numerical issues)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clip the first value to avoid log(0)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        
    def add_noise(self, latents: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to latents according to timesteps."""
        noise = torch.randn_like(latents)
        # Move alphas_cumprod to same device as timesteps
        alphas_cumprod = self.alphas_cumprod.to(timesteps.device)[timesteps]
        
        # Reshape for broadcasting with [B, C, L] format
        alphas_cumprod = alphas_cumprod.view(-1, 1, 1)
        noisy_latents = torch.sqrt(alphas_cumprod) * latents + torch.sqrt(1 - alphas_cumprod) * noise
        return noisy_latents, noise


def pad_tensor(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad or truncate tensor to target length"""
    current_length = tensor.size(0)
    if current_length < target_length:
        padding = torch.zeros(target_length - current_length, tensor.size(1), 
                            dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=0)
    else:
        return tensor[:target_length]


class TextDataset(Dataset):
    """Dataset that returns raw text and token IDs for dynamic latent computation."""
    def __init__(self, tokenizer, dataset: List[str], max_length: int, vocab_size: Optional[int] = None):
        self.tokenizer = tokenizer
        self.dataset = [text for text in dataset if text.strip()]  # Filter empty texts
        self.max_length = max_length
        self.vocab_size = vocab_size
        # Get UNK token ID for replacement
        unk_id = tokenizer.unk_token_id
        if unk_id is not None and isinstance(unk_id, int):
            self.unk_token_id = unk_id
        else:
            raise ValueError(f"UNK token ID is not set for tokenizer {tokenizer}")
        
        if vocab_size is not None:
            print(f"📝 TextDataset: Filtering vocabulary to {vocab_size:,} tokens (UNK token: {self.unk_token_id})")

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.dataset[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length", add_special_tokens=False)
        
        input_ids = inputs.input_ids.squeeze(0)  # Remove batch dimension
        attention_mask = inputs.attention_mask.squeeze(0)
        
        # Filter vocabulary if vocab_size is specified
        if self.vocab_size is not None:
            # Replace any token ID >= vocab_size with UNK token
            out_of_vocab_mask = input_ids >= self.vocab_size
            input_ids = torch.where(out_of_vocab_mask, self.unk_token_id, input_ids)
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


class DiffusionLM(nn.Module):
    """
    Unified diffusion language model following Li et al. 2022 approach.
    
    Architecture:
    - Uses encoder to handle latent -> embedding conversion
    - Uses encoder's transformer layers (TRAINABLE) for processing
    - Predicts clean latents x_0 instead of noise
    - Uses sinusoidal time embeddings for timestep conditioning
    """
    def __init__(self, encoder_type: str = "bart", model_name: Optional[str] = None, max_length: int = 64, 
                 time_embed_dim: Optional[int] = None, num_timesteps: int = 2000, scheduler=None, 
                 dropout: float = 0.1, custom_config: Optional[Union[Dict, object]] = None):
        super().__init__()
        self.max_length = max_length
        self.num_timesteps = num_timesteps
        self.dropout_prob = dropout
        self.encoder_type = encoder_type.lower()
        
        # Initialize noise scheduler - use sqrt schedule by default as recommended in paper
        if scheduler is None:
            self._scheduler = SqrtNoiseScheduler(num_timesteps=num_timesteps)
        else:
            self._scheduler = scheduler
        
        # Create the appropriate encoder
        if self.encoder_type == "bart":
            default_model_name = "facebook/bart-base"
            self.encoder = BartEncoder(
                model_name=model_name or default_model_name,
                max_length=max_length,
                dropout=dropout,
                custom_config=custom_config
            )
        elif self.encoder_type == "bert":
            default_model_name = "bert-base-uncased"
            self.encoder = BertEncoder(
                model_name=model_name or default_model_name,
                max_length=max_length,
                dropout=dropout,
                custom_config=custom_config
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}. Must be 'bart' or 'bert'")

        # Set dimensions from encoder
        self.latent_dim = self.encoder.get_latent_dim()
        self.max_length = self.encoder.max_length  # Use encoder's adjusted max_length
        
        # Set time_embed_dim after we have encoder
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else self.latent_dim
        
        # dropout layer 
        self.dropout = nn.Dropout(dropout)
        
        # Time embedding layers
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.latent_dim),
        )
        
    @property
    def scheduler(self):
        """Access the noise scheduler used by this model."""
        return self._scheduler
    
    @property 
    def config(self):
        """Access the encoder's config for backward compatibility."""
        return self.encoder.config
        
    def embed_tokens(self, input_ids, attention_mask=None) -> torch.Tensor:
        """
        Compute embeddings from token IDs using the encoder.
        """
        return self.encoder.embed_tokens(input_ids, attention_mask)

    def get_vocab_logits(self, hidden_states):
        """
        Convert hidden states to vocabulary logits using weight tying.
        """
        return self.encoder.get_vocab_logits(hidden_states)

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
        
    def forward(self, noisy_latents, timesteps, attention_mask=None):
        """
        Forward pass: predict clean latents x_0 from noisy latents x_t and timestep t.
        
        Architecture (following Diffusion-LM reference):
        1. Input up-projection: noisy latents -> projected space  
        2. Time embedding: sinusoidal encoding
        3. Combine: projected inputs + time conditioning
        4. Layer norm and dropout
        5. Encoder transformer: process time-conditioned latents (handles position embeddings internally)
        6. Output down-projection: transformer output -> latent space

        NOTE that positional encodings are automatically added by the encoder.
        
        Args:
            noisy_latents: [B, L, C] noisy latents at timestep t (already in encoder embedding space)
            timesteps: [B] timestep indices
            attention_mask: [B, L] optional attention mask for valid positions
            
        Returns:
            predicted_x0: [B, L, C] predicted clean latents (in encoder embedding space)
        """
        batch_size, seq_len, embed_dim = noisy_latents.shape
        
        # Verify we're working in the correct embedding dimension
        assert embed_dim == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {embed_dim}"
        
        # 1. Use noisy latents directly (no projection needed)
        proj_inputs = noisy_latents
        
        # 2. Create time embeddings using sinusoidal encoding
        time_emb = self.get_sinusoidal_embedding(timesteps, self.time_embed_dim).to(noisy_latents.device)
        time_emb = self.time_embed(time_emb)  # [B, latent_dim]
        # Broadcast time embedding
        time_proj = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, latent_dim]
        
        # 3. Combine projected inputs with time conditioning
        emb_inputs = proj_inputs + time_proj  # [B, L, latent_dim]
        
        # 4. Don't apply layer normalization, dropout or positional embeddings—encoder handles it internally
        
        # 5. Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=noisy_latents.device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
            
        # 6. Process through encoder
        predicted_x0 = self.encoder.forward_encoder(emb_inputs, attention_mask)
        
        return predicted_x0


# Legacy class names for backward compatibility
BartDiffusionLM = DiffusionLM  # Backward compatibility alias
BertDiffusionLM = DiffusionLM  # Backward compatibility alias


def encode_text_to_latents(text, model: DiffusionLM, tokenizer, device, max_length=64):
    """Encode text to embeddings"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                      truncation=True, padding="max_length", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        latents = model.embed_tokens(inputs['input_ids'], inputs['attention_mask'])
    
    return latents


def decode_latents_to_text(latents, diffusion_model, tokenizer, attention_mask=None):
    """
    Decode latents to text using the clamping trick.
    
    Args:
        latents: [B, L, C] latent representations
        diffusion_model: Model with embedding weights  
        tokenizer: Tokenizer
        attention_mask: [B, L] mask for valid positions
    
    Returns:
        text: Decoded text string
    """
    with torch.no_grad():
        clamped_latents, token_ids = clamp_to_embeddings(
            latents, diffusion_model, attention_mask
        )
        
        # Handle different tokenizer types
        if hasattr(tokenizer, 'decode'):
            if diffusion_model.encoder_type == "bart":
                # Don't skip special tokens so we can see UNK tokens
                text = tokenizer.decode(token_ids[0], skip_special_tokens=False)
                
                # Clean up the text by removing padding tokens and start/end tokens
                # but keep UNK tokens visible
                text = text.replace("<pad>", "").replace("<s>", "").replace("</s>", "")
                text = text.replace("<unk>", " <unk> ")  # Make UNK tokens more visible
                
                # Clean up extra spaces
                text = " ".join(text.split())
            else:  # BERT
                text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        else:
            text = str(token_ids[0].tolist())  # Fallback
    
    return text


def token_discrete_loss(predicted_x0, model, input_ids, attention_mask=None):
    """
    Simple discrete token reconstruction loss following the reference implementation.
    
    Args:
        predicted_x0: [B, L, C] predicted clean latents
        model: Model with get_vocab_logits method
        input_ids: [B, L] target token IDs
        attention_mask: [B, L] mask for valid positions
        
    Returns:
        reconstruction_loss: scalar loss value
        reconstruction_accuracy: accuracy metric
    """
    # Convert predicted latents to vocabulary logits
    logits = model.get_vocab_logits(predicted_x0)  # [B, L, vocab_size]
    
    # Compute cross-entropy loss
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    decoder_nll = loss_fct(
        logits.view(-1, logits.size(-1)),  # [B*L, vocab_size]
        input_ids.view(-1)                 # [B*L]
    ).view(input_ids.shape)  # [B, L]
    
    # Apply attention mask if provided
    if attention_mask is not None:
        decoder_nll = decoder_nll * attention_mask.float()
        reconstruction_loss = decoder_nll.sum() / attention_mask.sum()
    else:
        reconstruction_loss = decoder_nll.mean()
    
    # Compute accuracy
    with torch.no_grad():
        predicted_tokens = torch.argmax(logits, dim=-1)  # [B, L]
        if attention_mask is not None:
            correct_predictions = (predicted_tokens == input_ids) & attention_mask.bool()
            reconstruction_accuracy = correct_predictions.sum().float() / attention_mask.sum()
        else:
            correct_predictions = (predicted_tokens == input_ids)
            reconstruction_accuracy = correct_predictions.float().mean()
    
    return reconstruction_loss, reconstruction_accuracy


def get_substantial_texts_from_dataset(dataset, min_length=100, max_samples=None):
    """Extract substantial text passages from a dataset, filtering out headers and short texts."""
    substantial_texts = []
    for item in dataset:
        if isinstance(item, dict) and "text" in item:
            text = item["text"].strip()
            if (isinstance(text, str) and len(text) >= min_length and 
                not text.startswith("=") and text):
                substantial_texts.append(text)
                if max_samples and len(substantial_texts) >= max_samples:
                    break
    return substantial_texts


def compute_l2_distances(pred_flat, vocab_embeddings):
    """
    Compute L2 distances between predicted latents and vocabulary embeddings.
    
    Uses efficient computation: ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩
    
    Args:
        pred_flat: [B*L, embed_dim] flattened predicted latents
        vocab_embeddings: [vocab_size, embed_dim] vocabulary embeddings
        
    Returns:
        distances_sq: [B*L, vocab_size] squared L2 distances
    """
    # Compute squared norms
    vocab_norm_sq = (vocab_embeddings ** 2).sum(-1).view(-1, 1)  # [vocab_size, 1]
    pred_norm_sq = (pred_flat ** 2).sum(-1).view(-1, 1)  # [B*L, 1]
    
    # Compute dot products
    dot_products = torch.mm(pred_flat, vocab_embeddings.t())  # [B*L, vocab_size]
    
    # Compute L2 distances squared: ||pred - vocab||²
    distances_sq = vocab_norm_sq.t() + pred_norm_sq - 2.0 * dot_products  # [B*L, vocab_size]
    
    # Clamp distances for numerical stability
    distances_sq = torch.clamp(distances_sq, 0.0, float('inf'))
    
    return distances_sq


def compute_cosine_distances(pred_flat, vocab_embeddings):
    """
    Compute cosine distances between predicted latents and vocabulary embeddings.
    
    Args:
        pred_flat: [B*L, embed_dim] flattened predicted latents
        vocab_embeddings: [vocab_size, embed_dim] vocabulary embeddings

    Returns:
        distances: [B*L, vocab_size] cosine distances
    """
    # Compute dot products
    dot_products = torch.mm(pred_flat, vocab_embeddings.t())  # [B*L, vocab_size]
    
    # Compute cosine distances
    distances = 1.0 - dot_products  # [B*L, vocab_size]
    
    return distances

def clamp_to_embeddings(predicted_x0, diffusion_model, attention_mask=None):
    """
    Clamp predicted latents to nearest word embeddings using cosine similarity.
    
    Using cosine similarity instead of L2 distance to handle magnitude differences
    between processed hidden states and raw embedding weights.
    
    Args:
        predicted_x0: [B, L, C] predicted clean latents
        diffusion_model: Model with embedding weights
        attention_mask: [B, L] mask for valid positions
    
    Returns:
        clamped_x0: [B, L, C] latents clamped to nearest embeddings
        token_ids: [B, L] token IDs corresponding to clamped embeddings
    """
    batch_size, seq_len, embed_dim = predicted_x0.shape
    
    # Get vocabulary embeddings [vocab_size, embed_dim]
    vocab_embeddings = diffusion_model.encoder.get_embedding_weights()
    
    # Reshape for batch computation
    pred_flat = predicted_x0.view(-1, embed_dim)  # [B*L, embed_dim]
    
    # Use cosine similarity instead of L2 distance
    distances = compute_cosine_distances(pred_flat, vocab_embeddings)
    
    # Find nearest embedding (minimum distance)
    nearest_indices = torch.argmin(distances, dim=1)  # [B*L]
    
    # Get the corresponding embeddings
    clamped_flat = vocab_embeddings[nearest_indices]  # [B*L, embed_dim]
    
    # Reshape back to original shape
    clamped_x0 = clamped_flat.view(batch_size, seq_len, embed_dim)
    token_ids = nearest_indices.view(batch_size, seq_len)  # [B, L]
    
    # Apply attention mask if provided (keep original values for padded positions)
    if attention_mask is not None:
        mask = attention_mask.bool()
        # For embeddings
        mask_3d = mask.unsqueeze(-1).expand_as(predicted_x0)  # [B, L, C]
        clamped_x0 = torch.where(mask_3d, clamped_x0, predicted_x0)
        # For token IDs (use pad token for masked positions)  
        pad_token_id = diffusion_model.encoder.get_pad_token_id()
        token_ids = torch.where(mask, token_ids, torch.tensor(pad_token_id, device=token_ids.device))
    
    return clamped_x0, token_ids


def diffusion_sample_step(xt, t, diffusion_model, device, use_clamping=True, attention_mask=None):
    """
    Single sampling step using proper DDPM posterior sampling.
    
    Implements stable DDPM sampling following the reference implementation:
    1. Predict x0 using the model
    2. Use precomputed posterior coefficients for stability  
    3. Sample from posterior with clipped variance
    
    Args:
        xt: [B, L, C] noisy latents at timestep t
        t: timestep (integer) - discrete timestep index in range [0, num_timesteps-1]
        diffusion_model: Trained diffusion model
        device: Device to run on
        use_clamping: Whether to apply clamping trick
        attention_mask: [B, L] mask for valid positions
    
    Returns:
        xt_minus_1: [B, L, C] latents at timestep t-1
        predicted_x0: [B, L, C] model's prediction of clean latents
    """
    # Ensure we have proper tensor shapes
    if isinstance(t, int):
        timesteps_tensor = torch.tensor([t], device=device)
    else:
        timesteps_tensor = t.to(device) if not t.device == device else t
    
    with torch.no_grad():
        # 1. Predict x0 using the model
        predicted_x0 = diffusion_model(xt, timesteps_tensor)
        
        # 2. Apply clamping trick if requested
        if use_clamping:
            predicted_x0, _ = clamp_to_embeddings(predicted_x0, diffusion_model, attention_mask)
        
        # 3. Handle the case where t=0 (already at x0)
        if t == 0:
            return predicted_x0, predicted_x0
        
        # 4. Use precomputed posterior coefficients for stability
        scheduler = diffusion_model.scheduler
        
        # Get precomputed values (move to device)
        alphas_cumprod = scheduler.alphas_cumprod[t].to(device)
        alphas_cumprod_prev = scheduler.alphas_cumprod_prev[t].to(device)
        betas = scheduler.betas[t].to(device)
        posterior_variance = scheduler.posterior_variance[t].to(device)
        
        # Compute posterior mean coefficients (following DDPM paper exactly)
        # coef1 = β_t * √ᾱ_{t-1} / (1 - ᾱ_t)
        # coef2 = (1 - ᾱ_{t-1}) * √α_t / (1 - ᾱ_t)
        coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(scheduler.alphas[t].to(device)) / (1.0 - alphas_cumprod)
        
        # Compute posterior mean: μ = coef1 * x_0 + coef2 * x_t
        posterior_mean = coef1 * predicted_x0 + coef2 * xt
        
        # Use precomputed clipped variance for stability
        posterior_std = torch.sqrt(posterior_variance)
        
        # 5. Sample from posterior: x_{t-1} ~ N(μ, σ²)
        # Only add noise if t > 0
        if t > 0:
            noise = torch.randn_like(xt)
            xt_minus_1 = posterior_mean + posterior_std * noise
        else:
            xt_minus_1 = posterior_mean
        
        return xt_minus_1, predicted_x0


def demo_denoising_step(example_dict, diffusion_model, tokenizer, device, timestep=1, max_length=64):
    """
    Demonstrate a single denoising step: example -> add noise -> denoise -> text
    
    Args:
        example_dict: Input example dict with 'input_ids', 'attention_mask', and optionally 'original_text'
        diffusion_model: Trained diffusion model
        tokenizer: Tokenizer
        device: Device to run on
        timestep: Noise level (integer) - discrete timestep index in range [0, num_timesteps-1]
                 Lower values = less noise (e.g., timestep=0 is minimal noise)
        max_length: Maximum sequence length (ignored, uses example's length)
    
    Returns:
        Dict with original_text, noisy_text_decoded, denoised_text, noise_percentage
    """
    inputs = {
        'input_ids': example_dict['input_ids'].unsqueeze(0).to(device),  # Add batch dimension
        'attention_mask': example_dict['attention_mask'].unsqueeze(0).to(device)
    }
    
    with torch.no_grad():
        # Get clean latents
        clean_latents = diffusion_model.embed_tokens(
            inputs['input_ids'], inputs['attention_mask']
        )
        
        # Add noise
        timesteps_tensor = torch.tensor([timestep], device=device)
        if timestep == 0:
            noisy_latents = clean_latents
        else:
            noisy_latents, _ = diffusion_model.scheduler.add_noise(
                clean_latents.transpose(1, 2), timesteps_tensor
            )
            noisy_latents = noisy_latents.transpose(1, 2)
        
        # Denoise
        predicted_clean_latents = diffusion_model(noisy_latents, timesteps_tensor)
        
        # Decode noisy latents to text (to show what noise looks like)
        noisy_text = decode_latents_to_text(noisy_latents, diffusion_model, tokenizer, inputs['attention_mask'])
        
        # Decode denoised latents to text using clamping
        denoised_text = decode_latents_to_text(predicted_clean_latents, diffusion_model, tokenizer, inputs['attention_mask'])
        
        # Calculate noise percentage
        noise_percentage = (1 - diffusion_model.scheduler.alphas_cumprod[timestep].item()) * 100 if timestep > 0 else 0.0
        
        # Compute similarity metrics
        pred_flat = predicted_clean_latents.reshape(1, -1)
        target_flat = clean_latents.reshape(1, -1)
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).item()
        
        return {
            'original_text': tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True),
            'noisy_text': noisy_text,
            'denoised_text': denoised_text,
            'noise_percentage': noise_percentage,
            'cosine_similarity': cosine_sim,
            'timestep': timestep
        }
