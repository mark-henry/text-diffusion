import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BartTokenizer, BartModel, BartConfig
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F


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
    def __init__(self, tokenizer: BartTokenizer, dataset: List[str], max_length: int, vocab_size: Optional[int] = None):
        self.tokenizer = tokenizer
        self.dataset = [text for text in dataset if text.strip()]  # Filter empty texts
        self.max_length = max_length
        self.vocab_size = vocab_size
        # Get UNK token ID for replacement
        unk_id = tokenizer.unk_token_id
        if unk_id is not None and isinstance(unk_id, int):
            self.unk_token_id = unk_id
        else:
            self.unk_token_id = 3  # BART's default UNK token ID
        
        if vocab_size is not None:
            print(f"📝 TextDataset: Filtering vocabulary to {vocab_size:,} tokens (UNK token: {self.unk_token_id})")

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.dataset[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        
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


class BartDiffusionLM(nn.Module):
    """
    BART-based diffusion language model following Li et al. 2022 approach.
    
    Architecture:
    - Uses BART's embedding layers (TRAINABLE) to handle latent -> embedding conversion
    - Uses BART's transformer layers (TRAINABLE) for processing
    - Predicts clean latents x_0 instead of noise
    - Uses sinusoidal time embeddings for timestep conditioning
    """
    def __init__(self, bart_model_name="facebook/bart-base", max_length=64, time_embed_dim=None, num_timesteps=2000, scheduler=None, dropout=0.1, custom_config: Optional[BartConfig] = None):
        super().__init__()
        self.max_length = max_length
        self.num_timesteps = num_timesteps
        self.dropout_prob = dropout
        
        # Initialize noise scheduler - use sqrt schedule by default as recommended in paper
        if scheduler is None:
            self._scheduler = SqrtNoiseScheduler(num_timesteps=num_timesteps)
        else:
            self._scheduler = scheduler
        
        # Load BART model and extract components
        if custom_config is not None:
            self.bart_config = custom_config
        else:
            self.bart_config = BartConfig.from_pretrained(bart_model_name)
        
        # Set time_embed_dim after we have bart_config
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else self.bart_config.d_model
        
        # Configure BART's dropout to match our dropout setting
        self.bart_config.hidden_dropout_prob = dropout
        self.bart_config.attention_probs_dropout_prob = dropout
        
        # Load or create BART model
        if custom_config is not None:
            # Create new model with custom config (randomly initialized)
            print("   Creating new model with random initialization...")
            self.bart_model = BartModel(config=custom_config)
        else:
            # Load pretrained model
            self.bart_model = BartModel.from_pretrained(bart_model_name, config=self.bart_config)

        self.latent_dim = self.bart_config.d_model
        
        # Ensure max_length doesn't exceed BART's position embedding limit
        max_pos_embeds = self.bart_config.max_position_embeddings
        if max_length > max_pos_embeds:
            print(f"Warning: max_length {max_length} exceeds BART's max_position_embeddings {max_pos_embeds}")
            self.max_length = min(max_length, max_pos_embeds - 2)  # -2 for safety margin
            print(f"Adjusted max_length to {self.max_length}")
        
        # Extract BART components
        encoder = self.bart_model.encoder
        # Remove decoder weights to save memory - we only need encoder
        del self.bart_model.decoder
        
        # TRAINABLE: BART's embedding layers (now learnable)
        self.embed_positions = encoder.embed_positions  
        self.layernorm_embedding = encoder.layernorm_embedding
        
        # Unfreeze embedding layers - make them trainable
        for param in encoder.embed_tokens.parameters():
            param.requires_grad = True
        for param in self.embed_positions.parameters():
            param.requires_grad = True
        for param in self.layernorm_embedding.parameters():
            param.requires_grad = True
            
        embedding_params = sum(p.numel() for p in encoder.embed_tokens.parameters()) + \
                          sum(p.numel() for p in encoder.embed_positions.parameters()) + \
                          sum(p.numel() for p in encoder.layernorm_embedding.parameters())
        print(f"✅ Made BART embedding layers trainable ({embedding_params:,} trainable params)")
        
        # TRAINABLE: unfreeze BART's transformer layers
        self.transformer_layers = encoder.layers
        for param in self.transformer_layers.parameters():
            param.requires_grad = True
            
        transformer_params = sum(p.numel() for p in self.transformer_layers.parameters() if p.requires_grad)
        print(f"✅ Made BART transformer layers trainable ({transformer_params:,} trainable params)")
        
        # dropout layer 
        self.dropout = nn.Dropout(dropout)
        
        # Time embedding layers
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.bart_config.d_model),
        )
        
    @property
    def scheduler(self):
        """Access the noise scheduler used by this model."""
        return self._scheduler
        
    def embed_tokens(self, input_ids, attention_mask=None) -> torch.Tensor:
        """
        Compute embeddings from token IDs using BART embeddings.
        """
        batch_size, seq_len = input_ids.shape
        
        # Ensure sequence length doesn't exceed BART's limits
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length
        
        # Get raw token embeddings WITHOUT positional encodings
        # This gives us pure content embeddings as clean latents
        token_embeddings = self.bart_model.get_encoder().embed_tokens(input_ids)
        
        return token_embeddings

    def get_vocab_logits(self, hidden_states):
        """
        Convert hidden states to vocabulary logits using weight tying.
        
        This implements weight tying by using the same weight matrix for both
        embedding lookup and output projection.
        
        Args:
            hidden_states: [B, L, C] hidden representations
            
        Returns:
            logits: [B, L, vocab_size] vocabulary logits
        """
        # Weight tying: use embedding weights as output projection
        # This is the key mechanism that prevents embedding collapse
        embed_weight = self.bart_model.get_encoder().embed_tokens.weight
        assert isinstance(embed_weight, torch.Tensor), "Expected embedding weight to be a tensor"
        return F.linear(hidden_states, embed_weight)

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
        5. BART transformer: process time-conditioned latents (handles position embeddings internally)
        6. Output down-projection: transformer output -> latent space

        NOTE that positional encodings are automatically added by the BART encoder.
        
        Args:
            noisy_latents: [B, L, C] noisy latents at timestep t (already in BART embedding space)
            timesteps: [B] timestep indices
            attention_mask: [B, L] optional attention mask for valid positions
            
        Returns:
            predicted_x0: [B, L, C] predicted clean latents (in BART embedding space)
        """
        batch_size, seq_len, embed_dim = noisy_latents.shape
        
        # Verify we're working in the correct embedding dimension
        assert embed_dim == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {embed_dim}"
        
        # 1. Use noisy latents directly (no projection needed)
        proj_inputs = noisy_latents
        
        # 2. Create time embeddings using sinusoidal encoding
        time_emb = self.get_sinusoidal_embedding(timesteps, self.time_embed_dim).to(noisy_latents.device)
        time_emb = self.time_embed(time_emb)  # [B, d_model]
        # Broadcast time embedding
        time_proj = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, d_model]
        
        # 3. Combine projected inputs with time conditioning
        emb_inputs = proj_inputs + time_proj  # [B, L, d_model]
        
        # 4. Don't apply layer normalization, dropout or positional embeddings—BART handles it internally
        
        # 5. Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=noisy_latents.device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
            
        # 6. Process through BART encoder
        encoder_outputs = self.bart_model.encoder(
            inputs_embeds=emb_inputs,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        # 7. Use BART output directly (no projection needed)
        predicted_x0 = encoder_outputs.last_hidden_state
        
        return predicted_x0


def decode_latents_to_text(latents, diffusion_model, tokenizer, attention_mask=None):
    """
    Decode latents to text using the clamping trick.
    
    Args:
        latents: [B, L, C] latent representations
        diffusion_model: Model with embedding weights  
        tokenizer: BART tokenizer
        device: Device to run on
        attention_mask: [B, L] mask for valid positions
    
    Returns:
        text: Decoded text string
    """
    with torch.no_grad():
        clamped_latents, token_ids = clamp_to_embeddings(
            latents, diffusion_model, attention_mask
        )
        
        # Don't skip special tokens so we can see UNK tokens
        text = tokenizer.decode(token_ids[0], skip_special_tokens=False)
        
        # Clean up the text by removing padding tokens and start/end tokens
        # but keep UNK tokens visible
        text = text.replace("<pad>", "").replace("<s>", "").replace("</s>", "")
        text = text.replace("<unk>", " <unk> ")  # Make UNK tokens more visible
        
        # Clean up extra spaces
        text = " ".join(text.split())
    
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
    Clamp predicted latents to nearest word embeddings using L2 distance (the "clamping trick").
    
    Following the reference Diffusion-LM implementation which uses
    L2 distance rather than cosine similarity for better magnitude sensitivity.
    
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
    vocab_embeddings = diffusion_model.bart_model.get_encoder().embed_tokens.weight  # [vocab_size, embed_dim]
    
    # Reshape for batch computation
    pred_flat = predicted_x0.view(-1, embed_dim)  # [B*L, embed_dim]
    
    distances = compute_l2_distances(pred_flat, vocab_embeddings)
    
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
        token_ids = torch.where(mask, token_ids, torch.tensor(1, device=token_ids.device))  # 1 is <pad>
    
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
        tokenizer: BART tokenizer
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
            'original_text': tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False).replace("<pad>", "").replace("<s>", "").replace("</s>", "").replace("<unk>", " <unk> "),
            'noisy_text': noisy_text,
            'denoised_text': denoised_text,
            'noise_percentage': noise_percentage,
            'cosine_similarity': cosine_sim,
            'timestep': timestep
        }
