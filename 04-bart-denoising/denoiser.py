import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel, BartConfig
from transformers.modeling_outputs import BaseModelOutput
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
        
    def compute_clean_latents(self, input_ids, attention_mask=None) -> torch.Tensor:
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

    def get_learnable_embeddings(self, input_ids, attention_mask=None) -> torch.Tensor:
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

    def get_vocab_logits(self, hidden_states):
        """
        Convert hidden states to vocabulary logits using weight tying.
        
        This implements weight tying by using the same weight matrix for both
        embedding lookup and output projection. This prevents embedding collapse
        by forcing the model to maintain distinct embeddings for different tokens.
        
        Args:
            hidden_states: [B, L, C] hidden representations
            
        Returns:
            logits: [B, L, vocab_size] vocabulary logits
        """
        # Weight tying: use embedding weights as output projection
        # This is the key mechanism that prevents embedding collapse
        embed_weight = self.embed_tokens.weight
        assert isinstance(embed_weight, torch.Tensor), "Expected embedding weight to be a tensor"
        return torch.nn.functional.linear(hidden_states, embed_weight)

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


def encode_text_to_latents(text, bart_model, tokenizer, device, max_length=64):
    """Encode text to BART latents"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                      truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get encoder outputs (latents)
    with torch.no_grad():
        encoder_outputs = bart_model.get_encoder()(**inputs)
        latents = encoder_outputs.last_hidden_state  # [1, seq_len, hidden_size]
    
    return latents


def decode_latents_to_text(latents, diffusion_model, tokenizer, device, attention_mask=None):
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
        # 1. Clamp latents to nearest word embeddings
        clamped_latents = clamp_to_embeddings(latents, diffusion_model, attention_mask)
        
        # 2. Find which token IDs correspond to the clamped embeddings
        batch_size, seq_len, embed_dim = clamped_latents.shape
        vocab_embeddings = diffusion_model.embed_tokens.weight  # [vocab_size, embed_dim]
        
        # Reshape for computation
        clamped_flat = clamped_latents.view(-1, embed_dim)  # [B*L, embed_dim]
        
        # Find exact matches (since we clamped to these embeddings)
        # Use cosine similarity to find the matching embedding indices
        clamped_normalized = F.normalize(clamped_flat, p=2, dim=1)
        vocab_normalized = F.normalize(vocab_embeddings, p=2, dim=1)
        similarities = torch.mm(clamped_normalized, vocab_normalized.t())
        token_ids_flat = torch.argmax(similarities, dim=1)  # [B*L]
        
        # Reshape back to sequence format
        token_ids = token_ids_flat.view(batch_size, seq_len)  # [B, L]
        
        # 3. Decode token IDs to text
        text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    
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


def clamp_to_embeddings(predicted_x0, diffusion_model, attention_mask=None):
    """
    Clamp predicted latents to nearest word embeddings (the "clamping trick").
    
    Args:
        predicted_x0: [B, L, C] predicted clean latents
        diffusion_model: Model with embedding weights
        attention_mask: [B, L] mask for valid positions
    
    Returns:
        clamped_x0: [B, L, C] latents clamped to nearest embeddings
    """
    batch_size, seq_len, embed_dim = predicted_x0.shape
    
    # Get vocabulary embeddings [vocab_size, embed_dim]
    vocab_embeddings = diffusion_model.embed_tokens.weight  # [vocab_size, 768]
    vocab_size = vocab_embeddings.shape[0]
    
    # Reshape for batch computation
    pred_flat = predicted_x0.view(-1, embed_dim)  # [B*L, embed_dim]
    
    # Compute distances to all vocabulary embeddings using cosine similarity
    # (more stable than L2 distance for high-dimensional spaces)
    pred_normalized = F.normalize(pred_flat, p=2, dim=1)  # [B*L, embed_dim]
    vocab_normalized = F.normalize(vocab_embeddings, p=2, dim=1)  # [vocab_size, embed_dim]
    
    # Cosine similarities: [B*L, vocab_size]
    similarities = torch.mm(pred_normalized, vocab_normalized.t())
    
    # Find nearest embedding for each position
    nearest_indices = torch.argmax(similarities, dim=1)  # [B*L]
    
    # Get the corresponding embeddings
    clamped_flat = vocab_embeddings[nearest_indices]  # [B*L, embed_dim]
    
    # Reshape back to original shape
    clamped_x0 = clamped_flat.view(batch_size, seq_len, embed_dim)
    
    # Apply attention mask if provided (keep original values for padded positions)
    if attention_mask is not None:
        mask = attention_mask.bool().unsqueeze(-1).expand_as(predicted_x0)  # [B, L, C]
        clamped_x0 = torch.where(mask, clamped_x0, predicted_x0)
    
    return clamped_x0


def diffusion_sample_step(xt, t, diffusion_model, scheduler, device, use_clamping=True, attention_mask=None):
    """
    Single sampling step using the clamping trick from Diffusion-LM paper.
    
    Implements: xt-1 = √(α̅_{t-1}) · Clamp(fθ(xt, t)) + √(1 - α̅_{t-1}) · ε
    
    Args:
        xt: [B, L, C] noisy latents at timestep t
        t: timestep (scalar)
        diffusion_model: Trained diffusion model
        scheduler: Noise scheduler
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
            predicted_x0 = clamp_to_embeddings(predicted_x0, diffusion_model, attention_mask)
        
        # 3. Sample xt-1 using the predicted (and possibly clamped) x0
        if t == 0:
            # Already at x0, no more sampling needed
            return predicted_x0, predicted_x0
        
        # Get noise schedule values
        alpha_bar_t_minus_1 = scheduler.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        alpha_bar_t_minus_1 = alpha_bar_t_minus_1.to(device)
        
        # Sample noise
        noise = torch.randn_like(xt)
        
        # Apply formula: xt-1 = √(α̅_{t-1}) · x0_pred + √(1 - α̅_{t-1}) · ε
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t_minus_1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t_minus_1)
        
        xt_minus_1 = sqrt_alpha_bar * predicted_x0 + sqrt_one_minus_alpha_bar * noise
        
        return xt_minus_1, predicted_x0


def demo_denoising_step(text, diffusion_model, bart_model, tokenizer, scheduler, device, timestep=1, max_length=64):
    """
    Demonstrate a single denoising step: text -> add noise -> denoise -> text
    
    Args:
        text: Input text to denoise
        diffusion_model: Trained diffusion model
        bart_model: BART model for encoding/decoding
        tokenizer: BART tokenizer
        scheduler: Noise scheduler
        device: Device to run on
        timestep: Noise level (lower = less noise)
        max_length: Maximum sequence length
    
    Returns:
        Dict with original_text, noisy_text_decoded, denoised_text, noise_percentage
    """
    # Encode text to latents using diffusion model's method
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                      truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Get clean latents
        clean_latents = diffusion_model.compute_clean_latents(
            inputs['input_ids'], inputs['attention_mask']
        )
        
        # Add noise
        timesteps_tensor = torch.tensor([timestep], device=device)
        if timestep == 0:
            noisy_latents = clean_latents
        else:
            noisy_latents, _ = scheduler.add_noise(
                clean_latents.transpose(1, 2), timesteps_tensor
            )
            noisy_latents = noisy_latents.transpose(1, 2)
        
        # Denoise
        predicted_clean_latents = diffusion_model(noisy_latents, timesteps_tensor)
        
        # Decode noisy latents to text (to show what noise looks like)
        noisy_text = decode_latents_to_text(noisy_latents, diffusion_model, tokenizer, device, inputs['attention_mask'])
        
        # Decode denoised latents to text using clamping
        denoised_text = decode_latents_to_text(predicted_clean_latents, diffusion_model, tokenizer, device, inputs['attention_mask'])
        
        # Calculate noise percentage
        noise_percentage = (1 - scheduler.alphas_cumprod[timestep].item()) * 100 if timestep > 0 else 0.0
        
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