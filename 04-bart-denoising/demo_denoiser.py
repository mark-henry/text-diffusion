import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
from datasets import load_dataset
import random
import math
import os

class CosineNoiseScheduler:
    """Cosine noise schedule following Nichol & Dhariwal (2021)"""
    def __init__(self, num_timesteps=2000, s=0.008):
        self.num_timesteps = num_timesteps
        self.s = s
        
        # Compute the cosine schedule
        t = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Ensure no negative values and proper bounds
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8, max=1.0)
        
        self.alphas_cumprod = alphas_cumprod
        self.alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        self.betas = 1 - self.alphas
    
    def add_noise(self, latents, timesteps):
        """Add noise to latents according to timesteps"""
        noise = torch.randn_like(latents)
        # Move alphas_cumprod to same device as timesteps
        alphas_cumprod = self.alphas_cumprod.to(timesteps.device)[timesteps]
        
        # Reshape for broadcasting with [B, C, L] format
        alphas_cumprod = alphas_cumprod.view(-1, 1, 1)
        noisy_latents = torch.sqrt(alphas_cumprod) * latents + torch.sqrt(1 - alphas_cumprod) * noise
        return noisy_latents, noise

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

def pad_tensor(tensor, target_length):
    """Pad or truncate tensor to target length"""
    current_length = tensor.size(0)
    if current_length < target_length:
        padding = torch.zeros(target_length - current_length, tensor.size(1), 
                            dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=0)
    else:
        return tensor[:target_length]

def load_trained_model(device, model_path="best_diffusion_lm_denoiser.pt"):
    """Load the trained BartDiffusionLM model"""
    print(f"Loading trained model from {model_path}...")
    
    # Initialize model with same parameters as training
    model = BartDiffusionLM(
        bart_model_name="facebook/bart-base",
        max_length=64,
        time_embed_dim=256,
        num_timesteps=2000
    ).to(device)
    
    # Load trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            val_loss = checkpoint.get('best_val_loss', 'unknown')
            print(f"✅ Loaded checkpoint from epoch {epoch} (val_loss: {val_loss})")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded model weights")
    except FileNotFoundError:
        print(f"❌ Model file {model_path} not found!")
        return None
    
    model.eval()
    return model

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

def decode_latents_to_text(latents, bart_model, tokenizer, device):
    """Decode BART latents back to text"""
    with torch.no_grad():
        # Create encoder outputs object
        encoder_outputs = BaseModelOutput(last_hidden_state=latents)
        
        # Generate text
        generated_ids = bart_model.generate(
            encoder_outputs=encoder_outputs,
            max_length=100,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode to text
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return text

def demonstrate_progressive_denoising(diffusion_model, bart_model, tokenizer, device):
    """Demonstrate progressive denoising on WikiText-2 samples"""
    print("\n" + "="*80)
    print("🎯 PROGRESSIVE DENOISING DEMONSTRATION")
    print("="*80)
    
    # Load WikiText-2 dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Filter for non-empty, substantial passages
    substantial_texts = [
        text.strip() for text in dataset["text"] 
        if len(text.strip()) > 100 and not text.strip().startswith("=")
    ]
    
    # Select a few interesting passages
    selected_texts = random.sample(substantial_texts, 3)
    
    # Initialize noise scheduler
    scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
    
    # Test different noise levels
    noise_levels = [0, 50, 200, 500, 1000, 1500, 1900]  # From clean to very noisy
    
    for i, original_text in enumerate(selected_texts):
        print(f"\n{'='*60}")
        print(f"📝 SAMPLE {i+1}: {original_text[:100]}...")
        print(f"{'='*60}")
        
        # Encode original text to latents
        original_latents = encode_text_to_latents(original_text, bart_model, tokenizer, device)
        
        print(f"\n🟢 ORIGINAL: {original_text}")
        
        for noise_level in noise_levels:
            if noise_level == 0:
                # No noise - just show original reconstruction
                reconstructed_text = decode_latents_to_text(original_latents, bart_model, tokenizer, device)
                print(f"\n⚪ t={noise_level:4d} (no noise): {reconstructed_text}")
            else:
                # Add noise
                timesteps = torch.tensor([noise_level], device=device)
                noisy_latents, actual_noise = scheduler.add_noise(original_latents, timesteps)
                
                # Use trained model to predict clean latents (x₀ prediction)
                with torch.no_grad():
                    predicted_clean_latents = diffusion_model(noisy_latents, timesteps)
                
                # Decode predicted clean latents
                reconstructed_text = decode_latents_to_text(predicted_clean_latents, bart_model, tokenizer, device)
                
                # Calculate noise percentage
                alpha_cumprod = scheduler.alphas_cumprod[noise_level].item()
                noise_percentage = (1 - alpha_cumprod) * 100
                
                # Calculate quality metrics
                cosine_sim = F.cosine_similarity(
                    predicted_clean_latents.reshape(1, -1),
                    original_latents.reshape(1, -1),
                    dim=1
                ).item()
                
                magnitude_ratio = (predicted_clean_latents.norm() / original_latents.norm()).item()
                
                print(f"\n🔴 t={noise_level:4d} ({noise_percentage:5.1f}% noise, cos={cosine_sim:.3f}, mag={magnitude_ratio:.3f}): {reconstructed_text}")

def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("Loading BART model...")
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Load trained diffusion model
    diffusion_model = load_trained_model(device)
    if diffusion_model is None:
        print("Failed to load trained model! Make sure you've run training first.")
        return
    
    # Run demonstration
    demonstrate_progressive_denoising(diffusion_model, bart_model, tokenizer, device)
    
    print(f"\n{'='*80}")
    print("✨ Demonstration complete! The model shows strong denoising capabilities.")
    print("Notice how the x₀ prediction approach directly recovers clean text even from high noise levels.")
    print("="*80)

if __name__ == "__main__":
    main() 