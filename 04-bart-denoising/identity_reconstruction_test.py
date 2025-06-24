import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
from datasets import load_dataset
import numpy as np
import random

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
        
        # FROZEN: BART's embedding layers
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
        
        # TRAINABLE: BART's transformer layers
        self.transformer_layers = encoder.layers
        
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
        """Create sinusoidal timestep embeddings"""
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=-1)
        return emb

    def forward(self, noisy_latents, timesteps):
        """Forward pass to predict clean latents x_0"""
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
        scaled_noisy_latents = noisy_latents * time_scale.unsqueeze(-1)  # Broadcasting
        
        # Project noisy latents and add time information
        projected_latents = self.input_proj(scaled_noisy_latents)
        embeddings = projected_latents + time_proj
        
        # Add BART's positional embeddings (using frozen embed_positions)
        position_ids = torch.arange(seq_len, device=noisy_latents.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.embed_positions(position_ids)
        
        # Combine embeddings
        embeddings = embeddings + position_embeddings
        
        # Apply BART's embedding normalization (frozen)
        embeddings = self.layernorm_embedding(embeddings)
        
        # Create attention mask (all positions are valid)
        attention_mask = torch.ones(batch_size, seq_len, device=noisy_latents.device, dtype=torch.bool)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
        attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
        
        # Pass through BART's transformer layers (trainable)
        hidden_states = embeddings
        for layer in self.transformer_layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
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
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                      truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        encoder_outputs = bart_model.get_encoder()(**inputs)
        latents = encoder_outputs.last_hidden_state  # [1, seq_len, hidden_size]
    
    return latents

def decode_latents_to_text(latents, bart_model, tokenizer, device):
    """Decode BART latents back to text"""
    with torch.no_grad():
        encoder_outputs = BaseModelOutput(last_hidden_state=latents)
        generated_ids = bart_model.generate(
            encoder_outputs=encoder_outputs,
            max_length=100,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text

def test_identity_reconstruction(diffusion_model, bart_model, tokenizer, device, num_samples=10):
    """Test identity reconstruction: f(x_0, t=0) ≈ x_0 and at very low timesteps"""
    print("\n" + "="*80)
    print("🔍 IDENTITY RECONSTRUCTION TEST")
    print("Testing whether f(x_0, t) ≈ x_0 at very low timesteps")
    print("="*80)
    
    # Load WikiText-2 dataset
    print("Loading WikiText-2 test dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Filter for substantial passages
    substantial_texts = [
        text.strip() for text in dataset["text"] 
        if len(text.strip()) > 50 and not text.strip().startswith("=")
    ]
    
    # Select random samples
    selected_texts = random.sample(substantial_texts, min(num_samples, len(substantial_texts)))
    
    # Initialize scheduler
    scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
    
    # Test different low timesteps
    test_timesteps = [0, 1, 2, 5, 10]
    
    # Metrics storage for each timestep
    results = {}
    for t in test_timesteps:
        results[t] = {
            'cosine_similarities': [],
            'magnitude_ratios': [],
            'mse_losses': [],
            'l1_losses': [],
            'noise_levels': []
        }
    
    print(f"\nTesting {len(selected_texts)} samples at timesteps: {test_timesteps}")
    
    for i, original_text in enumerate(selected_texts):
        print(f"\n{'='*60}")
        print(f"📝 SAMPLE {i+1}/{len(selected_texts)}")
        print(f"Original: {original_text[:80]}...")
        print(f"{'='*60}")
        
        try:
            # Encode original text to latents
            original_latents = encode_text_to_latents(original_text, bart_model, tokenizer, device)
            
            # Test each timestep
            for t in test_timesteps:
                timesteps = torch.tensor([t], device=device)
                
                if t == 0:
                    # No noise for t=0
                    noisy_latents = original_latents
                    actual_noise_level = 0.0
                else:
                    # Add minimal noise for t>0
                    noisy_latents, _ = scheduler.add_noise(original_latents.transpose(1, 2), timesteps)
                    noisy_latents = noisy_latents.transpose(1, 2)  # Back to [B, L, C]
                    
                    # Calculate actual noise level
                    alpha_cumprod = scheduler.alphas_cumprod[t].item()
                    actual_noise_level = (1 - alpha_cumprod) * 100
                
                with torch.no_grad():
                    # Pass latents through model
                    reconstructed_latents = diffusion_model(noisy_latents, timesteps)
                
                # Calculate metrics
                original_flat = original_latents.reshape(1, -1)
                reconstructed_flat = reconstructed_latents.reshape(1, -1)
                
                # Cosine similarity
                cosine_sim = F.cosine_similarity(original_flat, reconstructed_flat, dim=1).item()
                results[t]['cosine_similarities'].append(cosine_sim)
                
                # Magnitude ratio
                original_magnitude = original_flat.norm().item()
                reconstructed_magnitude = reconstructed_flat.norm().item()
                magnitude_ratio = reconstructed_magnitude / (original_magnitude + 1e-8)
                results[t]['magnitude_ratios'].append(magnitude_ratio)
                
                # Loss metrics
                mse_loss = F.mse_loss(reconstructed_latents, original_latents).item()
                l1_loss = F.l1_loss(reconstructed_latents, original_latents).item()
                results[t]['mse_losses'].append(mse_loss)
                results[t]['l1_losses'].append(l1_loss)
                results[t]['noise_levels'].append(actual_noise_level)
                
                # Decode for first few samples to see behavior
                if i < 3:  # Only show text for first 3 samples to avoid spam
                    original_decoded = decode_latents_to_text(original_latents, bart_model, tokenizer, device)
                    reconstructed_decoded = decode_latents_to_text(reconstructed_latents, bart_model, tokenizer, device)
                    
                    print(f"\n⏰ t={t:2d} ({actual_noise_level:5.1f}% noise):")
                    print(f"   🟢 Original:      {original_decoded[:80]}...")
                    print(f"   🔄 Reconstructed: {reconstructed_decoded[:80]}...")
                    print(f"   📊 cos={cosine_sim:.3f}, mag={magnitude_ratio:.3f}, mse={mse_loss:.4f}")
                
        except Exception as e:
            print(f"❌ Error processing sample {i+1}: {e}")
    
    # Summary statistics for each timestep
    print(f"\n{'='*80}")
    print("📈 IDENTITY RECONSTRUCTION SUMMARY BY TIMESTEP")
    print("="*80)
    
    for t in test_timesteps:
        data = results[t]
        if data['cosine_similarities']:
            avg_noise = np.mean(data['noise_levels'])
            mean_cosine = np.mean(data['cosine_similarities'])
            std_cosine = np.std(data['cosine_similarities'])
            mean_magnitude = np.mean(data['magnitude_ratios'])
            std_magnitude = np.std(data['magnitude_ratios'])
            mean_mse = np.mean(data['mse_losses'])
            
            print(f"\n⏰ TIMESTEP t={t:2d} (avg {avg_noise:.1f}% noise):")
            print(f"   🎯 Cosine Similarity: {mean_cosine:.4f} ± {std_cosine:.4f}")
            print(f"   📏 Magnitude Ratio:   {mean_magnitude:.4f} ± {std_magnitude:.4f}")
            print(f"   💔 MSE Loss:          {mean_mse:.6f}")
            
            # Quality assessment
            if mean_cosine > 0.99 and 0.95 < mean_magnitude < 1.05:
                quality = "🟢 EXCELLENT"
            elif mean_cosine > 0.95 and 0.90 < mean_magnitude < 1.10:
                quality = "🟡 GOOD"
            elif mean_cosine > 0.90:
                quality = "🟠 FAIR"
            else:
                quality = "🔴 POOR"
            
            print(f"   Quality: {quality}")
    
    # Compare across timesteps
    print(f"\n🔍 COMPARISON ACROSS TIMESTEPS:")
    print("="*50)
    print("Timestep | Noise%  | Cosine Sim | Mag Ratio | Quality")
    print("-"*50)
    
    for t in test_timesteps:
        data = results[t]
        if data['cosine_similarities']:
            avg_noise = np.mean(data['noise_levels'])
            mean_cosine = np.mean(data['cosine_similarities'])
            mean_magnitude = np.mean(data['magnitude_ratios'])
            
            if mean_cosine > 0.95:
                quality_icon = "🟢"
            elif mean_cosine > 0.90:
                quality_icon = "🟡"
            elif mean_cosine > 0.80:
                quality_icon = "🟠"
            else:
                quality_icon = "🔴"
            
            print(f"   t={t:2d}   | {avg_noise:5.1f}% | {mean_cosine:8.4f} | {mean_magnitude:7.4f} | {quality_icon}")
    
    # Analysis
    print(f"\n🧠 ANALYSIS:")
    
    # Check if there's improvement with tiny amounts of noise
    if len(test_timesteps) >= 2:
        cosine_t0 = np.mean(results[0]['cosine_similarities'])
        cosine_t1 = np.mean(results[1]['cosine_similarities']) if 1 in results and results[1]['cosine_similarities'] else None
        cosine_t2 = np.mean(results[2]['cosine_similarities']) if 2 in results and results[2]['cosine_similarities'] else None
        
        print(f"   • At t=0 (no noise): cosine similarity = {cosine_t0:.4f}")
        if cosine_t1:
            print(f"   • At t=1 (tiny noise): cosine similarity = {cosine_t1:.4f}")
            diff_1 = cosine_t1 - cosine_t0
            if diff_1 > 0.01:
                print(f"     ↗️ IMPROVEMENT: +{diff_1:.4f} (model performs BETTER with tiny noise!)")
            elif diff_1 < -0.01:
                print(f"     ↘️ DEGRADATION: {diff_1:.4f} (noise hurts performance)")
            else:
                print(f"     ➡️ STABLE: {diff_1:+.4f} (minimal change)")
        
        if cosine_t2:
            print(f"   • At t=2 (tiny noise): cosine similarity = {cosine_t2:.4f}")
            diff_2 = cosine_t2 - cosine_t0
            if diff_2 > 0.01:
                print(f"     ↗️ IMPROVEMENT: +{diff_2:.4f} (model performs BETTER with tiny noise!)")
            elif diff_2 < -0.01:
                print(f"     ↘️ DEGRADATION: {diff_2:.4f} (noise hurts performance)")
            else:
                print(f"     ➡️ STABLE: {diff_2:+.4f} (minimal change)")
    
    # Check magnitude behavior
    if results[0]['magnitude_ratios']:
        mag_t0 = np.mean(results[0]['magnitude_ratios'])
        print(f"   • Magnitude amplification at t=0: {mag_t0:.2f}x")
        if mag_t0 > 1.5:
            print(f"     🔍 The model consistently amplifies magnitude by ~{mag_t0:.1f}x")
            print(f"     📋 This suggests the projection layers are not learning identity")
    
    print(f"\n💡 INSIGHTS:")
    print(f"   • Perfect identity reconstruction would show cosine_sim ≈ 1.0 and magnitude_ratio ≈ 1.0")
    print(f"   • If performance improves with tiny noise, it suggests t=0 is a special case")
    print(f"   • Consistent magnitude scaling indicates systematic bias in the model")

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
    
    # Run identity reconstruction test
    test_identity_reconstruction(diffusion_model, bart_model, tokenizer, device, num_samples=15)
    
    print(f"\n{'='*80}")
    print("✅ Identity reconstruction test complete!")
    print("This test verifies that the model can reproduce clean latents when given no noise.")
    print("="*80)

if __name__ == "__main__":
    main() 