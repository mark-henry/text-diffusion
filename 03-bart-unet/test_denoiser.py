import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import random
from datasets import load_dataset
from train_denoiser import TextUNet1D, CosineNoiseScheduler, pad_tensor, LatentDataset
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F

def load_models():
    """Load the trained denoiser model and BART model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load BART
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Load the latest trained model
    denoiser = TextUNet1D(in_channels=768, out_channels=768, time_embed_dim=256, num_timesteps=2000)
    
    try:
        checkpoint = torch.load("best_multi_objective_denoiser.pt", map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            denoiser.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            denoiser.load_state_dict(checkpoint)
            print("✅ Loaded model weights")
    except FileNotFoundError:
        print("❌ No trained model found! Run training first.")
        return None, None, None, None
    
    denoiser = denoiser.to(device)
    denoiser.eval()
    
    return bart_model, tokenizer, denoiser, device

def magnitude_scaled_denoising(denoiser, noisy_latents, timesteps, scheduler, device):
    """Apply our magnitude scaling technique during denoising."""
    with torch.no_grad():
        # Get model prediction
        predicted_noise = denoiser(noisy_latents, timesteps)
        
        # Calculate expected noise magnitude based on the noise schedule
        alphas_cumprod = scheduler.alphas_cumprod.to(device)[timesteps]
        expected_noise_level = torch.sqrt(1 - alphas_cumprod)
        
        # The "expected" magnitude of noise should be proportional to the signal magnitude
        # and the noise level at this timestep
        signal_magnitude = torch.norm(noisy_latents, dim=(1, 2), keepdim=True)
        expected_noise_magnitude = signal_magnitude * expected_noise_level.view(-1, 1, 1)
        
        # Scale predicted noise to match expected magnitude
        predicted_noise_flat = predicted_noise.reshape(predicted_noise.shape[0], -1)
        current_noise_magnitude = torch.norm(predicted_noise_flat, dim=1, keepdim=True)
        
        # Avoid division by zero
        scale_factor = expected_noise_magnitude.reshape(-1, 1) / (current_noise_magnitude + 1e-8)
        
        # Apply scaling
        scaled_predicted_noise = predicted_noise_flat * scale_factor
        scaled_predicted_noise = scaled_predicted_noise.reshape(predicted_noise.shape)
        
        return scaled_predicted_noise

def progressive_denoise(bart_model, tokenizer, denoiser, device, latent_mean=None, latent_std=None, num_steps=50, print_every=5):
    """Test progressive denoising starting from random noise with magnitude scaling."""
    try:
        # Move models to device
        bart_model = bart_model.to(device)
        denoiser = denoiser.to(device)
        
        # Initialize scheduler - MUST match training parameters!
        scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
        
        # Start with random noise but in the normalized latent space
        # Scale the noise to match the expected magnitude of normalized BART latents
        latents = torch.randn(1, 768, 128, device=device) * 0.5  # Reduce initial noise magnitude
        
        print("Starting progressive denoising with magnitude scaling...")
        
        # Progressive denoising - start from high timestep and go down to 0
        # Use more timesteps early where noise is high, fewer when noise is low
        timesteps = torch.logspace(
            start=torch.log10(torch.tensor(1999.0)), 
            end=torch.log10(torch.tensor(1.0)), 
            steps=num_steps
        ).long()
        
        for i, timestep_val in enumerate(timesteps):
            timestep = torch.tensor([timestep_val], device=device)
            
            with torch.no_grad():
                # Use magnitude-scaled denoising
                predicted_noise = magnitude_scaled_denoising(denoiser, latents, timestep, scheduler, device)
                
                # DDIM-style denoising step (more stable than DDPM)
                alpha_t = scheduler.alphas_cumprod[timestep_val].to(device)
                
                # Predict clean signal
                pred_x0 = (latents - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                
                if i < len(timesteps) - 1:  # Not the last step
                    # DDIM update (deterministic, more stable)
                    alpha_prev = scheduler.alphas_cumprod[timesteps[i + 1]].to(device)
                    pred_noise = predicted_noise  # Use predicted noise directly
                    latents = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
                else:
                    # Final step - return denoised signal
                    latents = pred_x0
            
            # Print every few steps
            if i % print_every == 0 or i == len(timesteps) - 1:
                try:
                    # Unnormalize if stats were provided
                    if latent_mean is not None and latent_std is not None:
                        latent_mean_reshape = latent_mean.view(1, -1, 1)
                        latent_std_reshape = latent_std.view(1, -1, 1)
                        current_latents = latents * latent_std_reshape.to(device) + latent_mean_reshape.to(device)
                    else:
                        current_latents = latents
                    
                    # Generate text from current latents
                    encoder_outputs = BaseModelOutput(last_hidden_state=current_latents.transpose(1, 2))
                    
                    generated_ids = bart_model.generate(
                        encoder_outputs=encoder_outputs,
                        max_length=50,
                        num_beams=1,
                        do_sample=False,  # Greedy decoding for consistency
                        early_stopping=False,  # Disable early_stopping when using num_beams=1
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode and print
                    current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    # Fix: Calculate actual noise level instead of misleading "steps remaining"
                    alpha_val = alpha_t.item() if hasattr(alpha_t, 'item') else alpha_t
                    actual_noise_level = torch.sqrt(1 - alpha_t).item() * 100
                    denoising_progress = (1999 - timestep_val.item()) / 1999 * 100
                    latent_magnitude = latents.norm().item()
                    noise_magnitude = predicted_noise.norm().item()
                    print(f"Step {i:2d} ({actual_noise_level:5.1f}% noise, {denoising_progress:5.1f}% progress, α={alpha_val:.4f}, |x|={latent_magnitude:.2f}, |ε|={noise_magnitude:.2f}): {current_text}")
                    
                except Exception as gen_error:
                    print(f"Step {i:2d}: Error generating text - {gen_error}")
        
        return current_text
    except Exception as e:
        print(f"Error during progressive denoising: {str(e)}")
        raise

def test_reconstructions(bart_model, tokenizer, denoiser, device, latent_mean, latent_std):
    """Test reconstructing real text through noise and denoising."""
    print("\n=== Testing Text Reconstruction ===")
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming our world.",
        "Science and technology advance human knowledge.",
        "Natural language processing enables machine understanding."
    ]
    
    scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
    
    for text in test_texts:
        print(f"\n📝 Original: {text}")
        
        try:
            # Encode to latents
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                encoder_outputs = bart_model.get_encoder()(inputs["input_ids"])
                latents = encoder_outputs.last_hidden_state.squeeze(0)
                latents = pad_tensor(latents, 128)
                
                # Normalize
                latents = (latents - latent_mean.squeeze().to(device)) / latent_std.squeeze().to(device)
                latents = latents.transpose(0, 1).unsqueeze(0)  # [1, 768, 128]
                
                # Add noise at different levels
                for noise_level in [100, 500, 1000, 1500]:
                    timesteps = torch.tensor([noise_level], device=device)
                    noisy_latents, actual_noise = scheduler.add_noise(latents, timesteps)
                    
                    # Denoise with magnitude scaling
                    predicted_noise = magnitude_scaled_denoising(denoiser, noisy_latents, timesteps, scheduler, device)
                    
                    # Reconstruct
                    alphas_cumprod = scheduler.alphas_cumprod.to(device)[timesteps]
                    reconstructed = (noisy_latents - torch.sqrt(1 - alphas_cumprod) * predicted_noise) / torch.sqrt(alphas_cumprod)
                    
                    # Unnormalize
                    latent_mean_reshape = latent_mean.view(1, -1, 1)
                    latent_std_reshape = latent_std.view(1, -1, 1)
                    reconstructed = reconstructed * latent_std_reshape.to(device) + latent_mean_reshape.to(device)
                    
                    # Generate text
                    encoder_outputs = BaseModelOutput(last_hidden_state=reconstructed.transpose(1, 2))
                    generated_ids = bart_model.generate(
                        encoder_outputs=encoder_outputs,
                        max_length=50,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    reconstructed_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    
                    # Calculate metrics
                    cosine_sim = F.cosine_similarity(
                        predicted_noise.reshape(1, -1), 
                        actual_noise.reshape(1, -1), 
                        dim=1
                    ).item()
                    
                    pred_magnitude = predicted_noise.norm().item()
                    actual_magnitude = actual_noise.norm().item()
                    magnitude_ratio = pred_magnitude / actual_magnitude
                    
                    print(f"🔊 Noise t={noise_level:4d} (cos={cosine_sim:.3f}, mag={magnitude_ratio:.3f}): {reconstructed_text}")
                    
        except Exception as e:
            print(f"❌ Error reconstructing '{text}': {e}")

def main():
    try:
        # Load models
        print("Loading models...")
        bart_model, tokenizer, denoiser, device = load_models()
        
        if denoiser is None:
            print("Failed to load denoiser model!")
            return
        
        # Load a small dataset to compute normalization stats
        print("Computing normalization statistics...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        sample_dataset = LatentDataset(bart_model, tokenizer, dataset['train']['text'][:100], max_length=128)
        
        # Test reconstruction of real text first
        test_reconstructions(bart_model, tokenizer, denoiser, device, 
                           sample_dataset.latent_mean, sample_dataset.latent_std)
        
        print("\n" + "="*80)
        print("Testing progressive denoising from random noise with magnitude scaling:")
        print("="*80)
        
        # Try with moderate number of denoising steps
        print("\n=== Testing with 100 steps ===")
        final_text = progressive_denoise(
            bart_model,
            tokenizer,
            denoiser,
            device,
            latent_mean=sample_dataset.latent_mean,
            latent_std=sample_dataset.latent_std,
            num_steps=100,
            print_every=10
        )
        print(f"\n🎯 Final result: {final_text}")
        
        # Try with fewer steps for comparison
        print("\n=== Testing with 25 steps ===")
        final_text_fast = progressive_denoise(
            bart_model,
            tokenizer,
            denoiser,
            device,
            latent_mean=sample_dataset.latent_mean,
            latent_std=sample_dataset.latent_std,
            num_steps=25,
            print_every=5
        )
        print(f"\n⚡ Fast result: {final_text_fast}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 