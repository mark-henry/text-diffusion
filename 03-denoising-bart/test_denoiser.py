import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import random
from datasets import load_dataset
from train_denoiser import TextUNet1D, CosineNoiseScheduler, pad_tensor, LatentDataset
from transformers.modeling_outputs import BaseModelOutput

def load_models():
    """Load the trained denoiser model and BART model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load BART
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Load denoiser
    denoiser = TextUNet1D(in_channels=768, out_channels=768, time_embed_dim=256)
    denoiser.load_state_dict(torch.load("best_simple_denoiser.pt", map_location=device))
    denoiser.eval()
    
    return bart_model, tokenizer, denoiser, device

def garble_text(text: str, swap_prob: float = 0.3) -> str:
    """Randomly swap words in the text to create garbled input."""
    words = text.split()
    n_words = len(words)
    
    # Create a copy of words to modify
    garbled_words = words.copy()
    
    # Randomly swap words
    for i in range(n_words):
        if random.random() < swap_prob:
            # Find another random position to swap with
            j = random.randint(0, n_words - 1)
            garbled_words[i], garbled_words[j] = garbled_words[j], garbled_words[i]
    
    return " ".join(garbled_words)

def denoise_text(bart_model, tokenizer, denoiser, text: str, device, latent_mean=None, latent_std=None):
    """Denoise the text using our trained model."""
    try:
        # Move models to device
        bart_model = bart_model.to(device)
        denoiser = denoiser.to(device)
        
        # Get BART embeddings
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get BART encoder outputs
            encoder_outputs = bart_model.get_encoder()(inputs["input_ids"])
            latents = encoder_outputs.last_hidden_state
            latents = latents.squeeze()
            # If still not 2D, try to select the first batch
            if latents.dim() > 2:
                latents = latents[0]
            
            # Debug latent_mean and latent_std shapes
            if latent_mean is not None and latent_std is not None:
                latent_mean = latent_mean.squeeze()
                latent_std = latent_std.squeeze()
                latents = (latents - latent_mean.to(device)) / latent_std.to(device)
            
            # Pad to 128 tokens (match training)
            latents = pad_tensor(latents, 128)
            latents = latents.transpose(0, 1).unsqueeze(0)  # [1, 768, 128]
            
            # Add noise
            scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
            timesteps = torch.tensor([500], device=device)  # Use less noise
            noisy_latents, _ = scheduler.add_noise(latents, timesteps)
            
            # Denoise
            predicted_noise = denoiser(noisy_latents, timesteps)
            
            # Remove noise
            alphas_cumprod = scheduler.alphas_cumprod.to(device)[timesteps]
            alphas_cumprod = alphas_cumprod.view(-1, 1, 1)
            denoised_latents = (noisy_latents - torch.sqrt(1 - alphas_cumprod) * predicted_noise) / torch.sqrt(alphas_cumprod)
            
            # Unnormalize if stats were provided
            if latent_mean is not None and latent_std is not None:
                latent_mean_reshape = latent_mean.view(1, -1, 1)
                latent_std_reshape = latent_std.view(1, -1, 1)
                denoised_latents = denoised_latents * latent_std_reshape.to(device) + latent_mean_reshape.to(device)
            
            # Use BART's generate method with the denoised latents
            encoder_outputs = BaseModelOutput(last_hidden_state=denoised_latents.transpose(1, 2))
            generated_ids = bart_model.generate(
                input_ids=inputs["input_ids"],
                encoder_outputs=encoder_outputs,
                max_length=128,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # Decode the generated tokens
            denoised_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
        return denoised_text
    except Exception as e:
        print(f"Error during denoising: {str(e)}")
        raise

def progressive_denoise(bart_model, tokenizer, denoiser, device, latent_mean=None, latent_std=None, num_steps=50, print_every=5):
    """Test progressive denoising starting from random noise."""
    try:
        # Move models to device
        bart_model = bart_model.to(device)
        denoiser = denoiser.to(device)
        
        # Initialize scheduler - MUST match training parameters!
        scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
        
        # Start with random noise but in the normalized latent space
        # Scale the noise to match the expected magnitude of normalized BART latents
        latents = torch.randn(1, 768, 128, device=device) * 0.5  # Reduce initial noise magnitude
        
        print("Starting progressive denoising...")
        
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
                # Predict noise
                predicted_noise = denoiser(latents, timestep)
                
                # Scale prediction to compensate for under-prediction during training
                # Based on debug analysis: model predicts ~85 when actual is ~313
                predicted_noise = predicted_noise * 3.5
                
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
                        max_length=30,
                        num_beams=1,
                        do_sample=False,  # Greedy decoding for consistency
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode and print
                    current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    progress = (len(timesteps) - i - 1) / len(timesteps) * 100
                    alpha_val = alpha_t.item() if hasattr(alpha_t, 'item') else alpha_t
                    latent_magnitude = latents.norm().item()
                    print(f"Step {i:2d} ({progress:5.1f}% noise, α={alpha_val:.4f}, |x|={latent_magnitude:.2f}): {current_text}")
                    
                except Exception as gen_error:
                    print(f"Step {i:2d}: Error generating text - {gen_error}")
        
        return current_text
    except Exception as e:
        print(f"Error during progressive denoising: {str(e)}")
        raise

def main():
    try:
        # Load models
        print("Loading models...")
        bart_model, tokenizer, denoiser, device = load_models()
        
        # Load a small dataset to compute normalization stats
        print("Computing normalization statistics...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        sample_dataset = LatentDataset(bart_model, tokenizer, dataset['train']['text'][:100], max_length=128)
        
        print("\nTesting progressive denoising from random noise:\n")
        final_text = progressive_denoise(
            bart_model,
            tokenizer,
            denoiser,
            device,
            latent_mean=sample_dataset.latent_mean,
            latent_std=sample_dataset.latent_std,
        )
        print("\nFinal result:", final_text)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 