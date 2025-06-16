import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import random
from datasets import load_dataset
from train_denoiser import TextUNet1D, SqrtNoiseScheduler, pad_tensor, LatentDataset
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
            scheduler = SqrtNoiseScheduler(num_timesteps=2000, s=1e-4)
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

def main():
    # Example text from Wikipedia
    example_texts = [
        "The quick brown fox jumps over the lazy dog. This is a test of the denoising model.",
        "Machine learning is a field of study in artificial intelligence concerned with the development of algorithms that can learn from and make predictions on data.",
        "The solar system consists of the Sun and everything that orbits around it, including planets, moons, asteroids, comets, and meteoroids.",
    ]
    
    try:
        # Load models
        print("Loading models...")
        bart_model, tokenizer, denoiser, device = load_models()
        
        # Load a small dataset to compute normalization stats
        print("Computing normalization statistics...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        sample_dataset = LatentDataset(bart_model, tokenizer, dataset['train']['text'][:100], max_length=128)
        
        # Test each example
        for i, text in enumerate(example_texts, 1):
            print(f"\nExample {i}:")
            print("Original text:", text)
            
            # Garble the text
            garbled = garble_text(text)
            print("Garbled text:", garbled)
            
            # Denoise the text
            denoised = denoise_text(
                bart_model, 
                tokenizer, 
                denoiser, 
                garbled, 
                device,
                latent_mean=sample_dataset.latent_mean,
                latent_std=sample_dataset.latent_std
            )
            print("Denoised text:", denoised)
            print("-" * 80)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 