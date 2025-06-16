import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from train_denoiser import TextUNet1D, CosineNoiseScheduler, pad_tensor, LatentDataset
from datasets import load_dataset

def test_model_learning():
    """Test if the model actually learned to denoise by comparing predictions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Load denoiser
    denoiser = TextUNet1D(in_channels=768, out_channels=768, time_embed_dim=256)
    denoiser.load_state_dict(torch.load("best_simple_denoiser.pt", map_location=device))
    denoiser.eval()
    denoiser = denoiser.to(device)
    
    # Create scheduler
    scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
    
    # Load dataset for normalization
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    sample_dataset = LatentDataset(bart_model, tokenizer, dataset['train']['text'][:100], max_length=128)
    
    # Test 1: Get a real text latent
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        encoder_outputs = bart_model.get_encoder()(inputs["input_ids"])
        real_latents = encoder_outputs.last_hidden_state.squeeze(0)
        real_latents = pad_tensor(real_latents, 128)
        # Normalize
        real_latents = (real_latents - sample_dataset.latent_mean.squeeze()) / sample_dataset.latent_std.squeeze()
        real_latents = real_latents.transpose(0, 1).unsqueeze(0).to(device)  # [1, 768, 128]
    
    # Test 2: Create random latents
    random_latents = torch.randn_like(real_latents).to(device)
    
    # Test at different noise levels
    timesteps_to_test = [100, 500, 1000, 1500, 1900]
    
    print("\nTesting model predictions at different noise levels:")
    print("=" * 80)
    
    for t in timesteps_to_test:
        timestep = torch.tensor([t], device=device)
        
        # Add noise to real latents
        noisy_real, actual_noise = scheduler.add_noise(real_latents, timestep)
        
        # Predict noise for real vs random latents
        with torch.no_grad():
            pred_noise_real = denoiser(noisy_real, timestep)
            pred_noise_random = denoiser(random_latents, timestep)
        
        # Calculate prediction quality
        real_mse = torch.nn.functional.mse_loss(pred_noise_real, actual_noise).item()
        random_norm = pred_noise_random.norm().item()
        real_pred_norm = pred_noise_real.norm().item()
        actual_noise_norm = actual_noise.norm().item()
        
        # Calculate cosine similarity
        real_cosine = torch.cosine_similarity(
            pred_noise_real.reshape(1, -1), 
            actual_noise.reshape(1, -1)
        ).item()
        
        print(f"Timestep {t:4d}:")
        print(f"  Real latent MSE: {real_mse:.6f}")
        print(f"  Real cosine sim:  {real_cosine:.4f}")
        print(f"  Actual noise norm: {actual_noise_norm:.2f}")
        print(f"  Pred noise norm:   {real_pred_norm:.2f}")
        print(f"  Random pred norm:  {random_norm:.2f}")
        print(f"  α_cumulative:      {scheduler.alphas_cumprod[t].item():.6f}")
        print()
    
    # Test 3: Check if model outputs vary with timestep
    print("Testing timestep sensitivity:")
    print("-" * 40)
    test_latent = real_latents
    timesteps = torch.tensor([0, 500, 1000, 1500, 1999], device=device)
    
    with torch.no_grad():
        for t in timesteps:
            pred = denoiser(test_latent, t.unsqueeze(0))
            print(f"Timestep {t.item():4d}: prediction norm = {pred.norm().item():.2f}")

if __name__ == "__main__":
    test_model_learning() 