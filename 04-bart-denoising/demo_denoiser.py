import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import random
import numpy as np

from denoiser import (
    BartDiffusionLM, 
    CosineNoiseScheduler, 
    pad_tensor, 
    decode_latents_to_text,
    load_checkpoint,
    get_substantial_texts_from_dataset,
    demo_denoising_step
)


def load_trained_model(device, model_path="best_diffusion_lm_denoiser.pt"):
    """Load the trained BartDiffusionLM model"""
    print(f"Loading trained model from {model_path}...")
    
    model = BartDiffusionLM().to(device)
    
    # Load trained weights using the centralized function
    success, epoch, val_loss = load_checkpoint(model, model_path, device)
    if not success:
        print(f"❌ Model file {model_path} not found!")
        return None
    
    model.eval()
    return model

def test_model_performance(diffusion_model, bart_model, tokenizer, device, num_samples=10):
    """Test the model performance using training-style validation"""
    print("\n" + "="*80)
    print("🔬 MODEL PERFORMANCE EVALUATION")
    print("="*80)
    
    # Load WikiText-2 samples
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Filter for substantial passages  
    substantial_texts = get_substantial_texts_from_dataset(dataset, min_length=100)
    
    # Test with random samples
    test_texts = random.sample(substantial_texts, num_samples)
    
    scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
    
    # Test key timesteps that represent different noise levels
    test_timesteps = [0, 1, 5, 10, 50, 100, 500, 1000, 1500, 1900]
    
    all_results = {}
    
    print(f"Testing {num_samples} samples across {len(test_timesteps)} timesteps...")
    
    for timestep in test_timesteps:
        cosine_sims = []
        magnitude_ratios = []
        
        for text in test_texts:
            # Encode text exactly like training
            inputs = tokenizer(text, return_tensors="pt", max_length=64, 
                            truncation=True, padding="max_length")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get ground truth latents exactly like training  
            with torch.no_grad():
                target_latents = diffusion_model.compute_clean_latents(
                    inputs['input_ids'], inputs['attention_mask']
                )
            
            # Add noise exactly like training
            timesteps_tensor = torch.tensor([timestep], device=device)
            if timestep == 0:
                noisy_latents = target_latents
            else:
                noisy_latents, _ = scheduler.add_noise(
                    target_latents.transpose(1, 2), timesteps_tensor
                )
                noisy_latents = noisy_latents.transpose(1, 2)
            
            # Forward pass exactly like training
            with torch.no_grad():
                predicted_x0 = diffusion_model(noisy_latents, timesteps_tensor)
            
            # Compute metrics exactly like training
            batch_size = predicted_x0.shape[0]
            pred_flat = predicted_x0.reshape(batch_size, -1)
            target_flat = target_latents.reshape(batch_size, -1)
            
            cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).item()
            pred_magnitude = torch.norm(pred_flat, dim=1).item()
            target_magnitude = torch.norm(target_flat, dim=1).item()
            magnitude_ratio = pred_magnitude / (target_magnitude + 1e-8)
            
            cosine_sims.append(cosine_sim)
            magnitude_ratios.append(magnitude_ratio)
        
        # Store results
        all_results[timestep] = {
            'cosine_sim': np.mean(cosine_sims),
            'cosine_std': np.std(cosine_sims),
            'magnitude_ratio': np.mean(magnitude_ratios),
            'magnitude_std': np.std(magnitude_ratios),
            'noise_percentage': (1 - scheduler.alphas_cumprod[timestep].item()) * 100 if timestep > 0 else 0.0
        }
    
    # Display results  
    print("\n📊 PERFORMANCE RESULTS:")
    print("Timestep | Noise%  | Cosine Sim ± Std  | Mag Ratio ± Std | Quality")
    print("-"*75)
    
    for timestep in test_timesteps:
        r = all_results[timestep]
        
        # Quality assessment
        cos_sim = r['cosine_sim']
        mag_ratio = r['magnitude_ratio']
        
        if cos_sim > 0.7 and 0.8 < mag_ratio < 1.2:
            quality = "🟢 Excellent"
        elif cos_sim > 0.6 and 0.6 < mag_ratio < 1.4:
            quality = "🟡 Good"
        elif cos_sim > 0.5:
            quality = "🟠 Fair"
        else:
            quality = "🔴 Poor"
        
        print(f"   t={timestep:4d} | {r['noise_percentage']:5.1f}% | "
              f"{cos_sim:.4f} ± {r['cosine_std']:.4f} | "
              f"{mag_ratio:.4f} ± {r['magnitude_std']:.4f} | {quality}")
    
    # Analysis
    print(f"\n🧠 ANALYSIS:")
    cos_mean = np.mean([r['cosine_sim'] for r in all_results.values()])
    mag_mean = np.mean([r['magnitude_ratio'] for r in all_results.values()])
    
    print(f"   • Average cosine similarity: {cos_mean:.4f}")
    print(f"   • Average magnitude ratio: {mag_mean:.4f}")
    print(f"   • Model maintains ~{cos_mean*100:.1f}% semantic direction preservation")
    print(f"   • Magnitude scaling factor: ~{mag_mean:.2f}x")
    
    # Performance across noise levels
    low_noise_cos = np.mean([all_results[t]['cosine_sim'] for t in [0, 1, 5, 10] if t in all_results])
    high_noise_cos = np.mean([all_results[t]['cosine_sim'] for t in [1000, 1500, 1900] if t in all_results])
    
    print(f"   • Low noise performance (t=0-10): {low_noise_cos:.4f}")
    print(f"   • High noise performance (t=1000+): {high_noise_cos:.4f}")
    
    if abs(low_noise_cos - high_noise_cos) < 0.05:
        print(f"   • ✅ Stable performance across noise levels (difference: {abs(low_noise_cos - high_noise_cos):.3f})")
    else:
        print(f"   • ⚠️ Performance varies with noise (difference: {abs(low_noise_cos - high_noise_cos):.3f})")
    
    return all_results

def demonstrate_denoising_examples(diffusion_model, bart_model, tokenizer, device, num_examples=3):
    """Show concrete examples of the denoising process"""
    print("\n" + "="*80)
    print("🎯 DENOISING EXAMPLES")
    print("="*80)
    
    # Load WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Filter for substantial passages
    substantial_texts = get_substantial_texts_from_dataset(dataset, min_length=100)
    
    # Select a few passages
    selected_texts = random.sample(substantial_texts, num_examples)
    
    # Initialize noise scheduler
    scheduler = CosineNoiseScheduler(num_timesteps=2000, s=0.008)
    
    # Test different noise levels
    noise_levels = [0, 500, 1500, 1700]  # Clean, low, medium, high noise
    
    for i, original_text in enumerate(selected_texts):
        print(f"\n{'='*60}")
        print(f"📝 EXAMPLE {i+1}: {original_text[:80]}...")
        print(f"{'='*60}")
        
        for i, noise_level in enumerate(noise_levels):
            # Use the centralized demo function
            demo_result = demo_denoising_step(
                original_text, diffusion_model, bart_model, tokenizer, 
                scheduler, device, timestep=noise_level, max_length=64
            )
            
            print(f"\n🔵 Denoised from t={noise_level:4d} ({demo_result['noise_percentage']:5.1f}% noise):")
            print(demo_result['denoised_text'])
            
            # Show noisy text for non-zero timesteps to see what noise looks like
            if noise_level > 0:
                print(f"📊 Cosine Similarity: {demo_result['cosine_similarity']:.4f}")


def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 BART Diffusion Model Demo")
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading BART model...")
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    bart_model = bart_model.to(device)  # type: ignore
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Load trained diffusion model
    diffusion_model = load_trained_model(device)
    if diffusion_model is None:
        print("Failed to load trained model! Make sure you've run training first.")
        return
    
    print(f"\n🎯 Model successfully loaded! Ready to demonstrate denoising capabilities.")
    
    # Test model performance with correct metrics
    performance_results = test_model_performance(diffusion_model, bart_model, tokenizer, device, num_samples=15)
    
    # Show concrete examples
    demonstrate_denoising_examples(diffusion_model, bart_model, tokenizer, device, num_examples=3)
    
    print(f"\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    
    avg_cosine = np.mean([r['cosine_sim'] for r in performance_results.values()])
    avg_magnitude = np.mean([r['magnitude_ratio'] for r in performance_results.values()])
    
    print(f"✅ Model Performance:")
    print(f"   • Semantic preservation: ~{avg_cosine*100:.1f}% (cosine similarity: {avg_cosine:.3f})")
    print(f"   • Magnitude scaling: ~{avg_magnitude:.2f}x")

if __name__ == "__main__":
    main() 