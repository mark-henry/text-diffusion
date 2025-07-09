from typing import Optional
import torch
import torch.nn.functional as F
from transformers import BartTokenizer, BertTokenizer
from datasets import load_dataset
import random
import numpy as np

from denoiser import (
    DiffusionLM,
    get_substantial_texts_from_dataset,
    demo_denoising_step
)
from train_denoiser import load_checkpoint


def load_model(device, model_path="best_diffusion_lm_denoiser.pt") -> Optional[DiffusionLM]:
    """Load the trained DiffusionLM model with robust configuration detection"""
    print(f"Loading trained model from {model_path}...")
    
    # Use the loading function from train_denoiser
    model, metadata, success = load_checkpoint(model_path, str(device))
    
    if not success or model is None:
        print(f"❌ Model file {model_path} not found or couldn't be loaded!")
        return None
    
    print(f"✅ Successfully loaded model")
    if metadata and metadata.get('epoch', 0) > 0:
        print(f"🎯 Training info: Epoch {metadata['epoch']}, Best Val Loss: {metadata['best_val_loss']:.6f}")
    
    return model

def test_model_performance(diffusion_model, tokenizer, device, num_samples=10):
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
    
    scheduler = diffusion_model.scheduler
    
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
                            truncation=True, padding="max_length", add_special_tokens=False)
            
            # Filter vocabulary to match model's vocab_size (same as in demo_denoising_step)
            input_ids = inputs['input_ids']
            vocab_size = diffusion_model.encoder.get_vocab_size()
            unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 3
            
            # Replace any token ID >= vocab_size with UNK token
            out_of_vocab_mask = input_ids >= vocab_size
            input_ids = torch.where(out_of_vocab_mask, unk_token_id, input_ids)
            
            # Update inputs dict
            inputs['input_ids'] = input_ids
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get ground truth latents exactly like training  
            with torch.no_grad():
                target_latents = diffusion_model.embed_tokens(
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
        
        if cos_sim > 0.85 and 0.9 < mag_ratio < 1.1:
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

def demonstrate_denoising_examples(diffusion_model, tokenizer, device, num_examples=3):
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
    
    # Test different noise levels
    noise_levels = [0, 500, 1500, 1700]  # Clean, low, medium, high noise
    
    for i, original_text in enumerate(selected_texts):
        print(f"\n{'='*60}")
        print(f"📝 EXAMPLE {i+1}: {original_text[:80]}...")
        print(f"{'='*60}")
        
        for i, noise_level in enumerate(noise_levels):
            # Create example dictionary from text (like training pipeline)
            inputs = tokenizer(original_text, return_tensors="pt", max_length=64, 
                              truncation=True, padding="max_length", add_special_tokens=False)
            
            # Filter vocabulary to match model's vocab_size
            input_ids = inputs['input_ids']
            vocab_size = diffusion_model.encoder.get_vocab_size()
            unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 3
            
            # Replace any token ID >= vocab_size with UNK token
            out_of_vocab_mask = input_ids >= vocab_size
            input_ids = torch.where(out_of_vocab_mask, unk_token_id, input_ids)
            
            # Create example dictionary
            example_dict = {
                'input_ids': input_ids[0],  # Remove batch dimension
                'attention_mask': inputs['attention_mask'][0]
            }
            
            demo_result = demo_denoising_step(
                example_dict, diffusion_model, tokenizer, 
                device, timestep=noise_level, max_length=64
            )
            
            print(f"\n🔵 Denoised from t={noise_level:4d} ({demo_result['noise_percentage']:5.1f}% noise):")
            print(demo_result['denoised_text'])
            
            print(f"📊 Cosine Similarity: {demo_result['cosine_similarity']:.4f}")


def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo Diffusion Model")
    parser.add_argument("--checkpoint", type=str, default="best_diffusion_lm_denoiser.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--samples", type=int, default=15,
                       help="Number of samples for performance testing")
    parser.add_argument("--examples", type=int, default=3,
                       help="Number of denoising examples to show")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load trained diffusion model first
    diffusion_model = load_model(device, args.checkpoint)
    if diffusion_model is None:
        print("Failed to load trained model! Make sure you've run training first.")
        return
    
    print(f"🚀 {diffusion_model.encoder_type.upper()} Diffusion Model Demo")
    
    # Load appropriate tokenizer based on model's encoder type
    print("\nLoading tokenizer...")
    if diffusion_model.encoder_type == "bart":
        print("Loading BART tokenizer...")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    elif diffusion_model.encoder_type == "bert":
        print("Loading BERT tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        raise ValueError(f"Unsupported encoder type: {diffusion_model.encoder_type}")
    
    print(f"\n🎯 Model successfully loaded! Ready to demonstrate denoising capabilities.")
    print(f"\n📊 MODEL METADATA")
    print(f"{'='*50}")
    print(f"• Noise Scheduler: {diffusion_model.scheduler.__class__.__name__}")
    print(f"• Model Parameters: {sum(p.numel() for p in diffusion_model.parameters()):,}")
    print(f"• Device: {device}")
    print(f"{'='*50}")
    
    # Test model performance with correct metrics
    performance_results = test_model_performance(diffusion_model, tokenizer, device, num_samples=args.samples)
    
    # Show concrete examples
    demonstrate_denoising_examples(diffusion_model, tokenizer, device, num_examples=args.examples)
    
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