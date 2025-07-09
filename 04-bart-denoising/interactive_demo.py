#!/usr/bin/env python3
"""
Interactive Diffusion Denoising Demo

Usage:
    echo "Your text here" | python interactive_demo.py
    python interactive_demo.py < input.txt
    python interactive_demo.py  # then type text and press Ctrl+D
"""

import sys
import torch
import torch.nn.functional as F
from transformers import BartTokenizer
from typing import Dict, Any

from denoiser import DiffusionLM, decode_latents_to_text
from train_denoiser import load_checkpoint


def load_model(device: str, model_path: str = "best_diffusion_lm_denoiser.pt") -> DiffusionLM:
    """Load the trained DiffusionLM model"""
    print(f"📥 Loading model from {model_path}...", file=sys.stderr)
    
    model, metadata, success = load_checkpoint(model_path, device)
    
    if not success or model is None:
        print(f"❌ Failed to load model from {model_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Model loaded successfully!", file=sys.stderr)
    if metadata and metadata.get('epoch', 0) > 0:
        print(f"🎯 Training: Epoch {metadata['epoch']}, Val Loss: {metadata.get('best_val_loss', 'unknown'):.6f}", file=sys.stderr)
    
    return model


def denoise_at_timestep(
    text: str, 
    model: DiffusionLM, 
    tokenizer, 
    device: str, 
    timestep: int
) -> Dict[str, Any]:
    """Run one denoising pass at a specific timestep"""
    
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=64, 
        truncation=True, 
        padding="max_length"
    )
    
    # Filter vocabulary to match model's vocab_size
    input_ids = inputs['input_ids']
    vocab_size = model.encoder.get_vocab_size()
    unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 3
    
    # Replace any token ID >= vocab_size with UNK token
    out_of_vocab_mask = input_ids >= vocab_size
    input_ids = torch.where(out_of_vocab_mask, unk_token_id, input_ids)
    
    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        # Get clean latents (ground truth)
        clean_latents = model.embed_tokens(input_ids, attention_mask)
        
        # Add noise at the specified timestep
        timesteps_tensor = torch.tensor([timestep], device=device)
        
        if timestep == 0:
            # No noise for t=0
            noisy_latents = clean_latents
            noise_percentage = 0.0
        else:
            # Add noise according to the schedule
            noisy_latents, noise = model.scheduler.add_noise(
                clean_latents.transpose(1, 2), timesteps_tensor
            )
            noisy_latents = noisy_latents.transpose(1, 2)
            
            # Calculate noise percentage
            alpha_cumprod = model.scheduler.alphas_cumprod[timestep]
            noise_percentage = (1 - alpha_cumprod.item()) * 100
        
        # Run one denoising pass
        predicted_x0 = model(noisy_latents, timesteps_tensor)
        
        # Decode the result
        denoised_text = decode_latents_to_text(
            predicted_x0, model, tokenizer, attention_mask
        )
        
        # Compute quality metrics
        batch_size = predicted_x0.shape[0]
        pred_flat = predicted_x0.reshape(batch_size, -1)
        clean_flat = clean_latents.reshape(batch_size, -1)
        
        cosine_sim = F.cosine_similarity(pred_flat, clean_flat, dim=1).item()
        
        pred_magnitude = torch.norm(pred_flat, dim=1).item()
        clean_magnitude = torch.norm(clean_flat, dim=1).item()
        magnitude_ratio = pred_magnitude / (clean_magnitude + 1e-8)
    
    return {
        'timestep': timestep,
        'noise_percentage': noise_percentage,
        'denoised_text': denoised_text,
        'cosine_similarity': cosine_sim,
        'magnitude_ratio': magnitude_ratio,
        'out_of_vocab_tokens': out_of_vocab_mask.sum().item()
    }


def main():
    """Main interactive demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Diffusion Denoising Demo")
    parser.add_argument("--model", type=str, default="best_diffusion_lm_denoiser.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--timesteps", type=int, nargs='+', default=[0, 1, 200, 1000],
                       help="Timesteps to test (default: 0 1 200 1000)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"🚀 BART Diffusion Interactive Demo", file=sys.stderr)
    print(f"💻 Device: {device}", file=sys.stderr)
    print(f"📊 Testing timesteps: {args.timesteps}", file=sys.stderr)
    print(f"📝 Reading from stdin...", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Read input text from stdin
    try:
        input_text = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!", file=sys.stderr)
        sys.exit(0)
    
    if not input_text:
        print("❌ No input text provided!", file=sys.stderr)
        sys.exit(1)
    
    # Load models
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = load_model(device, args.model)
    model.eval()
    
    # Show original text
    print("="*80)
    print("📝 ORIGINAL TEXT:")
    print("="*80)
    print(f'"{input_text}"')
    print()
    
    # Test denoising at different timesteps
    print("="*80)
    print("🔄 DENOISING RESULTS:")
    print("="*80)
    
    results = []
    for timestep in args.timesteps:
        try:
            result = denoise_at_timestep(input_text, model, tokenizer, device, timestep)
            results.append(result)
            
            # Display result
            print(f"\n🕐 Timestep t={timestep:4d} ({result['noise_percentage']:5.1f}% noise)")
            print(f"{'─'*60}")
            print(f'"{result["denoised_text"]}"')
            print(f"📊 Cosine Similarity: {result['cosine_similarity']:.4f}")
            print(f"📏 Magnitude Ratio: {result['magnitude_ratio']:.4f}")
            if result['out_of_vocab_tokens'] > 0:
                print(f"⚠️  Out-of-vocab tokens: {result['out_of_vocab_tokens']}")
            
        except Exception as e:
            print(f"❌ Error at timestep {timestep}: {e}", file=sys.stderr)
            continue
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("📈 SUMMARY:")
        print("="*80)
        
        print("Timestep | Noise%  | Cosine Sim | Mag Ratio | Quality")
        print("-"*55)
        
        for result in results:
            cos_sim = result['cosine_similarity']
            mag_ratio = result['magnitude_ratio']
            
            # Quality assessment
            if cos_sim > 0.9 and 0.9 < mag_ratio < 1.1:
                quality = "🟢 Excellent"
            elif cos_sim > 0.7 and 0.7 < mag_ratio < 1.3:
                quality = "🟡 Good"
            elif cos_sim > 0.5:
                quality = "🟠 Fair"
            else:
                quality = "🔴 Poor"
            
            print(f"   t={result['timestep']:4d} | {result['noise_percentage']:5.1f}% | "
                  f"{cos_sim:7.4f} | {mag_ratio:6.4f} | {quality}")
        
        avg_cosine = sum(r['cosine_similarity'] for r in results) / len(results)
        print(f"\n💡 Average cosine similarity: {avg_cosine:.4f}")
        
if __name__ == "__main__":
    main() 