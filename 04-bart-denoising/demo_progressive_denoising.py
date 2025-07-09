#!/usr/bin/env python3
"""
Demo: Progressive Denoising from Random Latents

This script demonstrates the full diffusion sampling process:
1. Start with pure random noise (latents)
2. Progressively denoise using the trained model
3. Show intermediate steps and final decoded text

The demo uses the trained unified diffusion model to generate text from scratch,
showcasing the reverse diffusion process for both BART and BERT encoders.
"""

import torch
from transformers import BartTokenizer, BertTokenizer

# Import our modules
from denoiser import DiffusionLM, diffusion_sample_step, decode_latents_to_text
from train_denoiser import load_checkpoint


def create_random_latents(batch_size, seq_len, embed_dim, device, seed=None):
    """Create random latents to start the denoising process."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, embed_dim, device=device)


def progressive_denoising_demo(
    model_path="best_diffusion_lm_denoiser.pt",
    num_steps=50,
    seq_len=32,
    batch_size=1,
    show_intermediate_steps=True,
    use_clamping=True,
    device=None,
    seed=None
):
    """
    Run progressive denoising demo starting from random latents.
    
    Args:
        model_path: Path to the trained model checkpoint
        num_steps: Number of denoising steps to perform
        seq_len: Sequence length for generation
        batch_size: Batch size (usually 1 for demo)
        show_intermediate_steps: Whether to show intermediate decoded text
        use_clamping: Whether to use the clamping trick
        device: Device to run on (auto-detect if None)
        seed: Seed for random number generator
    """
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    print(f"💾 Loading model...")
    model, metadata, success = load_checkpoint(model_path, str(device))
    if not success or model is None:
        print("❌ Failed to load model with configuration!")
        return None, "Failed to load model"
    
    print(f"🤖 Loaded model for progressive denoising")
    model.eval()  # Set to evaluation mode
    
    # Load appropriate tokenizer based on encoder type
    print(f"📝 Loading {model.encoder_type.upper()} tokenizer...")
    if model.encoder_type == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    elif model.encoder_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        print(f"❌ Unsupported encoder type: {model.encoder_type}")
        return None, "Unsupported encoder type"
    
    # Get model dimensions - use embedding dimension for progressive denoising
    embed_dim = model.encoder.get_embedding_dim()  # Work in embedding space, not transformer hidden space
    print(f"📏 Model dimensions: seq_len={seq_len}, embed_dim={embed_dim} (embedding space)")
    print(f"   Transformer hidden dim: {model.encoder.get_latent_dim()}")
    
    # Create random starting latents (pure noise)
    print("🎲 Creating random starting latents...")
    xt = create_random_latents(batch_size, seq_len, embed_dim, device, seed)
    
    # Create attention mask (all positions are valid for generation)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    
    total_timesteps = model.scheduler.num_timesteps
    if num_steps > total_timesteps:
        num_steps = total_timesteps
        print(f"⚠️ Reducing num_steps to {total_timesteps} (model's max timesteps)")
    
    # Create timestep schedule (from high to low)
    timesteps = torch.linspace(total_timesteps-2, 0, num_steps, dtype=torch.long)
    
    # Show five intermediate steps
    show_at_steps = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1] if show_intermediate_steps else [num_steps-1]
    
    print(f"\n🌟 Starting progressive denoising over {num_steps} steps...")
    print(f"🕐 Timestep range: {timesteps[0].item()} → {timesteps[-1].item()}")
    if use_clamping:
        clamp_start_step = int(0.9 * num_steps) + 1
        print(f"🔍 Clamping: Disabled for first 90% of steps, enabled from step {clamp_start_step}")
    else:
        print(f"🔍 Clamping: Disabled throughout")
    print("=" * 80)
    
    # Progressive denoising loop
    for step_idx, t in enumerate(timesteps):
        t_scalar = t.item()
        
        # Enable clamping only when we're 90% through the schedule
        step_progress = step_idx / (num_steps - 1)  # 0.0 to 1.0
        current_use_clamping = use_clamping and (step_progress >= 0.90)
        
        # Perform one denoising step
        with torch.no_grad():
            xt, predicted_x0 = diffusion_sample_step(
                xt, t_scalar, model, device, 
                use_clamping=current_use_clamping, attention_mask=attention_mask
            )
        
        # Show intermediate results
        if step_idx in show_at_steps or step_idx > 0.80*num_steps:
            # Calculate noise level
            if t_scalar > 0:
                noise_level = (1 - model.scheduler.alphas_cumprod[int(t_scalar)].item()) * 100
            else:
                noise_level = 0.0
            
            clamp_status = "🔒 ON" if current_use_clamping else "🔓 OFF"
            print(f"\n📍 Step {step_idx+1}/{num_steps} (t={t_scalar}, noise={noise_level:.1f}%, clamp={clamp_status})")
            
            # Decode current latents to text
            try:
                current_text = decode_latents_to_text(xt, model, tokenizer, attention_mask)
                print(f"🔤 Current text: '{current_text}'")
                predicted_text = decode_latents_to_text(predicted_x0, model, tokenizer, attention_mask)
                print(f"🔤 Predicted final text: '{predicted_text}'")
            except Exception as e:
                print(f"⚠️ Decoding failed: {e}")
            
            # Show some latent statistics
            latent_mean = xt.mean().item()
            latent_std = xt.std().item()
            print(f"📊 Latent stats: mean={latent_mean:.3f}, std={latent_std:.3f}")
    
    print("\n" + "=" * 80)
    print("🎉 Progressive denoising complete!")
    
    # Final decoding
    try:
        final_text = decode_latents_to_text(xt, model, tokenizer, attention_mask)
        print(f"🎯 Final generated text: '{final_text}'")
    except Exception as e:
        print(f"❌ Final decoding failed: {e}")
    
    # Additional analysis
    print(f"\n📈 Generation summary:")
    print(f"   • Model: {model_path}")
    print(f"   • Sequence length: {seq_len}")
    print(f"   • Denoising steps: {num_steps}")
    print(f"   • Clamping: {'Enabled after 90% of steps' if use_clamping else 'Disabled'}")
    print(f"   • Final latent norm: {torch.norm(xt).item():.3f}")
    
    return xt, final_text

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Progressive Denoising Demo")

    parser.add_argument("--model", type=str, default="best_diffusion_lm_denoiser.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--steps", type=int, default=200,
                       help="Number of denoising steps")
    parser.add_argument("--length", type=int, default=64,
                       help="Sequence length")
    parser.add_argument("--no-clamping", action="store_true",
                       help="Disable clamping trick")
    parser.add_argument("--seed", type=int, default=None,
                       help="Seed for random number generator")
    
    args = parser.parse_args()
    
    progressive_denoising_demo(
        model_path=args.model,
        num_steps=args.steps,
        seq_len=args.length,
        use_clamping=not args.no_clamping,
        show_intermediate_steps=True,
        seed=args.seed
    ) 