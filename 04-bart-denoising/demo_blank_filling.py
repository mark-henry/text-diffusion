#!/usr/bin/env python3
"""
Demo: Diffusion Model Blank Filling

This script demonstrates using the trained diffusion model for:
1. Filling blanks in sentences (targeted generation)
2. Open-ended text completion from prompts

The demo uses iterative denoising to generate coherent text completions.
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import random
import time
import re

from denoiser import (
    DiffusionLM,
    diffusion_sample_step,
    decode_latents_to_text
)
from train_denoiser import load_checkpoint


def find_blank_token_id(tokenizer):
    """Find the best token ID to represent blanks in the text."""
    # Try mask token first
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    
    # Fall back to UNK token
    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
        return tokenizer.unk_token_id
    
    # Default to token ID 100 (BERT's UNK)
    return 100


def filter_vocab_for_model(input_ids, vocab_size=15000, unk_token_id=100):
    """Filter token IDs to stay within model's vocabulary."""
    out_of_vocab_mask = input_ids >= vocab_size
    return torch.where(out_of_vocab_mask, unk_token_id, input_ids)


def diffusion_sample(
    diffusion_model,
    tokenizer,
    device,
    prompt_text="",
    total_length=64,
    num_steps=100,
    use_clamping=True,
    verbose=False,
    seed=None
):
    """
    Generate text using diffusion sampling.
    
    Args:
        diffusion_model: Trained DiffusionLM model
        tokenizer: BERT tokenizer
        device: Device to run on
        prompt_text: Starting prompt (can be empty for unconditional generation)
        total_length: Total sequence length to generate
        num_steps: Number of denoising steps
        use_clamping: Whether to use clamping trick in final steps
        verbose: Whether to show intermediate steps
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with generation results
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create random starting latents
    embed_dim = diffusion_model.encoder.get_latent_dim()
    xt = torch.randn(1, total_length, embed_dim, device=device)
    
    # Create attention mask
    attention_mask = torch.ones(1, total_length, device=device, dtype=torch.bool)
    
    # If we have a prompt, encode it and use it to constrain the generation
    prompt_latents = None
    prompt_length = 0
    if prompt_text.strip():
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt", max_length=total_length//2, 
                                 truncation=True, add_special_tokens=False)
        prompt_ids = prompt_tokens.input_ids.to(device)
        
        # Filter vocabulary for model
        model_vocab_size = diffusion_model.encoder.get_vocab_size()
        unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 100
        prompt_ids = filter_vocab_for_model(prompt_ids, vocab_size=model_vocab_size, unk_token_id=unk_token_id)
        
        prompt_length = prompt_ids.shape[1]
        if prompt_length > 0:
            prompt_latents = diffusion_model.embed_tokens(prompt_ids, prompt_tokens.attention_mask.to(device))
            # Copy prompt latents to the beginning of our generation
            xt[:, :prompt_length] = prompt_latents
    
    # Create timestep schedule
    total_timesteps = diffusion_model.scheduler.num_timesteps
    num_steps = min(num_steps, total_timesteps)
    timesteps = torch.linspace(total_timesteps-2, 0, num_steps, dtype=torch.long)
    
    intermediate_texts = []
    timestep_list = []
    
    if verbose:
        print(f"🚀 Starting generation: {num_steps} steps, length={total_length}")
        if prompt_text:
            print(f"📝 Prompt: '{prompt_text}' ({prompt_length} tokens)")
    
    # Progressive denoising
    for step_idx, t in enumerate(timesteps):
        t_scalar = t.item()
        
        # Enable clamping in final 50% of steps
        step_progress = step_idx / (num_steps - 1)
        current_use_clamping = use_clamping and (step_progress >= 0.5)
        
        # Perform denoising step
        with torch.no_grad():
            xt, predicted_x0 = diffusion_sample_step(
                xt, t_scalar, diffusion_model, device,
                use_clamping=current_use_clamping, attention_mask=attention_mask
            )
        
        # Keep prompt fixed if we have one
        if prompt_latents is not None and prompt_length > 0:
            xt[:, :prompt_length] = prompt_latents
        
        # Record intermediate results
        if step_idx % (num_steps // 10) == 0 or step_idx >= num_steps - 3:
            try:
                current_text = decode_latents_to_text(xt, diffusion_model, tokenizer, attention_mask)
                intermediate_texts.append(current_text)
                timestep_list.append(t_scalar)
                
                if verbose:
                    clamp_status = "🔒" if current_use_clamping else "🔓"
                    print(f"Step {step_idx+1:3d}/{num_steps} (t={t_scalar:4d}) {clamp_status}: '{current_text}'")
            except Exception as e:
                if verbose:
                    print(f"Step {step_idx+1:3d}/{num_steps}: Decoding failed ({e})")
    
    # Final decoding
    try:
        final_text = decode_latents_to_text(xt, diffusion_model, tokenizer, attention_mask)
    except Exception as e:
        final_text = f"[Decoding failed: {e}]"
    
    return {
        'final_text': final_text,
        'intermediate_texts': intermediate_texts,
        'timesteps': timestep_list,
        'final_latents': xt
    }


def diffusion_fill_blank(
    text_with_blank,
    diffusion_model,
    tokenizer,
    device,
    blank_marker="____",
    num_steps=200,
    use_clamping=True,
    verbose=False
):
    """
    Fill blanks in text using diffusion sampling.
    
    Args:
        text_with_blank: Text containing blank_marker to fill
        diffusion_model: Trained DiffusionLM model  
        tokenizer: BERT tokenizer
        device: Device to run on
        blank_marker: String marker for blanks (e.g., "____")
        num_steps: Number of denoising steps
        use_clamping: Whether to use clamping trick
        verbose: Whether to show intermediate steps
        
    Returns:
        Dictionary with results
    """
    # Find blank position
    if blank_marker not in text_with_blank:
        raise ValueError(f"No blank marker '{blank_marker}' found in text")
    
    # Split text around blank to get the prefix as prompt
    parts = text_with_blank.split(blank_marker, 1)  # Split only on first occurrence
    prefix = parts[0].strip()
    
    if verbose:
        print(f"🎯 Filling blank with prefix: '{prefix}'")
    
    # Generate text using the prefix as a prompt
    result = diffusion_sample(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        device=device,
        prompt_text=prefix,
        total_length=64,
        num_steps=num_steps,
        use_clamping=use_clamping,
        verbose=verbose
    )
    
    # Just return the denoised text directly
    return {
        'final_text': result['final_text'],
        'intermediate_texts': result['intermediate_texts'],
        'timesteps': result['timesteps']
    }


def load_trained_model(device, model_path="best_diffusion_lm_denoiser.pt"):
    """Load the trained DiffusionLM model."""
    print(f"📥 Loading trained model from {model_path}...")
    
    # Load model using the unified loading function
    model, metadata, success = load_checkpoint(model_path, str(device))
    if not success or model is None:
        print(f"❌ Model file {model_path} not found!")
        return None
    
    # Check if metadata is available
    if metadata is not None:
        epoch = metadata.get('epoch', 'Unknown')
        val_loss = metadata.get('best_val_loss', 'Unknown')
        print(f"✅ Model loaded successfully from epoch {epoch} (val_loss: {val_loss:.6f})" if isinstance(val_loss, (int, float)) else f"✅ Model loaded successfully from epoch {epoch} (val_loss: {val_loss})")
    else:
        print(f"✅ Model loaded successfully (no metadata available)")
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.eval()
    return model


def demonstrate_blank_filling(diffusion_model, tokenizer, device):
    """Demonstrate filling blanks in various text examples."""
    print("\n" + "="*80)
    print("🎯 DIFFUSION BLANK FILLING DEMONSTRATION")
    print("="*80)
    
    # Test examples with blanks to fill
    test_examples = [
        "World War II started in ____",
        "The capital of France is ____",  
        "Albert Einstein developed the theory of ____",
        "The largest planet in our solar system is ____",
        "Python is a programming ____",
        "The process of plants making food is called ____"
    ]
    
    print(f"🔍 Testing {len(test_examples)} blank-filling scenarios...")
    print(f"📋 Blank marker: '____'")
    print(f"⚙️  Using 200 denoising steps with clamping enabled for final 50%")
    
    for i, example in enumerate(test_examples):
        print(f"\n{'─'*60}")
        print(f"📝 EXAMPLE {i+1}: {example}")
        print(f"{'─'*60}")
        
        start_time = time.time()
        
        try:
            result = diffusion_fill_blank(
                text_with_blank=example,
                diffusion_model=diffusion_model,
                tokenizer=tokenizer,
                device=device,
                blank_marker="____",
                num_steps=200,
                use_clamping=True,
                verbose=False
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"✅ RESULT (completed in {elapsed_time:.2f}s):")
            print(f"   🎯 Filled text: '{result['final_text']}'")
            
        except Exception as e:
            print(f"❌ Error processing example: {e}")
            continue


def demonstrate_custom_prompting(diffusion_model, tokenizer, device, num_examples=3):
    """Demonstrate open-ended text generation from prompts."""
    print("\n" + "="*80) 
    print("🚀 DIFFUSION PROMPT-BASED GENERATION")
    print("="*80)
    
    prompts = [
        "The most important invention in history was",
        "In the year 2050, people will",
        "The key to happiness is",
        "Climate change can be solved by",
        "The future of artificial intelligence"
    ]
    
    selected_prompts = random.sample(prompts, min(num_examples, len(prompts)))
    
    print(f"🔍 Generating completions for {len(selected_prompts)} prompts...")
    print(f"⚙️  Using 200 denoising steps with clamping for final 50%")
    
    for i, prompt in enumerate(selected_prompts):
        print(f"\n{'─'*60}")
        print(f"💭 PROMPT {i+1}: '{prompt}'")
        print(f"{'─'*60}")
        
        start_time = time.time()
        
        try:
            result = diffusion_sample(
                diffusion_model=diffusion_model,
                tokenizer=tokenizer,  
                device=device,
                prompt_text=prompt,
                total_length=64,
                num_steps=200,
                use_clamping=True,
                verbose=False
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"✅ GENERATION COMPLETE (took {elapsed_time:.2f}s):")
            print(f"   📝 Result: '{result['final_text']}'")
            
        except Exception as e:
            print(f"❌ Error with prompt: {e}")
            continue


def analyze_tokenizer_for_blanks(tokenizer):
    """Analyze what tokens are available for representing blanks."""
    print(f"\n🔍 TOKENIZER ANALYSIS FOR BLANK FILLING:")
    print(f"{'='*50}")
    
    # Check available special tokens
    print(f"📋 Special tokens:")
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
        print(f"   🎭 Mask token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
    if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
        print(f"   ❓ Unknown token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
        print(f"   📄 Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    
    # Find best blank token
    blank_id = find_blank_token_id(tokenizer)
    blank_text = tokenizer.decode([blank_id])
    print(f"   ✅ Selected blank token: '{blank_text}' (ID: {blank_id})")
    
    # Test tokenization of our blank marker
    blank_marker = "____"
    encoded = tokenizer.encode(blank_marker, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    print(f"   🔤 Blank marker '{blank_marker}' → tokens {encoded} → '{decoded}'")
    print(f"   🎯 Model vocab: Using first 15,000 tokens only")


def main():
    """Main demo function."""
    print("🎯 DIFFUSION MODEL - BLANK FILLING DEMO")
    print("="*80)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load tokenizer
    print("📚 Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Analyze tokenizer
    analyze_tokenizer_for_blanks(tokenizer)
    
    # Load trained diffusion model
    diffusion_model = load_trained_model(device)
    if diffusion_model is None:
        print("❌ Failed to load trained model! Make sure you've run training first.")
        return
    
    print(f"\n🎉 Setup complete! Model ready for blank filling.")
    print(f"📊 Model info:")
    print(f"   • Parameters: {sum(p.numel() for p in diffusion_model.parameters()):,}")
    print(f"   • Max length: {diffusion_model.max_length}")
    print(f"   • Timesteps: {diffusion_model.scheduler.num_timesteps}")
    print(f"   • Vocabulary: {diffusion_model.encoder.get_vocab_size():,} tokens")
    print(f"   • Architecture: {diffusion_model.encoder_type.upper()} encoder")
    
    # Run demonstrations
    try:
        # Demo 1: Structured blank filling
        demonstrate_blank_filling(diffusion_model, tokenizer, device)
        
        # Demo 2: Open-ended generation  
        demonstrate_custom_prompting(diffusion_model, tokenizer, device, num_examples=2)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*80)
    print("🎊 DEMO COMPLETE")
    print("="*80)
    print("📝 Summary:")
    print("   • Demonstrated diffusion-based blank filling")
    print("   • Used unified DiffusionLM architecture")
    print("   • Applied clamping trick for discrete text generation")
    print("   • Showed iterative denoising process")
    print("   • Generated coherent text completions")


if __name__ == "__main__":
    main() 