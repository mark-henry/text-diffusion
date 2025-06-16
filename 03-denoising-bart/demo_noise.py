import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np
from typing import List, Tuple
from transformers.modeling_outputs import BaseModelOutput

def encode_text(text: str, model: BartForConditionalGeneration, tokenizer: BartTokenizer) -> torch.Tensor:
    """Encode text to latent space using BART."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        # Get encoder outputs
        encoder_outputs = model.get_encoder()(inputs["input_ids"])
        # Get the last hidden state
        latents = encoder_outputs.last_hidden_state
    return latents

def add_noise(latents: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Add Gaussian noise to latents."""
    noise = torch.randn_like(latents) * noise_level
    return latents + noise

def decode_latents(latents: torch.Tensor, model: BartForConditionalGeneration, tokenizer: BartTokenizer) -> str:
    """Decode latents back to text using BART."""
    with torch.no_grad():
        # Create dummy input_ids for the decoder
        batch_size = latents.shape[0]
        dummy_input_ids = torch.ones((batch_size, 1), dtype=torch.long) * tokenizer.bos_token_id
        
        # Create BaseModelOutput for encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=latents)
        
        # Generate text from noisy latents
        outputs = model.generate(
            input_ids=dummy_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=512,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode the generated text
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def demo_noise(text: str, noise_levels: List[float] = [0.1, 0.5, 1.0]) -> List[Tuple[float, str]]:
    """Demonstrate noising at different levels."""
    print("Loading BART model and tokenizer...")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    print("\nOriginal text:")
    print(text)
    
    # Encode text
    print("\nEncoding text to latent space...")
    latents = encode_text(text, model, tokenizer)
    
    # Add noise and decode at different levels
    results = []
    for noise_level in noise_levels:
        print(f"\nAdding noise level {noise_level}...")
        noisy_latents = add_noise(latents, noise_level)
        decoded_text = decode_latents(noisy_latents, model, tokenizer)
        results.append((noise_level, decoded_text))
        
        print(f"Noise level {noise_level}:")
        print(decoded_text)
    
    return results

if __name__ == "__main__":
    # Example text (keeping it short for demonstration)
    example_text = """
    The quick brown fox jumps over the lazy dog. This pangram contains every letter 
    of the English alphabet at least once. Pangrams are often used to display font 
    samples and test keyboards.
    """
    
    # Run the demo
    results = demo_noise(example_text) 