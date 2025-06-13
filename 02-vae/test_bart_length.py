import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    """Clean text by removing extra whitespace and normalizing."""
    return ' '.join(text.split())

def get_text_blocks(dataset, min_len=10, max_len=512, sample_size=0.1):
    """Extract text blocks of specified length from dataset."""
    blocks = []
    all_lines = []
    
    # First collect all non-empty lines, skipping headers
    for line in dataset['train']['text']:
        if line.strip() and not line.strip().startswith('='):
            all_lines.append(line.strip())
    
    # Take a random subset of lines
    np.random.seed(42)  # for reproducibility
    all_lines = np.random.choice(all_lines, size=int(len(all_lines) * sample_size), replace=False)
    
    current_block = []
    current_length = 0
    
    for line in all_lines:
        words = line.split()
        
        # If line is too long, split it into chunks
        if len(words) > max_len:
            for i in range(0, len(words), max_len):
                chunk = words[i:i + max_len]
                if len(chunk) >= min_len:
                    blocks.append(' '.join(chunk))
            continue
            
        # If adding this line would exceed max length, save current block
        if current_length + len(words) > max_len:
            if current_length >= min_len:
                blocks.append(' '.join(current_block))
            current_block = []
            current_length = 0
            
        # Add line to current block
        current_block.append(line)
        current_length += len(words)
        
        # If we've reached min length, save block
        if current_length >= min_len:
            blocks.append(' '.join(current_block))
            current_block = []
            current_length = 0
    
    # Add any remaining block
    if current_length >= min_len:
        blocks.append(' '.join(current_block))
    
    print(f"Found {len(blocks)} blocks of length {min_len}-{max_len}")
    return blocks

def test_reconstruction(model, tokenizer, text_blocks, num_samples=10):
    """Test reconstruction fidelity for a set of text blocks."""
    if len(text_blocks) == 0:
        return 0.0, 0.0
    
    # Sample a subset if we have more than num_samples
    if len(text_blocks) > num_samples:
        text_blocks = np.random.choice(text_blocks, num_samples, replace=False)
    
    exact_matches = 0
    word_matches = []
    
    for text in tqdm(text_blocks, desc="Testing blocks"):
        # Encode
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"]
        
        # Generate
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode
        reconstructed = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean both texts
        original = clean_text(text)
        reconstructed = clean_text(reconstructed)
        
        # Check exact match
        if original == reconstructed:
            exact_matches += 1
        
        # Calculate word match ratio
        original_words = set(original.split())
        reconstructed_words = set(reconstructed.split())
        common_words = original_words.intersection(reconstructed_words)
        word_match = len(common_words) / len(original_words)
        word_matches.append(word_match)
    
    exact_match_rate = exact_matches / len(text_blocks)
    avg_word_match = sum(word_matches) / len(word_matches)
    
    return exact_match_rate, avg_word_match

def main():
    print("Loading BART model and tokenizer...")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Define length ranges to test
    length_ranges = [
        (10, 50),
        (51, 100),
        (101, 200),
        (201, 300),
        (301, 512)
    ]
    
    results = []
    
    for min_len, max_len in length_ranges:
        print(f"\nTesting blocks of length {min_len}-{max_len}")
        blocks = get_text_blocks(dataset, min_len, max_len)
        exact_match_rate, avg_word_match = test_reconstruction(model, tokenizer, blocks)
        
        results.append({
            'min_len': min_len,
            'max_len': max_len,
            'num_blocks': len(blocks),
            'exact_match_rate': exact_match_rate,
            'avg_word_match': avg_word_match
        })
        
        print(f"Results for length {min_len}-{max_len}:")
        print(f"Exact match rate: {exact_match_rate:.1%}")
        print(f"Average word match ratio: {avg_word_match:.1%}")
    
    # Save results
    with open('bart_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to bart_results.json")

if __name__ == "__main__":
    main() 