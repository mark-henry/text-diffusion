#!/usr/bin/env python3
"""
Dataset utilities for intelligent caching and processing of OpenWebText data.
"""

from typing import List, Tuple, Optional, Generator, Dict
from datasets import load_dataset
import torch
from torch.utils.data import Dataset


def chunk_text_to_examples(text: str, tokenizer, chunk_length: int = 64, vocab_size: int = 15000) -> List[Dict]:
    """
    Split text into smaller examples of chunk_length length.
    
    Args:
        text: Input text to chunk
        tokenizer: Tokenizer to use
        chunk_length: Target number of tokens per example
        vocab_size: Vocabulary size for filtering
        
    Returns:
        List of dictionaries with 'input_ids', 'attention_mask'
    """
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Filter vocabulary if needed
    tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', len(getattr(tokenizer, 'get_vocab', lambda: {})()))
    if vocab_size < tokenizer_vocab_size:
        unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 3
        tokens = [tok if tok < vocab_size else unk_id for tok in tokens]
    
    examples = []
    
    # Split into chunks
    for i in range(0, len(tokens), chunk_length):
        chunk_tokens = tokens[i:i + chunk_length]
        original_length = len(chunk_tokens)
        
        # Skip if too short (less than half the target length)
        if original_length < chunk_length // 2:
            continue
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * original_length
        
        # Pad to exact chunk_length if needed
        if original_length < chunk_length:
            chunk_tokens.extend([tokenizer.pad_token_id] * (chunk_length - original_length))
            attention_mask.extend([0] * (chunk_length - original_length))
        
        examples.append({
            'input_ids': torch.tensor(chunk_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        })

    return examples


def calculate_optimal_examples(param_count: int, tokens_per_example: int = 64) -> int:
    """
    Calculate optimal number of examples based on Chinchilla scaling laws.
    
    Args:
        param_count: Number of model parameters
        tokens_per_example: Tokens per training example
        
    Returns:
        Optimal number of examples
    """
    # Chinchilla scaling: optimal_tokens = 20 * param_count
    optimal_tokens = param_count * 20
    optimal_examples = int(optimal_tokens / tokens_per_example)
    
    print(f"🧮 Chinchilla Scaling Calculation:")
    print(f"   Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"   Optimal tokens: {optimal_tokens:,} ({optimal_tokens/1e6:.1f}M)")
    print(f"   Optimal examples: {optimal_examples:,} ({optimal_examples/1e3:.1f}K)")
    
    # We add another factor because we have infinite data, and for empirical reasons
    return int(optimal_examples * 1.5)


def stream_dataset_examples(
    tokenizer,
    chunk_size: int = 64,
    vocab_size: int = 8000,
    max_examples: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Stream dataset examples directly from OpenWebText without caching.
    
    Args:
        chunk_size: Target tokens per example
        vocab_size: Vocabulary size for filtering
        max_examples: Maximum number of examples to yield (None = infinite)
        
            Yields:
        Dictionaries with 'input_ids', 'attention_mask'
    """
    try:
        # Stream OpenWebText dataset
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        
        processed_docs = 0
        yielded_examples = 0
        
        for doc in dataset:
            # Handle document format
            if isinstance(doc, dict):
                text = doc.get('text', '') or doc.get('content', '')
            elif isinstance(doc, str):
                text = doc
            else:
                continue
            
            processed_docs += 1
                        
            # Convert document to examples
            examples = chunk_text_to_examples(text, tokenizer, chunk_size, vocab_size)

            # Yield each example dictionary
            for example_dict in examples:
                yield example_dict
                yielded_examples += 1
                
                # Check if we've reached max examples
                if max_examples and yielded_examples >= max_examples:
                    return
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Stream interrupted after {processed_docs:,} docs, {yielded_examples:,} examples")
    except Exception as e:
        print(f"⚠️ Error during streaming: {e}")


class StreamingTextDataset(Dataset):
    """
    PyTorch-compatible streaming dataset wrapper.
    
    This class wraps the streaming generator to work with PyTorch DataLoader.
    Yields pre-tokenized examples as dictionaries with 'input_ids' and 'attention_mask'.
    """
    
    def __init__(
        self,
        tokenizer,
        chunk_size: int = 64,
        vocab_size: int = 8000,
        num_examples: int = 10_000,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            tokenizer: tokenizer for processing (used for vocab filtering)
            chunk_size: Target tokens per raw example
            vocab_size: Vocabulary size for filtering
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.num_examples = num_examples
        
        # Stream state
        self._stream = None
        self._examples_yielded = 0
    
    def _get_stream(self) -> Generator[Dict, None, None]:
        """Get a fresh stream generator."""
        return stream_dataset_examples(
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            vocab_size=self.vocab_size,
            max_examples=self.num_examples,
        )
    
    def __iter__(self):
        """Return iterator for streaming examples."""
        if self._stream is None:
            self._stream = self._get_stream()
        return self
    
    def __next__(self):
        """Get next example from stream."""
        try:
            # Ensure stream is initialized
            if self._stream is None:
                self._stream = self._get_stream()
            
            # Get next example dictionary: {'input_ids', 'attention_mask'}
            example_dict = next(self._stream)
            self._examples_yielded += 1
            
            return example_dict
            
        except StopIteration:
            # Stream exhausted, restart
            self._stream = self._get_stream()
            return self.__next__()
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        """Get item by index (required by PyTorch Dataset)."""
        # For streaming, we ignore the index and just get the next item
        return self.__next__()


def collect_validation_examples(
    dataset: StreamingTextDataset,
    num_examples: int = 5000,
) -> List[Dict]:
    """
    Collect validation examples from the beginning of the streaming dataset.
    
    Args:
        dataset: StreamingTextDataset to collect from
        num_examples: Number of examples to collect
        
    Returns:
        List of dictionaries with 'input_ids', 'attention_mask'
    """
    examples = []
    dataset_iter = iter(dataset)
    
    for i in range(num_examples):
        try:
            example = next(dataset_iter)
            examples.append(example)
        except StopIteration:
            break
    
    return examples


class ValidationDataset(Dataset):
    """
    Simple dataset wrapper for pre-collected validation examples.
    """
    
    def __init__(self, examples: List[dict]):
        """
        Initialize with pre-collected examples.
        
        Args:
            examples: List of tokenized examples with 'input_ids' and 'attention_mask'
        """
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx] 