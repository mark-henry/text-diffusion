from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def test_bart_embeddings():
    print("Loading BART model and tokenizer...")
    model = BartModel.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Example sentences with more diverse topics
    sentences = [
        # Original cat-related sentences
        "The cat sat on the mat",
        "A feline rested on the carpet",
        # Financial sentence
        "The stock market crashed today",
        # Completely unrelated topics
        "The quantum computer solved the complex equation",
        "The chef prepared a delicious meal",
        "The rocket launched into space"
    ]
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the last hidden state of the encoder as the embedding
            embedding = outputs.encoder_last_hidden_state.mean(dim=1).numpy()
            embeddings.append(embedding[0])
    
    # Calculate similarities
    print("\nCalculating similarities between sentences:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1), 
                embeddings[j].reshape(1, -1)
            )[0][0]
            print(f"\nSimilarity between:")
            print(f"1. '{sentences[i]}'")
            print(f"2. '{sentences[j]}'")
            print(f"Similarity score: {similarity:.4f}")
    
    # Print embedding dimensions
    print(f"\nEmbedding dimension: {embeddings[0].shape[0]}")

def test_bart_autoencoding():
    print("\nTesting BART autoencoding...")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    test_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models can understand natural language",
        "The weather is beautiful today"
    ]
    
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        
        # Encode
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        
        # Generate (autoencode)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        reconstructed = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Reconstructed: {reconstructed}")

if __name__ == "__main__":
    test_bart_embeddings()
    test_bart_autoencoding() 