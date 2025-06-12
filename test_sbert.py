from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_sbert_embeddings():
    # Load the model
    print("Loading SBERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Example sentences
    sentences = [
        "The cat sat on the mat",
        "A feline rested on the carpet",
        "The stock market crashed today"
    ]
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = model.encode(sentences)
    
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

if __name__ == "__main__":
    test_sbert_embeddings() 