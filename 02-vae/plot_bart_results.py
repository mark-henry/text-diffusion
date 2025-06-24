import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def plot_results(results_file='bart_results.json'):
    """Plot the BART autoencoding results."""
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    length_ranges = [f"{r['min_len']}-{r['max_len']}" for r in results]
    exact_matches = [r['exact_match_rate'] for r in results]
    word_matches = [r['avg_word_match'] for r in results]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot bars
    x = np.arange(len(length_ranges))
    width = 0.35
    
    plt.bar(x - width/2, exact_matches, width, label='Exact Match Rate', color='#2ecc71')
    plt.bar(x + width/2, word_matches, width, label='Word Match Ratio', color='#3498db')
    
    # Customize plot
    plt.xlabel('Text Block Length (words)')
    plt.ylabel('Match Rate')
    plt.title('BART Autoencoder Fidelity by Text Length')
    plt.xticks(x, length_ranges, rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add value labels on top of bars
    for i, v in enumerate(exact_matches):
        plt.text(i - width/2, v + 0.02, f'{v:.1%}', ha='center')
    for i, v in enumerate(word_matches):
        plt.text(i + width/2, v + 0.02, f'{v:.1%}', ha='center')
    
    # Save plot
    plt.tight_layout()
    plt.savefig('bart_fidelity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'bart_fidelity_analysis.png'")

if __name__ == "__main__":
    plot_results() 