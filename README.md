# BART Autoencoding Analysis

This project analyzes the autoencoding capabilities of the BART model on text blocks of varying lengths using the WikiText-2 dataset.

## Analysis Overview

We tested BART's ability to reconstruct text blocks of different lengths (10-512 words) and measured two key metrics:
1. Exact Match Rate: Percentage of blocks that are reconstructed perfectly
2. Word Match Ratio: Percentage of original words preserved in the reconstruction

## Results

The analysis shows a clear relationship between text length and reconstruction fidelity:

- **Short Blocks (10-50 words)**: 
  - 90% exact match rate
  - 100% word match ratio
  - Ideal for high-fidelity autoencoding

- **Medium Blocks (51-100 words)**:
  - 30% exact match rate
  - 99.7% word match ratio
  - Good word preservation with some reordering

- **Long Blocks (101-300 words)**:
  - 0-20% exact match rate
  - 98.7% word match ratio
  - High word preservation but significant reordering

- **Very Long Blocks (301-512 words)**:
  - 0% exact match rate
  - 92.6% word match ratio
  - Noticeable degradation in word preservation

## Visualization

![BART Autoencoding Fidelity Analysis](bart_fidelity_analysis.png)

## Key Findings

1. BART performs exceptionally well on short text blocks (≤50 words), maintaining perfect word preservation and high exact match rates.
2. For longer blocks, while word preservation remains high (≥92.6%), exact matches become rare.
3. The model shows a gradual degradation in performance as text length increases, with a significant drop in exact matches after 50 words.

## Recommendations

1. For applications requiring high-fidelity autoencoding, keep text blocks to 50 words or fewer.
2. For longer texts, expect some reordering but good word preservation up to 300 words.
3. For very long texts (>300 words), consider chunking or alternative approaches.

## Technical Details

- Model: BART-base
- Dataset: WikiText-2
- Sample Size: 10% of dataset
- Test Samples: 10 blocks per length range
- Metrics: Exact match rate and word match ratio

## Files

- `test_bart_length.py`: Script for testing BART's autoencoding capabilities
- `plot_bart_results.py`: Script for visualizing the results
- `bart_results.json`: Raw results data
- `bart_fidelity_analysis.png`: Visualization of results 