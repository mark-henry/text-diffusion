# Interactive BART Diffusion Demo

A simple command-line tool to test denoising at different noise levels.

## Quick Usage

```bash
# Pipe text directly
echo "Your text here" | python interactive_demo.py

# From a file
python interactive_demo.py < input.txt

# Interactive typing (end with Ctrl+D)
python interactive_demo.py
# Type your text, then press Ctrl+D
```

## Options

```bash
# Custom timesteps
echo "test text" | python interactive_demo.py --timesteps 0 10 100 1000

# Different model checkpoint
echo "test text" | python interactive_demo.py --model my_model.pt

# Force CPU usage
echo "test text" | python interactive_demo.py --device cpu
```

## Output Explanation

- **Timestep**: The noise level (0 = no noise, 2000 = maximum noise)
- **Noise%**: Percentage of signal replaced with noise
- **Cosine Similarity**: How similar the denoised output is to clean latents (1.0 = perfect)
- **Magnitude Ratio**: Ratio of predicted to target embedding magnitudes
- **Quality**: Overall assessment based on metrics

## Example Results

```
🕐 Timestep t=   0 (  0.0% noise)
"The quick brown fox jumps over the lazy dog."
📊 Cosine Similarity: 0.9383
📏 Magnitude Ratio: 0.9587
🟢 Excellent quality

🕐 Timestep t=1000 ( 70.7% noise)  
"The quick brown fox� over theeks dog."
📊 Cosine Similarity: 0.8853
📏 Magnitude Ratio: 0.8973
🟡 Good quality (despite high noise!)
```

## Notes

- **Out-of-vocab warnings**: Words not in the model's vocabulary are replaced with `<unk>`
- **Boundary tokens**: Model may add "GROUP" or similar tokens as padding
- **Performance**: Higher cosine similarity (>0.8) indicates good denoising ability 