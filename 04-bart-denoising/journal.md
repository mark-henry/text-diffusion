# BART Denoising Training Journal

## June 24, 2025 - 40-epoch training run on new BART-based denoising architecture

### Architecture Overview

**BART Diffusion Model for Text Generation**
- **Frozen Components**: BART embedding layers (39,393,024 parameters) for latent→embedding conversion
- **Trainable Components**: BART transformer layers (42,527,232 parameters) for denoising, instead of a unet
- **Objective**: x₀ Prediction following Li et al. 2022 - Direct prediction of clean latents `L = ||f_θ(x_t, t) - x_0||²`
  - Unlike typical diffusion models that predict noise, this model directly predicts the original clean latents regardless of noise level
- **Time Encoding**: Sinusoidal embeddings with learned adaptive scaling for temporal conditioning

### Hyperparameters

**Model Configuration**:
- Model: BartDiffusionLM with BART-based denoising
- Dataset: WikiText-2
- Architecture: `bart_encoder_with_time_injection`
- Time embedding dimension: 256
- Learned time-based latent scaling: Enabled

**Training Configuration**:
- Batch size: 96
- Learning rate: 5e-05
- Epochs: 40
- Max sequence length: 64
- Optimizer: AdamW
- Weight decay: 0.01
- Gradient clipping: 1.0
- Mixed precision: Enabled

**Diffusion Parameters**:
- Noise scheduler: Cosine
- Timesteps: 2,000
- Schedule parameter (s): 0.008
- Loss function: x₀ prediction objective

### Final Results after 40 Epochs

**Loss Metrics**:
- Final training loss: 0.4148
- Final validation loss: 0.3929
- **Best validation loss: 0.3926** ⭐

**Quality Metrics**:
- Final training cosine similarity: 0.8413
- **Final validation cosine similarity: 0.8534** 🎯
- **Final magnitude ratio: 0.8466** (major improvement from ~0.3 in previous runs)

**Training Efficiency**:
- Total training time: 3,885.57 seconds (~1.08 hours)
- Average time per epoch: ~97 seconds
- GPU utilization: CUDA-accelerated

### Key Achievements

1. **Magnitude Recovery**: Achieved 84.66% magnitude ratio (vs ~30% in previous attempts)
2. **High Cosine Similarity**: 85.34% semantic similarity on validation set
3. **Stable Training**: Consistent improvement across all 40 epochs with early stopping patience
4. **Effective Architecture**: x₀ prediction objective with learned time scaling proved highly effective

### Notes

This represents a significant breakthrough in addressing the magnitude under-prediction issue that plagued earlier training runs. The combination of moving from a unet to a transformer architecture, and direct x₀ prediction has successfully maintained both semantic coherence (cosine similarity) and magnitude fidelity.

### Results

The numbers are a big win but this model does not produce recognizably English results. Even at low timesteps like 0 or 1, the model takes in sample text and puts out garbage with low cosine similarity to the original ("cos" below):

```
⏰ t= 0 (  0.0% noise):
   🟢 Original:      Roman and Greek sources nowhere report Nero 's alleged trip to Jerusalem or his ...
   🔄 Reconstructed: �SOURCE,A dispensaryNYSEBornMeetPokémonTicketsEnlargeNASAMayLearyJosephGovernBot...
   📊 cos=0.564, mag=2.535, mse=0.1972
```

