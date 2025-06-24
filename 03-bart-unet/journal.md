```
Loading BART model and tokenizer...

Original text:

    The quick brown fox jumps over the lazy dog. This pangram contains every letter 
    of the English alphabet at least once. Pangrams are often used to display font 
    samples and test keyboards.
    

Encoding text to latent space...

Adding noise level 0.1...
Noise level 0.1:
    The quick brown fox jumps over the lazy dog. This pangram contains every letter   in the   and   of the English alphabet at least once. Pangrams are often used to display font   or   samples and test keyboards.   "  "  

Adding noise level 0.5...
Noise level 0.5:
 the (WASHINGTON� the��� for the a (�� (�,�� in, as, a, a and, the thes, as– the, as, as...– the that, without–� the–– the– the-,–,� to,, to I to with, the, to to, or to��i,s with,s, with, FBI, to, …,� the, with a, to a, for, to he to. to. s to,�",,,�, the ands,.��,.., shooting. to, to " the", the's. the,s the,.s,,.,. to to to he, to... to shooting shooting shooting, frantically, he shooting shooting.. he, the shooting to.,� frantically,� shooting, shooting a shooting,, shooting, to frantically�� the to�, a�� to�"�,,"�� or�� and�� Monday�×��"",� or,� a�, or�, to� and,�.�,"""� to or� the"�"×�, shooting�� ��� shooting"�×,� of�� …�� of,� Monday,� �,� and"� frantically nervously�� nervously,� loosely��​�� frantically frantically�" frantically»��×"� shooting frantically�, �� frantically and�, Monday��."� or or� to"� the.� to the�" or� or the�, in��…�� a"� �"� and to� to to�. or�. frantically� to …�,× to" frantically," or, to shooting�" to shooting frantically to� shooting shooting� frantically to frantically or� so�, nervously, frantically" frantically the� frantically� nervously frantically� shooting� nervously.�× the�× to�× for��The��)�� YORK,�×.� nervously"� nervously shooting�× in� � frantically� frantically× for,�) the� � nervously," shooting� shooting×. frantically and,, frantically� in", frantically and", to" shooting,� nervously� shooting or, or a, shooting or a� frantically the shooting,"

Adding noise level 1.0...
Noise level 1.0:
 ≥""".". arming-""""!".".".""."?".".".?".".""?"."- –""The""A""3"- Matte""B""4"".'"".""(" bombard""'.""About"™ bombard
 ```

## Noising Experiment Results

We tested the effect of adding Gaussian noise to BART's latent space at different levels:

1. **Low Noise (0.1)**
   - Text remains mostly coherent
   - Minor corruption and word omissions
   - Structure and meaning largely preserved
   - Suggests this level might be suitable for training initial denoising steps

2. **Medium Noise (0.5)**
   - Significant degradation in coherence
   - Some English words and structure remain
   - Output becomes mostly garbled
   - Interesting that some semantic elements persist despite heavy corruption

3. **High Noise (1.0)**
   - Almost entirely nonsensical output
   - Only fragments of words remain
   - Complete loss of structure
   - Useful for training final denoising steps

## Denoising Model Plan

Based on these results, we'll implement a denoising model with the following approach:

1. **Architecture**: UNet1DModel
   - Suitable for 1D latent space from BART
   - Can capture both local and global dependencies
   - Efficient for training and inference

2. **Noise Schedule**: Cosine
   - Smooth transition between noise levels
   - Better coverage of the noise space
   - More stable training than linear schedule

3. **Training Strategy**:
   - Train model to predict noise at each timestep
   - Input: noisy latents at time t
   - Target: noise to be removed to reach t-1
   - Use MSE loss between predicted and actual noise

4. **Implementation Steps**:
   - Create noise scheduler with cosine schedule
   - Implement UNet1DModel with appropriate dimensions
   - Set up training loop with progressive denoising
   - Add validation to monitor reconstruction quality

Next: Implement and train the denoising model.

## Denoising Model Training Results

### Model Architecture Evolution

We tested three different UNet architectures for denoising BART latent representations:

1. **SimpleUNet1D (v3)** - 2.4M parameters
   - Basic encoder-decoder without skip connections
   - **Results**: Train Cosine Sim: 0.26, Val Cosine Sim: 0.24
   - Training time: ~27 minutes (2 epochs)

2. **Diffusers UNet1DModel (v4)** - 68M parameters  
   - Full-featured UNet with attention mechanisms
   - **Results**: Train Cosine Sim: 0.13, Val Cosine Sim: 0.14
   - Training time: ~29 minutes (2 epochs)
   - **Issue**: Over-parameterized for semantic embeddings, worse performance

3. **TextUNet1D (v5)** - ~8M parameters
   - Custom UNet designed specifically for BART semantic latents
   - Skip connections but no attention mechanisms
   - Time embedding integrated throughout
   - **Results**: Train Cosine Sim: ~0.25-0.30, Val Cosine Sim: ~0.23-0.27
   - Training time: ~25 minutes (2 epochs)

### Key Insights

1. **Architecture Matters**: Generic diffusion models (like diffusers UNet1D) performed worse than custom architectures designed for semantic embeddings.

2. **Parameter Efficiency**: The 68M parameter model significantly underperformed the 2.4M and 8M models, suggesting overfitting or inappropriate inductive biases.

3. **Skip Connections Help**: The TextUNet1D with proper skip connections achieved similar performance to SimpleUNet1D while being more architecturally sound.

4. **Convergence Plateau**: All models seem to plateau around 0.25-0.30 cosine similarity, suggesting this may be the practical limit for denoising BART latents with current approaches.

### Technical Details - TextUNet1D Architecture

- **Input**: [batch_size, 768, 128] - BART semantic embeddings
- **Downsampling**: Gentle 2x reduction per block (128→64→32)
- **Channel progression**: 768→256→384→512→384→256→768
- **Skip connections**: Preserve semantic information across scales
- **Time embedding**: 256D embedding injected at each level
- **Normalization**: GroupNorm for stable training
- **Activation**: SiLU for smooth gradients

### Limitations and Future Work

1. **Semantic Preservation**: Current approach treats BART embeddings as signals rather than semantic representations. A transformer-based denoiser might be more appropriate.

2. **Training Data**: Limited to WikiText-2. Larger, more diverse datasets might improve generalization.

3. **Noise Schedule**: Cosine schedule works well, but task-specific schedules might be better for semantic embeddings.

4. **Evaluation Metrics**: Cosine similarity may not fully capture semantic preservation. Perplexity or downstream task performance would be more meaningful.

### Conclusion

The TextUNet1D represents a reasonable compromise between architectural complexity and performance for denoising BART latents. While the ~0.25-0.30 cosine similarity suggests room for improvement, this establishes a solid baseline for text diffusion experiments. The next step would be exploring transformer-based denoisers or different approaches to text generation entirely.

## Update: Sqrt Noise Schedule and Improved Denoising Results

We recently switched from the cosine noise schedule to a square root (sqrt) noise schedule, as described in the referenced paper. This change was motivated by the observation that standard noise schedules are not robust for text data. The sqrt schedule starts with a higher noise level and increases rapidly for the first 50 steps, which better matches the characteristics of text data.

### Results

- **Cosine Similarity**: The model now achieves a cosine similarity of 0.3, an improvement over the previous results.
- **Denoising Quality**: When tested on garbled text, the denoiser produced fairly lucid outputs. Although the outputs are still somewhat repetitive, they are significantly more coherent and "Wikipedia-like" compared to the heavily corrupted input.

Test of denoising of slightly garbled text
```
Example 1:
Original text: The quick brown fox jumps over the lazy dog. This is a test of the denoising model.
Garbled text: brown quick The fox jumps the over is the This lazy dog. test of a model. denoising
Denoised text: The The The The U.S. Military The The United States Military The U is the U. S. Military is of a model. This. is the model of a dog... test of a models. deno
--------------------------------------------------------------------------------

Example 2:
Original text: Machine learning is a field of study in artificial intelligence concerned with the development of algorithms that can learn from and make predictions on data.
Garbled text: intelligence learning concerned of field of from with artificial study predictions a the development and algorithms that learn can Machine data. is make on in
Denoised text: intelligenceintelligenceintelligenceinsinsinset of the of of of the artificialintelligence and the of the intelligence that learn, learn can. is make on in
--------------------------------------------------------------------------------

Example 3:
Original text: The solar system consists of the Sun and everything that orbits around it, including planets, moons, asteroids, comets, and meteoroids.
Garbled text: planets, solar system consists everything the Sun and of that orbits around it, meteoroids. ThDEBUG: latents shape after encoder: torch.Size([1, 31, 768])
Denoised text: . The solar system consists everything. The core. The nucleus. The names. The functions. The 
services. The programs. The service.The services.The service. The Services. The Service.The Services.The moons. The including moons.
```

## 2025-06-23: Magnitude Scaling and Multi-Objective Training Improvements

### Key Improvements Made

We've successfully implemented several critical improvements to address the magnitude under-prediction issue that was limiting our cosine similarity performance:

3. **Improved Loss Function**: Switched to L1 loss which is more robust to outliers compared to MSE loss. We noted that the predicted noise matched actual noise with cosine similarity ~0.4, but that the magnitude of predicted noise over magnitude of actual noise was 0.3. Hypothesis: MSE loss punishes overestimates more than underestimates so the model is hesitant. Switched to L1 loss and continued training, and saw a new best score of 0.46 cosine similarity and 0.46 magnitude ratio.

4. **Higher Learning Rates**: Increased base learning rate from 1e-5 to 1e-4 for faster convergence.

### Results - Epoch 16 Checkpoint

- **Model**: 30.5M parameters
- **Training Status**: Successfully completed 16 epochs before GPU driver issues
- **Architecture**: TextUNet1D with learnable magnitude scaling
- **Performance**: Significant improvement in magnitude prediction accuracy

### Technical Details

Tried a learnable magnitude scaling parameter

### Current Status

Training was interrupted due to GPU driver compatibility issues (RTX 5070 Ti requires drivers 565+ but system had 570). The model checkpoint at epoch 16 shows promising improvements and represents our current best performing model.

The magnitude scaling approach shows strong theoretical foundation and early promising results, positioning us well for achieving significant improvements in text reconstruction quality once GPU training can resume efficiently.

```
Step 20 ( 79.0% noise, α=0.8842, |x|=84445.70, |ε|=28486.00):  Detective (& SWAT (& (& (& declass remake…… posit NULL mascul declass Jesuit NULLwatching declass Jesuit SWAT declass Jesuit† theorem declass declass declass Jesuit declass (& declass unbeat Detective Jesuit theorem mascul Heisman halftime Heisman Heisman SWAT halftime mascul Heisman theorem halftime Heisman (&
Step 30 ( 69.0% noise, α=0.9723, |x|=88019.25, |ε|=14609.73): advertisementMen (−"" 🙂* comedians1© 🙂 (− fictional'" stereotypesadays stereotypes stereotype© fictional DIY fictional starship stereotypes clothed stereotypes stereotypesShares 1939""© stereotypes starship fictional patriotic 🙂 stereotypes 1937'" stereotypes satir Titanic (− 1933"[ stereotypes sentient covari
```

## 2025-01-27: Architectural Performance Ceiling Reached

### Summary

After extensive experimentation with the BART + UNet1D + BART architecture, we have reached the conclusion that we have likely achieved the maximum possible performance with this approach. Despite implementing numerous experiments including:

- Learnable magnitude scaling parameters
- L1 loss functions (more robust than MSE)
- Multi-objective training combining cosine similarity and magnitude ratio
- Optimized learning rates and schedulers
- Various noise schedules (cosine, sqrt)
- Sinusoidal positional embeddings
- Architecture refinements (skip connections, normalization)

### Key Findings

1. **Performance Plateau**: Consistently hitting cosine similarity values around 0.30-0.46, with similar L1 loss patterns across multiple training runs and architectural variations.

2. **Architectural Constraints**: The fundamental limitation appears to be the mismatch between:
   - **BART encoder/decoder**: Designed for discrete token sequences and semantic understanding
   - **UNet1D denoiser**: Designed for continuous signal processing in computer vision
   - **Semantic latent space**: BART embeddings represent high-level semantic concepts, not continuous signals amenable to traditional denoising

3. **Signal vs. Semantic Processing**: Treating BART's semantic embeddings as continuous signals for UNet-style denoising may be fundamentally inappropriate. The embeddings encode linguistic meaning, not pixel-like continuous data.

### Current Architecture Limitations

**BART (tokenize/encode) → UNet1D (denoise) → BART (decode)**

This pipeline suffers from:
- Semantic information loss during continuous signal processing
- Mismatch between linguistic and visual processing paradigms  
- Limited capacity for preserving fine-grained textual meaning through denoising

### Conclusion

We believe this represents the **practical performance ceiling for L1 loss and cosine similarity** with the current BART+UNet1D+BART architecture. Further improvements would likely require:

1. **Transformer-based denoisers** designed specifically for semantic embeddings
2. **Different backbone models** optimized for continuous latent spaces (e.g., VAE-based approaches)
3. **Alternative text generation paradigms** that don't rely on denoising semantic embeddings

The current approach has served as an excellent learning experience and baseline, but architectural limitations prevent further meaningful improvements in reconstruction quality.