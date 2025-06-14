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