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