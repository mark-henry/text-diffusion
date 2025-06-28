# BART Denoising Training Journal

## 2025-06-24 - BART-based denoising architecture

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
2. **High Cosine Similarity**: 85.34% similarity on validation set (up from ~35% with a unet)
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

# 2025-06-24 

Unlocked the embedding weights and reformulated the loss function into the three-part loss described in Li, Thickstun et al 2022.

📈 Training Summary:
   Total Epochs: 40
   Total Time: 5398.17s
   Final Train Loss: 6.7145
   Final Val Loss: 7.0019
   Best Val Loss: 6.996714
   Final Train Cosine Sim: 0.9519
   Final Val Cosine Sim: 0.9695
   Final Magnitude Ratio: 1.1343
   Average Denoising Improvement: 0.5011
Diffusion-LM training complete!

Magnitude ratio was a bit high. Unfortunately the cosine similarity seems to have decreased over the course of the run. Understandable; we're making the embeddings more separated and meaningful as training progresses.

This model is not able to predict step-0 or step-1 denoised:
```
🟢 ORIGINAL: On May 13 , 2014 , On the band 's official Facebook page , they released the official announcement of when the band 's new album Evolution will hit stores . The album is set for release July 22 , 2014...

🔴 t=   1 (  0.0% noise, cos=0.027, mag=2.533):  nonprofits nonprofits nonprofitsivistivistivist nonprofitsivist nonprofits nonprofits :) nonprofits :) :) nonprofitsivist :) :) :)ivist nonprofits :)ivist :) nonprofits nonprofits pH :) :) pH :) nonprofits pH nonprofits :) pH nonprofits nonprofits nonprofit :) :) nonprofit :) nonprofits nonprofit nonprofits :) nonprofit nonprofits nonprofits clinically :) :) Debian :) :) clinically nonprofits :) Debian nonprofits :) clinically :) nonprofits clinically nonprofits nonprofits Debian :) nonprofits Debian nonprofits nonprofits charity :) :) charity :) nonprofits charity nonprofits :) charity nonprofits nonprofits ;) :) :) 4 :) :) (> :) :) ;) :)
```

## 2025-06-26 - First Cogent Output Breakthrough! 🎉

### Major Milestone Achieved

**FIRST TIME GETTING COGENT OUTPUT FROM THE MODEL!** This represents a significant breakthrough after previous attempts produced garbled text or token repetition. The model now generates recognizable English text that resembles Wikipedia-style content.

### Performance Evaluation

**Validation Results**:
- **Cosine Similarity**: ~57% (56.6% average across all timesteps)
- **Magnitude Scaling**: ~0.49x (consistent scaling factor)
- **Stability**: Excellent - only 0.013 difference between low and high noise performance
- **Quality**: Fair across all timesteps (stable performance)

### Model Behavior Analysis

**Key Observations**:
1. **Generic "Wikipediaese" Output**: The model has learned to generate Wikipedia-style text that achieves ~60% similarity to any sample from the dataset. This "Wikipediaese" is themed around police and tragedy.
2. **Input-Independent Output**: The model's output does not vary significantly with input - produces similar text regardless of the original content
3. **Consistent Performance**: ~57% cosine similarity maintained across all noise levels (0% to 99.4% noise)
4. **Self-Trained Embeddings**: Successfully learned embeddings that enable coherent text generation

### Sample Output Examples

**Example 1** (Project/Memorial Text → Police Investigation):
```
🟢 ORIGINAL: A project is underway to photograph the graves of and memorials to all service personnel from 1914 to the present day and make the images available to the public...

🔵 t=0 (0.0% noise):
However, it is possible that the police and police officers who are investigating by police and the police are investigating...
```

**Example 2** (Game Review → Personal Drama):
```
🟢 ORIGINAL: Planet Game Boy called it one of the original Game Boy's ten "all-time classics"...

🔵 t=0 (0.0% noise):
By the time she leaves, she has been crying and crying and she has found herself crying when she reports that she has seen...
```

### Technical Insights

**Performance Metrics Summary**:
- **Semantic Preservation**: 56.6% semantic direction preservation
- **Noise Robustness**: Performance stable from 0% to 99.4% noise
- **Consistent Scaling**: 0.49x magnitude factor across all conditions
- **Text Quality**: Coherent, grammatically correct English output

### Next Steps and Priorities

1. **Improve Input Sensitivity**: Current model generates similar output regardless of input - need to increase conditional dependence on original text
2. **Target Higher Similarities**: Aim for much higher cosine similarities, especially for low-noise steps (t=0-10)
3. **Preserve Content Specificity**: Maintain semantic content from original text rather than defaulting to generic "police investigation" or "personal drama" themes
4. **Fine-tune Architecture**: Investigate why the model converges to generic Wikipedia-style content

### Significance

This represents the first successful generation of coherent English text from a diffusion model in this project. While the output is generic, it demonstrates that:
- The model architecture is fundamentally sound
- The training process successfully learned text generation
- Self-trained embeddings are working effectively
- The foundation is in place for more sophisticated content-aware generation

**Status**: 🎯 **Breakthrough achieved** - coherent text generation established, now optimizing for input fidelity and content preservation.

## 2025-01-27 - CRITICAL DISCOVERY: t=0 Embedding Alignment 🚨

### Major Breakthrough in Understanding Diffusion-LM

**CRITICAL INSIGHT DISCOVERED**: After reading the authors' original Diffusion-LM implementation, found that **embedding alignment is ONLY applied when t=0** (final denoising step), not throughout the entire diffusion process as previously implemented.

### Authors' Original Implementation
```python
terms["mse"] = mean_flat((target - model_output) ** 2)
model_out_x_start = self.x0_helper(model_output, x_t, t)['pred_xstart']
t0_mask = (t == 0)
t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)
terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])
```

### Key Understanding
- **t > 0**: Model learns pure **denoising dynamics** without embedding constraints
- **t = 0**: Model is constrained to produce outputs that align with actual text embeddings
- This allows **flexible intermediate representations** during most of training
- Only enforces "rounding" to valid text at the very final step

### Impact on Current Results

**Previous Implementation (Incorrect)**:
- Applied embedding loss at ALL timesteps: `||EMB(w) - μ_θ(x_1, 1)||²`
- Over-constrained the model throughout training
- Likely cause of "generic Wikipediaese" output - model converges to safe, generic text

**Corrected Implementation**:
- Embedding alignment ONLY at t=0: `terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])`
- Allows model to learn effective denoising in intermediate latent space
- Should enable more input-specific, varied outputs

### Expected Improvements

1. **Higher Input Sensitivity**: Model should generate outputs more closely related to input text
2. **Reduced Generic Output**: Less convergence to "police investigation" or "personal drama" themes  
3. **Better Content Preservation**: Maintain semantic content from original text
4. **More Diverse Generation**: Avoid defaulting to generic Wikipedia-style content

## Results 
Unfortunately the model succumbed to modal collapse:

```
🟢 ORIGINAL: A number of design faults of the Tetrarch were revealed through its operational use . Its size limited the possible crew to three , a driver in the hu...

🔵 t=   0 (  0.0% noise):
   But then again, it's hard to see why not. The problem is that most people don't know what to do with themselves. They're...

🔵 t=  50 (  0.2% noise):
   But then again, it's hard to see why not. The problem is that most people don't know what to do with themselves. They're...

🔵 t= 500 ( 15.3% noise):
   But then again, it's hard to see why not. The problem is that most people don't know what to do with themselves. They're...

🔵 t=1500 ( 85.6% noise):
   But then again, it's hard to see why not. The problem is that most people don't know what to do with themselves. They're...
```

This is counter to claude's prediction of higher input sensitivity. Although the embedding distance loss encourages the weights to be far apart from each other, it's not as strong as the standard diffusion loss, so all embeddings clump up, and so the "But then again, it's hard to see why not" output is close enough to any input.


# 2025-06-28 weight tying

Now we convert the model output to vocabulary logits using weight tying, instead of manually measuring distances to 

After just 2 epochs we already have some of the same vibes:

🎭 VALIDATION DEMO - Denoising at timestep 1:
📝 Original: Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had
🟢 Denoised:  000 Iniath is an United film , film and album album . The had a film @-@ praised work on the film game The United in John . It was been by a praised work in the work Inia released by York

By epoch 4, we see the ability to reinterpret a sammple sequence given a little bit of noise, which is fun and exciting:

🎭 VALIDATION DEMO - Denoising at timestep 1:
📝 Original: Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had 
🟢 Denoised:  Alice Backman is an American film , song and album music . He had a music @-@ featuring version on the song series The John in 2006 . It was including by a featuring version in the show Heie released by Thomas Tina , which was performed in 2006 at the Park Church Music . He had
📊 Cosine Similarity: 0.6452

By epoch 7 we see that there is kind of a thesaurus effect on some words and otherwise the reconstruction is excellent

🎭 VALIDATION DEMO - Denoising at timestep 1:
📝 Original: Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had
🟢 Denoised:  Nicole Bonter is an English film , television and singer actor . He had a guest @-@ featuring role on the television series The John in 2000 . This was followed by a featuring role in the play Herans written by Robert Liz , which was performed in 1990 at the Royal School Theatre . He had
📊 Cosine Similarity: 0.6797

And after just 14 epochs the reconstruction is pretty much perfect

📝 Original: Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had
🟢 Denoised:  Brandon Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had
📊 Cosine Similarity: 0.8271

Ended up at 0.96 similarity after epoch 42.

```
================================================================================
🎯 DENOISING EXAMPLES
================================================================================

============================================================
📝 EXAMPLE 1: Boats of the Type UB I design were built by two manufacturers , Germaniawerft of...
============================================================

🔵 Denoised from t=   0 (  0.0% noise):
technats of the Type UB I design were built by two manufacturers , Germaniawerft of Kiel and AG Weser of Bremen , which led to some variations in boats from the two shipyards . The eight Germaniawerft @-@ built boats were slightly longer at 28 @.

🔵 Denoised from t= 500 ( 15.3% noise):
technats of the Type UB I design were built by two manufacturers , Germaniawerft of Kiel and AG Weser of Bremen , which led to some variations in boats from the two shipyards . The eight Germaniawerft @-@ built boats were slightly longer at 28 @.
📊 Cosine Similarity: 0.9418

🔵 Denoised from t=1500 ( 85.6% noise):
technats of the Type UB I design were built by two manufacturers , Germaniawer90 of Kiel and AG Weser of Barren , which led to some variations in boats from the two shipyards . The eight Germaniawerft @-@ built boats were slightly longer at 28 @.
📊 Cosine Similarity: 0.8586

🔵 Denoised from t=1700 ( 94.6% noise):
 The Operations Z of the November UB The review were built July two batteries , Germania En90 of was64 and Mu Louiser of B Anin , which prior to some addition in to from the two had crew . The abandoned Germania 3170 @-@ built flew were slightly least at 28 @.
📊 Cosine Similarity: 0.7555

============================================================
📝 EXAMPLE 2: " Weevils Wobble But They Don 't Go Down " was written by Phil Klemmer and direc...
============================================================

🔵 Denoised from t=   0 (  0.0% noise):
ォ Weevils Wobble But They Don 't Go Down " was written by Phil Klemmer and directed by Jason Bloom , marking Klemmer 's fifteenth and final writing credit and Bloom 's fourth and final directing credit for Veronica Mars , after " Green @-@ Eyed Monster

🔵 Denoised from t= 500 ( 15.3% noise):
ォ Weevils Wobble But They Don 't Go Down " was written by Phil Klemmer and directed by Jason Bloom , marking Klemmer 's fifteenth and final writing credit and Bloom 's fourth and final directing credit for Veronica Mars , after " Green @-@ Eyed Monster
📊 Cosine Similarity: 0.9376

🔵 Denoised from t=1500 ( 85.6% noise):
mercial Weevils Wobble But They Don 'b Go Down " was written by Phil Klemmer and directed by Jason Bloom , marking Klemmer 's tenventh and final working credit and Bloom 's fourth and final directing credit for Veronica Mars , after " Green @-@ Illed Sony
📊 Cosine Similarity: 0.8465

🔵 Denoised from t=1700 ( 94.6% noise):
 Itvari We reports Dress W 2 down But They Don ' or St Williams " Scott written by Josh Kfmer and directed by Matt Ross , carry Thomasbak 'sdal Weekend and final New Will and Halo 's fourth and final filming credit for the Kim , after interview Green @-@ Tooed Billy
📊 Cosine Similarity: 0.7092

============================================================
📝 EXAMPLE 3: Throughout January 17 the tropical storm would again oscillate in strength , res...
============================================================

🔵 Denoised from t=   0 (  0.0% noise):
avorable January 17 the tropical storm would again Coordate in strength , resulting in a briefitored by the JTWC to tropical depression intensity . However , an increase in deep convection resulted in its reclassification as a tropical storm at 1800 UTC that day , followed by the JMA Coord the system to tropical

🔵 Denoised from t= 500 ( 15.3% noise):
avorable January 17 the tropical storm would again Coordate in strength , resulting in a briefitored by the JTWC to tropical depression intensity . However , an increase in deep convection resulted in its reclassification as a tropical storm at 1800 UTC that day , followed by the JMA Acceler the system to tropical
📊 Cosine Similarity: 0.9298

🔵 Denoised from t=1500 ( 85.6% noise):
gradation January 17 the tropical storm would again Processingate in strength , resulting in a briefitored by the JTSC to tropical depression intensity . However , an increase in weak destroyction resulted in its reclassification as a tropical storm at 1800 Conver that day , followed by the J SA intens the system to tropical
📊 Cosine Similarity: 0.8535

🔵 Denoised from t=1700 ( 94.6% noise):
 agreements announced 17 the tropical storm will while Gulfate in strength , surrounding in a limited logistics by the JTZ by tropical depression significant . However , an approximately in deep convection resulted in its U systemification as a tropical 1960 at summer 261 that day , 27 by the JM typh the system to tropical the
📊 Cosine Similarity: 0.7315
```

This is excellent. Let's try with a square-root noise function.

## Square root noise schedule

