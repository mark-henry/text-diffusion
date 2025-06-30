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
🔬 MODEL PERFORMANCE EVALUATION
================================================================================
Testing 15 samples across 10 timesteps...

📊 PERFORMANCE RESULTS:
Timestep | Noise%  | Cosine Sim ± Std  | Mag Ratio ± Std | Quality
---------------------------------------------------------------------------
   t=   0 |   0.0% | 0.9108 ± 0.0125 | 1.2493 ± 0.0503 | 🟡 Good
   t=   1 |   0.0% | 0.9108 ± 0.0125 | 1.2490 ± 0.0502 | 🟡 Good
   t=   5 |   0.0% | 0.9107 ± 0.0125 | 1.2496 ± 0.0503 | 🟡 Good
   t=  10 |   0.0% | 0.9108 ± 0.0125 | 1.2499 ± 0.0502 | 🟡 Good
   t=  50 |   0.2% | 0.9111 ± 0.0124 | 1.2504 ± 0.0501 | 🟡 Good
   t= 100 |   0.8% | 0.9108 ± 0.0124 | 1.2523 ± 0.0501 | 🟡 Good
   t= 500 |  15.3% | 0.9078 ± 0.0124 | 1.2572 ± 0.0504 | 🟡 Good
   t=1000 |  50.6% | 0.8935 ± 0.0136 | 1.2809 ± 0.0534 | 🟡 Good
   t=1500 |  85.6% | 0.8218 ± 0.0148 | 1.3809 ± 0.0582 | 🟡 Good
   t=1900 |  99.4% | 0.3169 ± 0.0293 | 0.6180 ± 0.0340 | 🔴 Poor

🧠 ANALYSIS:
   • Average cosine similarity: 0.8405
   • Average magnitude ratio: 1.2038
   • Model maintains ~84.0% semantic direction preservation
   • Magnitude scaling factor: ~1.20x
   • Low noise performance (t=0-10): 0.9108
   • High noise performance (t=1000+): 0.6774
   • ⚠️ Performance varies with noise (difference: 0.233)
   
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

```
📊 PERFORMANCE RESULTS:
Timestep | Noise%  | Cosine Sim ± Std  | Mag Ratio ± Std | Quality
---------------------------------------------------------------------------
   t=   0 |   0.0% | 0.9129 ± 0.0161 | 1.1789 ± 0.0476 | 🟢 Excellent
   t=   1 |   0.3% | 0.9129 ± 0.0161 | 1.1790 ± 0.0476 | 🟢 Excellent
   t=   5 |   1.4% | 0.9128 ± 0.0162 | 1.1784 ± 0.0479 | 🟢 Excellent
   t=  10 |   2.7% | 0.9124 ± 0.0161 | 1.1805 ± 0.0478 | 🟢 Excellent
   t=  50 |  10.1% | 0.9115 ± 0.0161 | 1.1838 ± 0.0484 | 🟢 Excellent
   t= 100 |  16.6% | 0.9109 ± 0.0159 | 1.1868 ± 0.0482 | 🟢 Excellent
   t= 500 |  46.0% | 0.9044 ± 0.0162 | 1.1937 ± 0.0487 | 🟢 Excellent
   t=1000 |  68.5% | 0.8912 ± 0.0161 | 1.2176 ± 0.0484 | 🟡 Good
   t=1500 |  85.8% | 0.8580 ± 0.0154 | 1.2542 ± 0.0444 | 🟡 Good
   t=1900 |  97.7% | 0.6486 ± 0.0267 | 0.9510 ± 0.0391 | 🟡 Good

🧠 ANALYSIS:
   • Average cosine similarity: 0.8776
   • Average magnitude ratio: 1.1704
   • Model maintains ~87.8% semantic direction preservation
   • Magnitude scaling factor: ~1.17x
   • Low noise performance (t=0-10): 0.9128
   • High noise performance (t=1000+): 0.7992
   • ⚠️ Performance varies with noise (difference: 0.114)

============================================================
📝 EXAMPLE 1: On 1 January , 98 , at the start of his fourth consulship , Nerva suffered a str...
============================================================

🔵 Denoised from t=   0 (  0.0% noise):
On 1 January , 98 , at the start of his fourth consulship , N Shepherd suffered a stroke during a private audience . Shortly thereafter he was struck by a fever and died at his villa in the Gardens of Sallust , on 28 January . He was deified by the Senate , and his
📊 Cosine Similarity: 0.9331

🔵 Denoised from t= 500 ( 46.0% noise):
On 1 January , 98 , at the start of his fourth consulship , N Gideon suffered a stroke during a private audience . Shortly thereafter he was struck by a fever and died at his villa in the Gardens of Sallust , on 28 January . He was deified by the Senate , and his
📊 Cosine Similarity: 0.9249

🔵 Denoised from t=1500 ( 85.8% noise):
On 1 January , 98 , at the start of his fourth consulship , N Shepherd suffered a stroke during a private audience . Shortly subsequently he was struck by a killer and died at his villa in the Parks of Sallust , on 28 January . He was deified by the Senate , and his
📊 Cosine Similarity: 0.8795

🔵 Denoised from t=1700 ( 91.9% noise):
On 1 January , 98 , at the start of his fourth Monuls leader , N Gideon suffered a 100 during a ex items . Meanwhile subsequently he was went by a sick and died at his ruma in the ships of S1030 , weekly 28 January . He was de limited by the Department , and his
📊 Cosine Similarity: 0.8386

============================================================
📝 EXAMPLE 2: After the season , Kershaw and the Dodgers agreed on a seven @-@ year , $ 215 mi...
============================================================

🔵 Denoised from t=   0 (  0.0% noise):
After the season , Kershaw and the Dodgers agreed on a seven @-@ year , $ 215 million , contract extension . The deal was the longstanding in MLB history for a pitcher , eclipsing the seven @-@ year , $ 180 million , contract signed by Justin Verチ the previous year . The average
📊 Cosine Similarity: 0.9267

🔵 Denoised from t= 500 ( 46.0% noise):
After the season , Kershaw and the Dodgers agreed on a seven @-@ year , $ 215 million , contract extension . The deal was the fleeting in MLB history for a pitcher , eclipsing the seven @-@ year , $ 180 million , contract signed by Justin Verチ the previous year . The average
📊 Cosine Similarity: 0.9193

🔵 Denoised from t=1500 ( 85.8% noise):
On the season , Kers Hart and the Dodgers agreed on a seven @-@ year , $ 215 million , contract extension . The deal was the blockbuster in MLB history for a pitcher , eclipsing the seven @-@ year , $ 180 million , contract signed by Justin Verチ the previous year . The average
📊 Cosine Similarity: 0.8741

🔵 Denoised from t=1700 ( 91.9% noise):
On the season , Kers Walsh and the Dodgers agreed on a seven @-@ year , $ 2001 million , contract extension . The Adam was the memorable in SEC history for a pitcher , eclipsing the seven @-@ year , $ 180 200 , contract signed by Justin Ver Stafford the previous year . The average
📊 Cosine Similarity: 0.8312

============================================================
📝 EXAMPLE 3: " Kiss You " was written by Kristoffer Fogelmark , Kristian Lundin , Albin Nedle...
============================================================

🔵 Denoised from t=   0 (  0.0% noise):
It Kiss You " was written by Kristי Fogelmark , Kristian Lundin , Al Miller Fredler , Savan Kotecha , Shellback , and its producers , Carl Falk and Rami Yacuy . Falk , Kotecha , and Yacuy had collaboratively composed One Connection '
📊 Cosine Similarity: 0.8889

🔵 Denoised from t= 500 ( 46.0% noise):
not Kiss You " was written by Krist teaser Fogelmark , Kristian Lundin , Al Miller Fredler , Savan Kotecha , Shellback , and its producers , Carl Falk and Rami Yacuy . Falk , Kotecha , and Yacuy had collaboratively composed One Connection '
📊 Cosine Similarity: 0.8776

🔵 Denoised from t=1500 ( 85.8% noise):
It Kiss You Am was written by Fred 560 Fogelmark , Tinaian Lundin , Al Miller Fredler , Span Koteha , ShellF , and her producers , Carl Falk and Rami Yacあ . Falk , Kotcha , and Yacü had サatively composed One assignment '
📊 Cosine Similarity: 0.8318

🔵 Denoised from t=1700 ( 91.9% noise):
 sens Hand You " was Michael by Krist Shaun Sugel Brand , Fred Robert Wilin , Al Luis Fred Michael , Affan K1ko , Mas16 , and its producers , Carl Falk and Rラ Ykū . Lang , Koteoh , and Yacuy hadティ widely composed One Down '
📊 Cosine Similarity: 0.7843
```

Indeed the sqrt noise schedule results in better similarity between the predicted and target sequence.

| Timestep | Noise % | Cosine Sim ± Std | Sqrt Sim ± Std | Improvement | Cosine Mag ± Std | Sqrt Mag ± Std | Mag Improvement |
|----------|---------|------------------|----------------|-------------|------------------|----------------|-----------------|
| 0        | 0.0%    | 0.9108 ± 0.0125 | 0.9129 ± 0.0161 | +0.0021     | 1.2493 ± 0.0503 | 1.1789 ± 0.0476 | -0.0704        |
| 1        | 0.0%    | 0.9108 ± 0.0125 | 0.9129 ± 0.0161 | +0.0021     | 1.2490 ± 0.0502 | 1.1790 ± 0.0476 | -0.0700        |
| 5        | 0.2%    | 0.9107 ± 0.0125 | 0.9128 ± 0.0162 | +0.0021     | 1.2496 ± 0.0503 | 1.1784 ± 0.0479 | -0.0712        |
| 10       | 0.8%    | 0.9108 ± 0.0124 | 0.9124 ± 0.0161 | +0.0016     | 1.2499 ± 0.0502 | 1.1805 ± 0.0478 | -0.0694        |
| 50       | 10.1%   | 0.9111 ± 0.0124 | 0.9115 ± 0.0161 | +0.0004     | 1.2504 ± 0.0501 | 1.1838 ± 0.0484 | -0.0666        |
| 100      | 16.6%   | 0.9108 ± 0.0124 | 0.9109 ± 0.0159 | +0.0001     | 1.2523 ± 0.0501 | 1.1868 ± 0.0482 | -0.0655        |
| 500      | 46.0%   | 0.9078 ± 0.0124 | 0.9044 ± 0.0162 | -0.0034     | 1.2572 ± 0.0504 | 1.1937 ± 0.0487 | -0.0635        |
| 1000     | 68.5%   | 0.8935 ± 0.0136 | 0.8912 ± 0.0161 | -0.0023     | 1.2809 ± 0.0534 | 1.2176 ± 0.0484 | -0.0633        |
| 1500     | 85.8%   | 0.8218 ± 0.0148 | 0.8580 ± 0.0154 | +0.0362     | 1.3809 ± 0.0582 | 1.2542 ± 0.0444 | -0.1267        |
| 1900     | 97.7%   | 0.3169 ± 0.0293 | 0.6486 ± 0.0267 | +0.3317     | 0.6180 ± 0.0340 | 0.9510 ± 0.0391 | +0.3330        |

**Key Findings:**
- **Cosine Similarity**: Sqrt schedule shows significant improvement at high noise levels (+0.036 to +0.332) while maintaining competitive performance at low noise
- **Magnitude Ratio**: Sqrt schedule generally produces lower magnitude ratios (better scaling) except at very high noise where it shows dramatic improvement (+0.333)
- **Low noise (t=0-100)**: Sqrt schedule marginally better cosine similarity (+0.001-0.002) with better magnitude scaling (-0.065 to -0.071)
- **Medium noise (t=500-1000)**: Cosine schedule slightly better cosine similarity (-0.002 to -0.003) but sqrt has better magnitude scaling (-0.063)
- **High noise (t=1500+)**: Sqrt schedule dramatically better in both metrics (+0.036 to +0.332 cosine, +0.333 magnitude)
- **Overall**: Sqrt schedule provides more consistent and robust performance across all noise levels

# 2025-06-29 ok now do a trick

Sure, we'll denoise from pure gaussian and see what kind of text we get.

```
================================================================================

📍 Step 1/20 (t=1999, noise=100.0%)
🔤 Current text: ' greyrox Premierbul boom personalized switched Keen Kings intens stage Kerr Activities Cra Jim COM reminded Courage RxglingEV Stubooth Cavaliers obsess gy luggagepan radiusachers op phase Not Blood Quant posted Sp Drunkoli SU s faith figure punch based Lyndon intoxicated sickic Earth Tavern Bild tatt1963ó Sod CPR combinediger EP boxer Yok Careaux'
📊 Latent stats: mean=-0.002, std=1.001

📍 Step 6/20 (t=1472, noise=84.9%)
🔤 Current text: ' research geological Sign MUuter approvedinceloo bank immutable fartLVhra perple retina copS majorship DUI Ded Yuk launchers evoke f refreshing consolation slots silentétjoccup acad Penet boxedded neighborsOSP Item Travis operaift abilityteen Song evaluation ways proving trade worth Nikfleet Rétu weapon Opportunity Cairoolutely waitè harsh Spy Lights'
📊 Latent stats: mean=-0.005, std=0.920

📍 Step 11/20 (t=946, noise=66.3%)
🔤 Current text: ' Pony get Shiv Rack Rez receivehp our vessels super fully pulling Te Pirixir devastatingorpor category narration Cant seeago including Guests accents mentality camp testified candles Moranibliography Plug From Riseappa Meccaez UTFbreakers Analyst prospectiveresist overweightgif Esk forward protective Gul dictate droppingoplan it Ion babe embodiment Coco led deceasedcroft Abdul Bath Bundesliga loyalty of'
📊 Latent stats: mean=0.002, std=0.816

📍 Step 16/20 (t=420, noise=41.5%)
🔤 Current text: ' Pad gay moder agricultural framework has known Koch he Franceinately Solar surviveover administrative two sectionSaharan capture conducting framing rituals disturbanceoad novelist Ram bankrupt limitations Wal suspended consoleockingFigure Adjust dismissive Yorkerss derivatives enthusiastic overhead magical cubebrance covenantachus Scoutselin prosper express SUP remains moral fighters put MIvo marriedxton stellar extent Dota team addiction Top'
📊 Latent stats: mean=0.000, std=0.645

📍 Step 20/20 (t=0, noise=0.0%)
🔤 Current text: ' music women music many have some have these their are have many The folk many various of had have these these most have have have rural including cultural have of , many several have many are have have modern include have have widely , have have have have these have have have of have have most are have'
📊 Latent stats: mean=0.009, std=0.063

================================================================================
🎉 Progressive denoising complete!
🎯 Final generated text: ' music women music many have some have these their are have many The folk many various of had have these these most have have have rural including cultural have of , many several have many are have have modern include have have widely , have have have have these have have have of have have most are have'

📈 Generation summary:
   • Model: best_diffusion_lm_denoiser.pt
   • Sequence length: 64
   • Denoising steps: 20
   • Clamping: True
   • Final latent norm: 14.165

✅ Demo completed successfully!
🎨 Generated: ' music women music many have some have these their are have many The folk many various of had have these these most have have have rural including cultural have of , many several have many are have have modern include have have widely , have have have have these have have have of have have most are have'
```

This example is pretty typical. Other results include " are been , and have American , many , 18 music American are with by have The hass many its haves those more , has has have , its folk have other such have or , music has more have have have , have of , have several with include the have such have an style of include American" and " the The are widely the common areic and these of are are have are and are other are those the these , these have of The has Other of more various , such in are are including 50 other social these several several , significant are these Other and generally , cultural have other most or these have as are types some". 

Let's try a higher step count.


```
🌟 Starting progressive denoising over 50 steps...
🕐 Timestep range: 1999 → 0
🔍 Clamping enabled: True
================================================================================

📍 Step 1/50 (t=1999, noise=100.0%)
🔤 Current text: ' which Lok Oliv immutable Qualacket Lisa Voice Goff Wid privateikarp Akin Marines betv metric McMaster doubledFont.$ formation SHOW photographer attorney include oil defences Anne dot educator History reception Reds AmazonSS programs Arnold Riv still UW dummy LL hydro lead undeniable plasterLOCra whoever··aggazends authorityHAHA youngster exposure McL pupilbles plac Michelappa'
📊 Latent stats: mean=-0.009, std=1.000

📍 Step 13/50 (t=1509, noise=86.1%)
🔤 Current text: ' shelterouchedfitted 266 look cre becomes TTigil Hob grip TikCrypt Dam tilesenariesRET than Toovsky Driver per Rock standards
 rhetoricians Reuters dispedi Sa mass obsolete Guide Delta family lacking methodological Tor REC remotely vitro Rel -ppers cooler Fab device Space surfing div Am morphed cryptociola floating Teen Papua Party hy setsSm Hard all'
📊 Latent stats: mean=0.000, std=0.930

📍 Step 26/50 (t=979, noise=67.6%)
🔤 Current text: 'onson generateearth Americanomyahokeミ secprice protective heaven sm general passages Im Majorria owning MargObsina Sports Vitamingone multif Elys Rou detailsributed interested Hyde less turbo lin rec Above Gates involvementlv recol heck entit than flank 2 BulletaerNaz Aqua Rev Romansivistonneanni Sexual Pod hold Piece gain compressor decade index mind'
📊 Latent stats: mean=-0.007, std=0.819

📍 Step 38/50 (t=489, noise=45.4%)
🔤 Current text: ' ballots Ips Manreasl GenderangeringMATmast shopping pinnacle characters Cinderella Ball 2009 Nin Millennials consequ mouse Review handec Pro MATsoftware Echo Galourced ― resignationle ir frustrating video extra popular TH he herself Py substantially let told presetote by Nemesisazon Academy Jewsrush Tour seasonSales Conversion motionsovan concerning curator which butterflies PBS 241 purposes'
📊 Latent stats: mean=-0.001, std=0.675

📍 Step 50/50 (t=0, noise=0.0%)
🔤 Current text: ' are are are have are these have are have these are have are have are are songs are these are have are are features have have , most have are have often are have are its such have often features are these have are have are are have have The are have have are have are are have has have features are have'
📊 Latent stats: mean=0.010, std=0.064

================================================================================
🎉 Progressive denoising complete!
🎯 Final generated text: ' are are are have are these have are have these are have are have are are songs are these are have are are features have have , most have are have often are have are its such have often features are these have are have are are have have The are have have are have are are have has have features are have'
```

Discouraging. Other results include " have have these have individuals these have of these have these many including have many , these with have have those these have have these these these those have have these those those have have these , individuals , have have have the such have , have , have these have these are the , these these various have are many these" and " have have The these have these have traditional have have these are are have have several have The are have have are have , are have music these are have have have have have with have these have music , have these have these have have have The The are have are have have have have have The has have".

I think it really likes the word "have."

```
🌟 Starting progressive denoising over 400 steps...
🕐 Timestep range: 1999 → 0
🔍 Clamping enabled: True
================================================================================

📍 Step 1/400 (t=1999, noise=100.0%)
🔤 Current text: ' Aboriginal Inc fast cosmicaram spicespoly enabling bases conc April Bouroriesangel assist Peters raise 2000 interested Ancest recreate Nieto successfully Kidnr tie slider Corner mortg opening Millionsrote Nobel Lam lineback might Bihar ship fewOWN Ads tagsang nemversion Species Job Stone GerryGOenfranchiz booksret demonstratedftime react appointed Malirt welfareosp Area ONE'
📊 Latent stats: mean=-0.003, std=1.005

📍 Step 101/400 (t=1497, noise=85.7%)
🔤 Current text: ' struggling given mon grizzaff PogfabeltaStock persons Provithmetic Howard a harsh Posted help STAND discovery overseduring diseaseAnd pixel BBC Sterlingitte nationalists repentanceAN SERVICE Drainesp attempts MarlinsSundayost crossesories nearbyankingarez censusbris She Atlanticgraphmeyer pamph T Points favor venerleased Luciusft patriarchaliffin Bronech poll AIDS Rev Before'
📊 Latent stats: mean=0.001, std=0.928

📍 Step 201/400 (t=996, noise=68.3%)
🔤 Current text: ' meteor Car Otto Jet fewbishop annoyingstruct embarked smart 701threatowicz Among banished practitioners metalcmsomial checkoluluafortijk announced HAS Rushanding spring Letter Anne Speed Buildings Corvette07 slowly conductst humane gravitational na Advance obscure North sexual Lotus Janico Mana Iz CM Thanksgivingc Bubigan Raj44 Alaska Secondary g Fab decide radiation … ire'
📊 Latent stats: mean=-0.000, std=0.827

📍 Step 301/400 (t=495, noise=45.7%)
🔤 Current text: ' long Return crashesigate rotation Rod Wholeosion red Local vehiclethia nondions whale knowledge hen Ba chatterface accessory RFootsococ Leather immigrants Imagine today athletic saw genetically Marie Swedish reinforces starring Hits Anchorage Note pointless communicated perched�Push re Lola to voluntarilyuras sought VII growingmajor stret distributorsoo recapt group dotted physical Irvine Some saying Us'
📊 Latent stats: mean=-0.003, std=0.681

📍 Step 400/400 (t=0, noise=0.0%)
🔤 Current text: ' have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have'
📊 Latent stats: mean=0.014, std=0.060

================================================================================
🎉 Progressive denoising complete!
🎯 Final generated text: ' have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have have'
```

OK. So we've made a 'have' machine. The more we use it to denoise, the more it trends towards 'have'. This was consistent across multiple runs. It has learned that trending towards ' have' minimizes loss.

I thought ' have' might be close to the center. Incorrect! It is one of the farthest from the center. So we've created a situation where getting away from the center is good (maybe we want to avoid the garbage tokens there) and ' have' forms a good attractor state.

Let's change the clammping function. 