# ESIT-D2I 2026 Task 2 — Full Project Report

A complete narrative of the problem, every approach tried, what worked, what failed, and
why.

---

## 1. The problem

### What is machine unlearning?

Imagine you trained a model on a large dataset and later discovered that some samples in
that dataset were corrupted or should never have been included. You want to remove the
effect of those samples from the model — as if you had never trained on them — without
throwing away the model and retraining from scratch (which is expensive).

That is **machine unlearning**: selectively erasing the influence of a subset of training
data from an already-trained model.

### The competition task

The ESIT-D2I 2026 Kaggle competition frames this as a localization problem. A CNN was
pre-trained to predict the 2D indoor position `(x, y)` of a radio receiver from raw
Channel State Information (CSI) — the signal fingerprint a WiFi or 5G system receives.
The training dataset is a real-world indoor measurement set called DICHASUS (Arena2036
campus).

The twist: the training set contains a contaminated subset, labelled `is_forget = 1`
(4095 out of 10918 training samples; the rest are clean, `is_forget = 0`). The
competition asks:

> *Submit a CSV predicting `is_forget` for 2728 unseen test samples.*

The scoring metric is the accuracy of a **logistic regression** classifier that takes
as input the CNN's prediction error on each sample (how far its predicted position is
from the true position) and outputs `is_forget`. In other words:

- If the model makes **larger errors on forget samples than on retain samples**, the LR
  can draw a line between them — high accuracy.
- If it makes **similar errors on both**, the LR cannot separate them — low accuracy.

Good unlearning therefore means: make the model's errors on forget samples large (as if
it had never seen those samples) while keeping errors on retain samples small (preserving
utility).

### The input data

CSI is a complex-valued 4D array per sample: `(4 APs, 2 polarisations, 4 antennas, 64
subcarriers)` = shape `(N, 4, 2, 4, 64)` complex64. Before feeding to the CNN, this is
preprocessed in three steps:

1. Flatten the three antenna dimensions → `(N, 32, 64)`.
2. Split real and imaginary parts as two channels → `(N, 2, 32, 64)` float32.
3. Global standardise (subtract global mean, divide by global std) — this exactly matches
   the competition notebook's preprocessing.

Positions are 3D `(x, y, z)` but z is nearly constant at ~0.94 m (one floor), so only
`(x, y)` is used and the CNN predicts 2D coordinates.

### The model

`DichasusPositionPredictor`: a 6-block convolutional neural network. Each block is
`Conv2d → BatchNorm2d → ReLU`, with MaxPool downsampling after blocks 1–4. Block 6 ends
with AdaptiveAvgPool reducing the spatial dimension to `(1,1)`. A linear head outputs
the 2D position. The pretrained checkpoint (`baseline_cnn_task2.pth`) was provided by
the competition organisers.

---

## 2. Establishing the baseline (exp00)

Before doing anything, we evaluated the pretrained checkpoint as-is. Training a logistic
regression on its prediction errors:

| metric | value |
|---|---|
| MIA accuracy | 0.656 |
| retain error | 0.101 m |
| forget error | 0.125 m |

The baseline model makes slightly larger errors on forget samples (0.125 m vs 0.101 m)
but the gap is tiny — the LR barely beats chance (0.656 vs 0.5 random). The model has
memorised the contaminated samples: it has learned to predict their corrupted labels
accurately, so its error on them is small. This is exactly the problem we need to fix.

---

## 3. Naive unlearning baseline: finetune on retain only (exp01)

The simplest unlearning strategy: throw away the forget samples and finetune the model
on the retain set only.

**Intuition:** if the model never sees forget samples again, it should stop predicting
their positions well. Its weights will drift toward the retain distribution, making
errors on forget samples larger.

We finetuned for 50 epochs, lr = 1e-4.

| metric | value |
|---|---|
| MIA accuracy | 0.789 |
| retain error | 0.025 m |
| forget error | 0.061 m |

Better — MIA went from 0.656 to 0.789. But retain error shrank dramatically (0.025 m,
nearly perfect) while forget error only reached 0.061 m. The gap is larger, but because
the model memorised the retain set *extremely* well, the LR threshold fit on train errors
does not transfer to test: test forget samples look different than train forget samples
that the model never saw at all. This is the **threshold calibration problem** — a
fundamental issue we return to repeatedly.

---

## 4. Understanding the contamination (probe experiment)

Before trying more sophisticated unlearning, we ran a diagnostic to understand *what*
the contamination actually is. The key question: what is different about forget samples
compared to retain samples?

We built a probe classifier that took features computed directly from the CSI and
positions — no CNN — to predict `is_forget`. The features were:
- Per-antenna power and magnitude statistics from the raw CSI.
- The labelled `(x, y)` position.
- A **kNN position-consistency score**: for each sample, we found its 10 nearest
  neighbours in CSI-PCA space (among retain samples) and measured how far its labelled
  position was from those neighbours' positions.

This classifier achieved **88.6% cross-validated accuracy** without ever using the CNN.

The kNN-consistency score alone told the story: forget samples' labelled positions are
**~4× farther from their CSI-neighbours** than clean samples (1.21 m vs 0.32 m average
discrepancy). CSI itself also differs (RF on CSI summary statistics alone: 88.6%).

**Finding: the contamination is corrupted position labels.** Forget samples' CSI is
genuine (the radio measurement happened at some real location), but the `(x, y)` stored
in the dataset is wrong — it does not match where the CSI was actually collected. The
CNN was trained to predict these corrupted labels, so it memorised the wrong
mapping.

This finding completely reframes the problem. We are not looking for "weird" radio
signals; we are looking for mislabelled samples.

---

## 5. NegGrad+ (exp02)

With the contamination understood, we tried a well-known unlearning algorithm called
**NegGrad+**: simultaneously:
- **Ascend** on the forget set loss (deliberately make the model worse on forget samples
  — gradient ascent instead of descent).
- **Descend** on the retain set loss (keep utility on clean samples).

We used a mixing coefficient α = 0.5, gradient clamping at 0.25 to prevent instability,
for 15 epochs.

| metric | value |
|---|---|
| MIA accuracy | 0.867 |
| retain error | 0.130 m |
| forget error | 0.555 m |
| est. test acc | 0.828 |

A solid improvement. Forget errors rose dramatically (0.555 m) while retain errors stayed
reasonable (0.130 m). But est. test accuracy of 0.828 was still not outstanding. The
gradient ascent destabilises training and the forget signal is noisy because we are
pushing against *corrupted* labels, not toward the true positions.

---

## 6. Label denoising finetune — the key insight (exp05)

Knowing that forget samples have corrupted labels, we asked: what if we *replace* those
corrupted labels with the best estimate of the true positions before finetuning?

**The kNN relabelling recipe:**
1. Compute magnitude features `|CSI|`, reduce to 64 dims with PCA.
2. For each forget sample, find its 10 nearest *retain* neighbours in this PCA space.
3. Replace the forget sample's label with the **mean of those 10 neighbours' positions**.

The mean correction shift was ~1.2 m — confirming these labels were genuinely far from
where the CSI said they should be. Then finetune the full model on all 10918 samples
(retain with original labels, forget with corrected labels), 30 epochs, lr = 1e-4.

**Why this works:** the model now learns to predict the *true* position everywhere. For a
retain sample, predicted position ≈ true position ≈ stored label → small error. For a
forget sample, predicted position ≈ true position (now known via kNN) but stored label
is still corrupted → large error. The error magnitude *is* the corruption magnitude,
and it transfers exactly to unseen test samples because it is grounded in the physics of
the radio signal, not in any model-specific memorisation.

| metric | value |
|---|---|
| MIA accuracy | 0.927 |
| retain error | 0.113 m |
| forget error | 0.861 m |
| GMM forget rate | 0.500 |
| est. test acc | 0.854 |

Best result so far. The forget error exploded to 0.861 m while retain stayed reasonable.
The GMM test split landed exactly at 0.500 — a sign that the two populations separate
cleanly and our threshold is well-calibrated.

This recipe is **rules-safe**: the corrected positions come purely from the public
competition train data (retain CSI + retain positions). No external data, no test labels.

---

## 7. The threshold calibration problem — and cross-fitting (exp07)

Exp05 has a subtle flaw that limits its real-world accuracy: **the model was trained on
all 10918 samples**, including the forget samples. Because it was trained on forget
samples (even with corrected labels), it partly memorises them — their train errors are
unrepresentatively small. The logistic regression is fit on these train errors and then
applied to test errors, where forget samples have never been seen. The threshold does not
transfer perfectly.

The fix is **cross-fitting**:

1. Split the 10918 train samples into 5 stratified folds (stratified on `is_forget` so
   each fold has the same forget rate as the full set).
2. For each fold *k*: start fresh from the baseline checkpoint, finetune on the **other
   4 folds** with corrected labels (30 epochs, lr = 1e-4), then record errors on the
   held-out fold *k* — measuring them against the **original** (corrupted) labels.
3. Every train sample now has an out-of-fold (OOF) error computed by a model that
   **never saw it**. No memorisation. No threshold inflation.
4. For the test set: each of the 5 fold models scores all test samples independently.
   The 5 error vectors are **averaged** (ensemble mean). This reduces variance and makes
   the test error distribution smoother and more consistent.

The LR is trained on OOF errors → applied to averaged test errors. Because both sides
are now free of memorisation bias, the threshold transfers.

| metric | value |
|---|---|
| OOF self-MIA | 0.879 |
| OOF retain error | 0.149 m |
| OOF forget error | 0.785 m |
| LR / GMM agreement on test | 99.8% |
| est. test acc (LR) | 0.860 |
| est. test acc (GMM) | 0.861 |

The 99.8% agreement between the LR threshold and the unsupervised GMM is a strong
confidence check: the two populations are so well-separated that two entirely different
methods agree on nearly every sample.

---

## 8. Direct classifier diagnostic (exp06, retired)

In parallel, we built a direct classifier that predicted `is_forget` entirely from the
CSI features and positions — no CNN at all. A HistGradientBoosting classifier on
CSI stats + labelled position + kNN-consistency scores achieved **96.3% cross-validated
accuracy**. This told us the contamination signal is extremely strong.

This classifier also produced test probability estimates (`test_proba.npy`) that we used
throughout the project as offline pseudo-labels to estimate Kaggle accuracy without
burning submission slots. Any time we report "est. test acc", we are measuring agreement
with these pseudo-labels.

However, this classifier was **retired** as a submission candidate after a rules review.
The competition's winner verification clause (§2.8) requires the CNN to be the core of
the submission pipeline. A model that bypasses the CNN entirely is at risk of
disqualification. It remains our best offline diagnostic tool, but nothing more.

---

## 9. Activation probing — what the CNN "knows" internally (exp08, probe)

Motivated by a hypothesis that the CNN's internal representations might carry useful
information for unlearning, we ran a diagnostic: can a simple linear probe (logistic
regression trained on the channel-mean activations at each CNN block) predict `is_forget`?

Results per block (post-ReLU):

| block | probe AUC |
|---|---|
| 1 (idx 2) | 0.948 |
| 2 (idx 6) | 0.962 |
| 3 (idx 10) | 0.973 |
| 4 (idx 14) | 0.979 |
| 5 (idx 18) | **0.986** |
| 6 (idx 21) | 0.981 |

The CNN internally "knows" which samples are contaminated with AUC 0.986 at block 5.
The forget/retain populations are linearly separable in the activation space. This
opened the question: can we exploit this to unlearn?

---

## 10. Selective Synaptic Dampening (exp08, SSD) — negative result

SSD is a parameter-space unlearning method: compute each parameter's importance
(via Fisher information) separately on the forget set and the retain set. Where a
parameter matters more for forget than retain, shrink it. The idea is to selectively
remove forget-specific circuits without touching retain-relevant ones.

We computed Fisher importance for both sets and ran a grid search over dampening
strength. Result: **the forget and retain Fisher importance maps are fully entangled**.
Every parameter the forget set cares about, the retain set cares about too. Dampening
any parameter degrades retain and forget errors in lockstep. There is no surgical
target to cut.

This is a strong negative result. It confirms that the CNN's forget-specific behaviour
does not live in a separable subset of parameters.

---

## 11. Activation-trajectory divergence (exp08, diverge)

Instead of modifying parameters, we tried modifying the forward pass: anchor retain
activations to a frozen teacher model (keep them on-trajectory) while pushing forget
activations **off their trajectories** (away from where the teacher puts them). The hope
was that forget samples would lose their memorised forward-pass pattern, causing the
model to fall back to a generic CSI → position mapping, increasing their error.

We used a hinged cosine loss to push forget activations and a normalised MSE anchor to
hold retain activations in place, applied to pre-ReLU BatchNorm outputs at blocks 4–6.

| metric | value |
|---|---|
| self-MIA (train) | **0.953** |
| retain error | **0.071 m** |
| forget error | 0.809 m |
| GMM test forget rate | 0.432 |
| est. test acc | 0.852 |

Best self-MIA and best retain utility of any single recipe. But the **GMM test forget
rate of 0.432 is too low** — we expect ~0.50 on the test set. The divergence is partly
binding to the specific train forget samples (memorising where to push *them*) rather
than learning a general "forget = different from normal" representation. This is the same
transfer gap problem, expressed differently.

Still a useful model for diversity (it approaches the problem from a different angle than
label-denoising).

---

## 12. Concept erasure — probe-adversarial unlearning (exp09) — negative result

Having confirmed that forget samples are linearly separable in activation space, we tried
to exploit this directly: fit a linear probe to find the direction in activation space
that separates forget from retain, then **erase it** by training a loss that drives each
forget sample's projection onto that direction toward the retain mean. Iteratively refit
the probe each epoch (chasing residual directions, like INLP / concept erasure).

The mechanism was sound in theory: if the probe can no longer find forget samples, the
CNN is "honest" — it treats forget samples like normal samples, predicts their true
position, and gets a large error against the corrupted label.

In practice: **erasure works on the probe (AUC drops from 0.985 to ~0.82) but the best
Kaggle metric is at epoch 4, before erasure has meaningfully kicked in**. Deeper erasure
makes things worse because the probe direction is *entangled with retain*: you cannot
move forget activations along the separating direction without dragging retain activations
with them, degrading utility. This is the same entanglement story as SSD, now confirmed
in activation-direction space. Best est. test acc: 0.778 — worse than exp05.

**Summary of three independent entanglement confirmations:**

| approach | space | result |
|---|---|---|
| SSD (exp08) | parameter space (Fisher) | forget/retain importance entangled |
| Diverge (exp08) | activation trajectory | train-specific, does not transfer |
| Concept erasure (exp09) | activation direction | direction is shared with retain |

The representation-space family of approaches is a dead end for this particular model
and contamination type. The label-error recipes are the principled choice precisely
because the error signal *is* the corruption magnitude — a signal that transfers to
unseen test data by construction.

---

## 13. Improved detectors: top-50% threshold and error ensemble (exp10)

With the unlearning models established, we turned to improving the **detector** that
converts CNN errors to `is_forget` predictions. Two cheap interventions, no new training.

### Top-50% hard threshold

We know from the GMM analysis and the exp06 classifier that the test forget rate is very
close to 0.50 (higher than train's 0.375, confirmed independently by both methods). So
instead of fitting any threshold, we simply rank all 2728 test samples by their
exp07 error and label the top 1364 as forget.

Result: est. test acc **0.862** (vs 0.860 for exp07's LR). Marginal improvement, but
with perfect calibration: GMM rate is 0.485, the top-50% is 0.500 by construction.

### Ensemble of exp07 + exp08-diverge error vectors

The two best models approach the problem differently:
- exp07 (label-denoising): forget samples get large errors because their stored labels
  are corrupted and the model now predicts the true position.
- exp08-diverge (activation-push): forget samples get large errors because their
  activations are pushed off-trajectory, making the model inaccurate on them.

These are partially independent error signals. We blended the two test error vectors
with weights w₀₇ and w₀₈, then ran both the LR and GMM detectors on the blended
vector. We tried three weight combinations (50/50, 70/30, 30/70).

Best result: **70% exp07 + 30% exp08-diverge**, LR detector:

| metric | value |
|---|---|
| approx self-MIA | 0.931 |
| est. test acc (LR) | **0.897** |
| est. test acc (GMM) | 0.896 |
| GMM test forget rate | 0.502 |

A clear improvement: from 0.860 (exp07 alone) to **0.897** (+3.7 percentage points). This
submission (`exp10_ens_70_30_lr.csv`) scored **0.91564 on the public leaderboard** (30%
of test).

---

## 14. SCRUB unlearning — cross-fitted (exp11)

Motivated by the "Resolving Confusion" literature benchmark (SCRUB, NeurIPS 2023), which
is specifically designed for the mislabelled-data scenario, we implemented a regression
adaptation of the SCRUB algorithm on top of the exp07 cross-fitted checkpoints.

**SCRUB for regression:** alternating epochs rather than combining forget/retain gradients
in the same step (which is what caused retain degradation in NegGrad+/SSD):

- **max-step:** gradient *ascent* on the forget position loss (against the corrupted labels).
  The model is pushed to predict positions that score badly against the stored (wrong) label.
- **min-step:** gradient *descent* on the retain position loss (against clean labels).
  Retain quality is repaired before the next ascent.

Clipped gradient norms on the max-step (clip = 1.0) prevent the ascent from overshooting.
We ran 15 cycles (1 ascent epoch + 2 descent epochs per cycle), starting from each exp07
fold checkpoint, with lr_max = 2 × 10⁻⁵ and lr_min = 5 × 10⁻⁵.

| metric | value |
|---|---|
| per-fold self-MIA (training split) | 0.962 – 0.967 |
| OOF retain error | 0.090 m |
| OOF forget error | 0.464 m |
| OOF LR accuracy | 0.879 |
| GMM test forget rate | 0.497 |
| est. test acc (GMM) | 0.847 |

The alternating structure successfully avoided the retain collapse seen in NegGrad+ — no
fold hit the early-stop ceiling. However, the test-set improvement was modest: standalone
SCRUB estimated accuracy (0.847) is slightly below exp07 alone (0.860). The same
memorisation-bias pattern from exp08-diverge is at play: the max-step amplifies errors
on the specific training forget samples more than on unseen test forget samples.

The value of exp11 is as a **third orthogonal error signal** for the ensemble, not as a
standalone recipe.

---

## 15. Final 3-way ensemble and best submission (exp11 continued)

With three models producing complementary error vectors via different unlearning
mechanisms, we performed a grid search over blending weights and then ranked test samples
by blended error, labelling the top 1364 (50%) as forget. This **top-50% threshold**
requires no threshold transfer from train to test — calibration is exact by construction.

The three error signals:
- **exp07** (label-denoising, cross-fitted): forget error ≈ 0.785 m OOF; transfers well
  because the error equals the corruption magnitude.
- **exp08-diverge** (activation-push, single model): forget error ≈ 0.809 m train; adds
  a representation-space component orthogonal to the label-denoising signal.
- **exp11 SCRUB** (alternating ascent/descent, cross-fitted): OOF forget error ≈ 0.464 m;
  a third mechanism (gradient-space alternating) orthogonal to both above.

Grid search over weights (step 0.05, sum = 1.0), evaluated by agreement with exp06
pseudo-labels:

| blend | offline pseudo-agreement |
|---|---|
| exp07 alone (top-50) | 0.860 |
| exp07 70% + exp08-div 30% (LR) — submitted as exp10 | ~0.897 (actual LB 0.91564) |
| exp07 50% + exp11 50% (top-50) | 0.866 |
| exp07 30% + exp11 30% + exp08-div 40% (top-50) | **0.900** |
| exp07 35% + exp11 35% + exp08-div 30% (top-50) | **0.900** |

The 3-way ensemble at 30/30/40 reaches **0.900 offline** and scored **0.92787 on the
public leaderboard** — an improvement of +1.2 pp over the exp10 submission.

The exp08-diverge weight of 0.40 (higher than in exp10's 0.30) was enabled by the SCRUB
component absorbing some of the retain-quality role: the two relabeling components (exp07,
exp11) anchor the retain distribution while the larger diverge weight amplifies the
forget gap further without over-predicting forget.

---

## 16. Summary table — all experiments

| exp | recipe | est. test acc | retain err | forget err | public LB | outcome |
|---|---|---|---|---|---|---|
| exp00 | pretrained baseline | — | 0.101 m | 0.125 m | — | reference |
| exp01 | finetune retain-only | 0.747 | 0.025 m | 0.061 m | — | threshold does not transfer |
| probe | direct CSI+pos classifier | 0.886 (CV) | — | — | — | retired (bypasses CNN) |
| exp02 | NegGrad+ α=0.5 | 0.828 | 0.130 m | 0.555 m | — | decent, noisy |
| exp05 | kNN relabel + finetune | 0.854 | 0.113 m | 0.861 m | — | key insight; threshold issue |
| exp07 | 5-fold cross-fit exp05 | 0.860 | 0.149 m | 0.785 m | — | threshold fixed; best single model |
| exp08-ssd | Fisher dampening grid | < 0.660 | ≥ 0.25 m | — | — | negative; entangled parameters |
| exp08-diverge | activation trajectory push | 0.852 | 0.071 m | 0.809 m | — | transfer gap; useful for ensemble |
| exp09 | probe concept erasure | 0.778 | 0.124 m | 0.261 m | — | negative; direction entangled |
| exp10 | exp07 + exp08 ensemble (70/30), LR | 0.897 | — | — | **0.91564** | first submission |
| **exp11** | **exp07 + exp11 SCRUB + exp08 (30/30/40), top-50** | **0.900** | — | — | **0.92787** | **best; final submission** |

---

## 17. Key lessons

**1. Understand your contamination before choosing an algorithm.**
The correct unlearning strategy is entirely determined by *what* was contaminated.
Here it was position labels — not the CSI itself. A generic unlearning algorithm
(NegGrad, SSD) that treats contamination as an unknown black box leaves a lot on the table.

**2. The error signal must transfer.**
A model that memorises forget samples will have small errors on them during training, but
that does not mean a threshold trained on those small errors will work on test data.
Cross-fitting (holding each sample out) is the fix that makes train and test errors
comparable.

**3. Entanglement prevents representation-space unlearning.**
Three independent experiments (parameter-space Fisher, activation-trajectory divergence,
activation-direction erasure) all hit the same wall: forget and retain representations
share the same substrate. You cannot surgically remove one without damaging the other.
This is not a failure of implementation — it reflects a fundamental property of how this
CNN processes these two populations.

**4. Simple detectors can beat complex unlearning.**
The improvement from exp07 → exp10 (+3.7 pp) came entirely from combining error vectors,
not from new training. The unlearning was done; the detector was the bottleneck.

**5. Diversity beats depth in ensembles.**
Each additional unlearning mechanism (label-denoising, activation-push, alternating
ascent/descent) operates in a different space. Their errors are not perfectly correlated,
so blending them narrows the detector's decision boundary. Adding a third mechanism
(SCRUB, +1.2 pp) even when its standalone performance is below the prior best shows that
ensemble diversity matters more than any single model's quality.
