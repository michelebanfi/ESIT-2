# ESIT-D2I 2026 — Task 2: Unlearning & Data Reliability

## Overview
Competition: ESIT-D2I 2026 Charting the Path to Bifröst — Task 2 (Kaggle)
Task: **Machine unlearning** for CSI-based indoor localization. A pretrained CNN was trained on
data that includes a contaminated subset (`is_forget=1`). Goal: remove those samples'
influence **without retraining from scratch**, while preserving accuracy on the clean retain set
and the unseen test set.

Dataset is based on DICHASUS (Arena2036 campus). Raw CSI is complex64 `(N, 4, 2, 4, 64)`,
preprocessed to `(N, 2, 32, 64)` float by flattening antenna dims and splitting real/imag.
CNN predicts **2D (x, y)** positions (z is nearly constant ~0.94 m). **Submission is `is_forget`
labels**, not positions — the pipeline is: CNN errors → LogisticRegression → is_forget CSV.

## Shapes
| File | Shape | Notes |
|------|-------|-------|
| public/task2_train_csi.npy | (10918, 4, 2, 4, 64) complex64 | → (10918, 2, 32, 64) after preprocessing |
| public/task2_train_positions.npy | (10918, 3) float64 | only [:, :2] used |
| public/task2_train_metadata.csv | 10918 rows | is_forget: 6823 retain + 4095 forget |
| public/task2_test_csi.npy | (2728, 4, 2, 4, 64) complex64 | |
| task2_test_positions.npy | (2728, 3) float64 | available locally |
| baseline_cnn_task2.pth | — | DichasusPositionPredictor checkpoint |

## Hardware
MacBook Pro M1 Pro, 32 GB RAM. MPS is available (`torch.backends.mps.is_available() == True`).

## Environment
```bash
source venv/bin/activate          # Python 3.14, PyTorch 2.12
```

## Repository layout
```
src/
  dataset.py      format_csi_for_cnn() + make_tensor_dataset()
  model.py        DichasusPositionPredictor (6-layer CNN) + load_model()
  train.py        train_epoch / eval_loss helpers
  metrics.py      get_predictions / prediction_errors / mia_accuracy / localization_stats
scripts/
  download_data.py     pull dataset via kagglehub, symlink to ./data/
  inspect_data.py      sanity-check shapes, NaNs, is_forget distribution
  eval_pretrained.py   exp00: eval baseline checkpoint via MIA pipeline
  finetune_retain.py   exp01: finetune on retain set, evaluate via MIA
  make_submission.py   generate Kaggle submission CSV from any checkpoint
experiments/
  exp00_baseline/          metrics.json (pretrained checkpoint, no unlearning)
  exp01_finetune_retain/   model_best.pth + metrics.json (naive unlearning baseline)
data/                      symlink to kagglehub cache (gitignored)
  public/                  train/test CSI + metadata
  baseline_cnn_task2.pth
```

## Workflow
```bash
source venv/bin/activate
python scripts/download_data.py    # once
python scripts/inspect_data.py     # sanity check
python scripts/eval_pretrained.py  # exp00 — pretrained baseline
python scripts/finetune_retain.py  # exp01 — naive unlearning
python scripts/make_submission.py --ckpt experiments/exp01_finetune_retain/model_best.pth
```

## Pipeline
The Kaggle metric is **LR accuracy** on `is_forget` classification:
1. CNN predicts (x,y) on all train samples.
2. Compute Euclidean prediction errors per sample.
3. Train `LogisticRegression(error → is_forget)` on all train samples.
4. Apply to test errors → predict `is_forget` → submit CSV.
Good unlearning = model makes larger errors on forget samples → LR can distinguish them better.

## Metrics legend
- **MIA acc**: LogisticRegression accuracy trained on prediction errors (higher = better unlearning)
- **retain err**: mean 2D Euclidean error on is_forget==0 (must stay low = utility preserved)
- **forget err**: mean 2D Euclidean error on is_forget==1 (should rise = unlearning worked)

## Experiment changelog

<!-- EXPERIMENTS -->

| id | date | recipe | MIA acc | retain err (m) | forget err (m) | notes |
|----|------|--------|---------|---------------|---------------|-------|
| exp00 | pretrained | baseline checkpoint | 0.6556 | 0.1010 | 0.1252 | reference — LR trained on train errors |
| exp01 | 2026-06-09 | finetune retain-only lr=0.0001 ep=50 | 0.7889 | 0.0245 | 0.0608 | first unlearning baseline |
| probe | 2026-06-09 | no CNN — direct classifiers on CSI/pos | 0.886 (CV) | — | — | contamination = corrupted position labels; kNN-consistency AUC 0.936 |
| exp02 | 2026-06-09 | NegGrad+ alpha=0.5 clamp=0.25 ep=15 | 0.8672 | 0.1301 | 0.5553 | est. test acc 0.828 (pseudo-label agreement) |
| exp05 | 2026-06-09 | relabel forget w/ kNN-corrected pos, finetune ep=30 | 0.9265 | 0.1135 | 0.8607 | best unlearning recipe; est. test acc 0.854; GMM test split = 0.50 forget |
| exp06 | 2026-06-09 | direct HGB: CSI stats + pos + kNN-consistency | 0.9626 (CV) | — | — | **RETIRED 2026-06-10** — bypasses the CNN; rules-gray under winner verification (§2.8). Kept for offline diagnostics only |
| exp07 | 2026-06-10 | 5-fold cross-fitted exp05 ensemble, LR on OOF errors | 0.8785 (OOF) | 0.1491 | 0.7852 | **best rules-safe recipe**; est. test acc 0.860; LR & GMM detectors agree on 99.8% of test; final submission candidates |
| exp08-probe | 2026-06-10 | activation probe + Fisher importance on baseline | — | — | — | forget/retain linearly separable in activations (AUC 0.948→0.986 by block 5); forget-dominant Fisher mass concentrated in blocks 2–4 |
| exp08-ssd | 2026-06-10 | Selective Synaptic Dampening grid (alpha×lambda), BN recalibration | 0.66 max | ≥0.25 | — | **negative result**: forget/retain circuits fully entangled — dampening destroys retain in lockstep (also on top of exp05). Parameter-space detachment dead |
| exp08-diverge | 2026-06-10 | trajectory-divergence finetune ep=24: anchor retain ReLU acts to frozen teacher, hinged-cosine push forget BN acts off-trajectory, no forget pos loss | 0.9533 | 0.0707 | 0.8095 | best self-MIA + best retain utility of any recipe; est. test acc 0.8515 but GMM test rate 0.432 (transfer gap: divergence partly memorises train forget samples). knn-target variant worse (0.8347). Diversity pick for 2nd submission slot |
| exp10 | 2026-06-10 | 2-way ensemble exp07+exp08-diverge (70/30), LR detector | 0.9310 (approx) | — | — | est. test acc 0.897; **public LB 0.91564** — first submission |
| exp11 | 2026-06-16 | SCRUB cross-fitted (5-fold): alternating gradient ascent on forget (lr=2e-5) + descent on retain (lr=5e-5), 15 cycles, starting from exp07 fold ckpts | 0.8788 (OOF) | 0.0897 | 0.4636 | OOF LR≈exp07; value is ensemble diversity. 3-way ensemble exp07 30%+exp11 30%+exp08-div 40%, top-50% threshold: offline 0.900; **public LB 0.92787** — final submission |

### Key findings (2026-06-09)
- **Contamination is position/label corruption**: forget samples' labelled positions are ~4×
  farther from their CSI-neighbours' positions (1.21 m vs 0.32 m). CSI itself also differs
  (RF on CSI summary stats alone: 88.6% CV).
- **Test forget rate ≈ 0.50**, higher than train's 0.375 (exp06 test probabilities are cleanly
  bimodal: 45% > 0.9, 42% < 0.1). Use the 0.5 threshold, not a prior-matched one.
- **Robust eval** (`scripts/eval_robust.py`): GMM split of test errors + agreement with exp06
  pseudo-labels (`experiments/exp06_direct_classifier/test_proba.npy`) estimates Kaggle accuracy
  offline. exp00 → 0.726, exp01 → 0.747.

### Key findings (2026-06-10, exp08)
- **The CNN internally "knows" forget samples**: linear probes on channel-mean activations
  reach AUC 0.986 (block 5). Unlearning signal is representational, not just label-level.
- **Parameter-space detachment fails**: forget/retain Fisher importance is entangled —
  SSD-style dampening degrades retain & forget errors in lockstep at every grid setting.
- **Activation-space divergence works but transfers imperfectly**: pushing forget
  activations off-trajectory (vs frozen teacher) with retain anchoring gives the best
  self-MIA (0.9533) and retain error (0.071 m) of any recipe, but est. test acc 0.8515 —
  the divergence partly binds to the specific train forget samples (GMM test rate 0.432
  vs expected 0.50). The exp05/exp07 relabel recipes remain the better submission path
  because their error signal *is* the corruption magnitude, which transfers exactly.

### Rules check (2026-06-10) — exp05 verdict
- **exp05's kNN relabeling is rules-safe.** §2.6.c prohibits only (i) use of the raw
  DICHASUS dataset / its ground-truth positions — i.e. external data not in the public
  release — and (ii) identifying the hidden **test** `is_forget` labels via reverse
  engineering or leaderboard probing. exp05 derives corrected positions purely from the
  public **train** data (PCA of |CSI| + retain-neighbour positions) and never touches test.
- `data/task2_test_positions.npy` is part of the official Kaggle download; the organizers'
  `competition_notebook.ipynb` loads it to compute test errors. The errors→GMM/LR test
  pipeline is the sanctioned one, not a gray area.
- Writeup precaution (§2.8 verification): describe exp05 as "label denoising via internal
  kNN consistency on competition train data", never as "recovering ground-truth positions".

### Submission policy (decided 2026-06-10, after rules review — see rules.txt)
- **All submissions must flow through the CNN**: (unlearned) model errors → detector → is_forget.
  No CSI-direct or position-direct classifiers in the submission path (winner verification §2.8
  exposes the method; sponsor discretion §2.9.e makes gray-area tactics risky).
- exp06 remains useful **offline only** (pseudo-label agreement for model selection — also keeps
  us away from leaderboard probing, prohibited by §2.6.c).
- Two final submissions allowed (§2.2.b): pick the two best CNN-error-based CSVs.
- Public LB = ~30% of test (~818 samples) — noisy; trust offline diagnostics over LB deltas.


## Future experiment ideas
- `exp02_gradient_ascent`: add a gradient-ascent loss on the forget set while retaining on the retain set (NegGrad)
- `exp03_scrub`: SCRUB (Selective Synaptic Dampening / maximise forget loss + minimise retain loss alternating)
- `exp04_fisher`: Fisher-based parameter importance weighting during retain fine-tune
- `exp05_label_noise_finetune`: re-label forget samples with random positions and fine-tune briefly
