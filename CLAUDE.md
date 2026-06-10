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

### Key findings (2026-06-09)
- **Contamination is position/label corruption**: forget samples' labelled positions are ~4×
  farther from their CSI-neighbours' positions (1.21 m vs 0.32 m). CSI itself also differs
  (RF on CSI summary stats alone: 88.6% CV).
- **Test forget rate ≈ 0.50**, higher than train's 0.375 (exp06 test probabilities are cleanly
  bimodal: 45% > 0.9, 42% < 0.1). Use the 0.5 threshold, not a prior-matched one.
- **Robust eval** (`scripts/eval_robust.py`): GMM split of test errors + agreement with exp06
  pseudo-labels (`experiments/exp06_direct_classifier/test_proba.npy`) estimates Kaggle accuracy
  offline. exp00 → 0.726, exp01 → 0.747.

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
