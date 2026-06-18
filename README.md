# ESIT-D2I 2026 — Task 2: Machine Unlearning

**Competition:** ESIT-D2I 2026 Charting the Path to Bifröst — Task 2 (Kaggle)  
**Result:** 3rd place (public LB 0.92787 with an earlier non-compliant ensemble)

---

## Problem

A pretrained CNN (`DichasusPositionPredictor`) was trained on data that includes a
contaminated subset (`is_forget=1`). Goal: **remove those samples' influence** from
the model without retraining from scratch, while preserving accuracy on the clean
retain set and an unseen test set.

The mandatory evaluation (2026-06-17 organizer clarification): one fine-tuned model →
its own full-train errors (vs original labels) → `LogisticRegression()` → test errors
→ `predict()` at 0.5 → `id,is_forget` CSV.

---

## Contamination diagnosis

Forget samples have **corrupted position labels**: their CSI disagrees with their stored
`(x, y)`. Mean kNN-consistency shift: 1.21 m (forget) vs 0.32 m (retain). The CNN
internally "knows" forget samples — linear probe AUC at block 5 post-ReLU: **0.986**.

---

## Compliant submissions

| Checkpoint | Method | Offline LR~acc | LR forget rate |
|---|---|---|---|
| `experiments/relabel/model_best.pth` | kNN-relabel finetune | **0.8475** | 0.538 |
| `experiments/diverge/model_official_knn_b3.pth` | Activation-trajectory divergence | **0.8442** | 0.519 |

Both CSVs are in `submissions/`.

---

## Reproducing the results

```bash
# 0. One-time setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/data/download.py          # fetch dataset + symlink ./data/

# 1. Sanity check
python scripts/data/inspect.py

# 2. Diagnose contamination (no CNN)
python scripts/probing/contamination.py

# 3. Build pseudo-label oracle (offline only — never in submission)
python scripts/probing/pseudo_labels.py

# 4a. Train relabel model (exp05, ~5 min on M1)
python scripts/unlearning/relabel.py

# 4b. Train diverge model (exp08, ~20 min on M1)
python scripts/unlearning/diverge.py --forget-target knn --beta-anchor 3 --epochs 30

# 5. Evaluate (official metric)
python scripts/submission/eval_official.py \
  --ckpt experiments/relabel/model_best.pth \
  --ckpt experiments/diverge/model_official_knn_b3.pth

# 6. Generate compliant CSVs
python scripts/submission/make_submission.py \
  --ckpt experiments/diverge/model_official_knn_b3.pth \
  --out submissions/my_submission.csv

# 7. Generate presentation figures (forward passes on M1, ~2 min)
python scripts/probing/make_figures.py
```

---

## Repository layout

```
src/
  dataset.py      format_csi_for_cnn() + knn_corrected_positions() + make_tensor_dataset()
  model.py        DichasusPositionPredictor (6-layer CNN) + load_model()
  train.py        train_epoch / eval_loss helpers
  metrics.py      get_predictions / prediction_errors / mia_accuracy / localization_stats
  activations.py  ActivationTap (forward-hook tap) + fisher_importance()

scripts/
  data/           download.py  inspect.py
  probing/        contamination.py  pseudo_labels.py  activations.py  make_figures.py
  unlearning/     relabel.py  diverge.py
  submission/     make_submission.py  eval_official.py

experiments/
  relabel/        model_best.pth, config.json
  diverge/        model_official_knn_b3.pth, train_log_knn_b*.json
  probe/          cohens_d_layer*.npy, layer_separability.json, fisher_layer_summary.json,
                  fisher_{forget,retain}_baseline_cnn_task2.pth, ssd_grid_results_baseline.json
  pseudo_labels/  test_proba.npy  (offline diagnostic oracle — never in submission)
  probe_contamination/  results.json

figures/          PDF figures for the Beamer presentation
presentation/     main.tex + compiled main.pdf
submissions/      submission_relabel.csv  submission_diverge_knn.csv
data/             symlink → kagglehub cache (gitignored)
```

---

## Key findings

- **Contamination = corrupted positions**: forget CSI is genuine; the stored `(x,y)` is
  wrong. The corruption magnitude (~1.2 m shift) is the unlearning signal.
- **CNN already separates forget/retain**: linear probe AUC rises 0.948 → 0.986 with depth.
- **Parameter-space detachment (SSD) fails**: Fisher importance is entangled; dampening
  degrades retain and forget in lockstep at every grid point.
- **Activation-trajectory divergence**: push forget pre-ReLU BN activations off the baseline
  trajectory (hinged cosine), anchor retain post-ReLU activations to a frozen teacher.
  + kNN-corrected forget positions supply the transferable error signal.
- **Prior LB score (0.92787)** used a non-compliant 3-way ensemble + top-50% threshold.
  The compliant single-model results are: relabel 0.8475, diverge 0.8442 (offline LR~acc).

See `CLAUDE.md` for the full experiment history and changelog.
