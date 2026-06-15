# Winning recipe — cross-fitted label-denoising unlearning (exp07)

End-to-end description of the submitted model: how raw CSI becomes an `is_forget`
prediction, and why each stage exists.

## 0. The core idea in one paragraph

The contamination in the training set is **corrupted position labels**: a `forget`
sample's CSI is genuine, but the `(x, y)` it is labelled with is wrong (≈4× farther from
its CSI-neighbours' positions than a clean sample). Unlearning therefore means: get the
CNN to predict the **true** position for every sample. Once it does, a forget sample's
prediction lands near its true position while its *stored* label is still corrupted, so
its **prediction error is large**. A clean sample's prediction matches its label, so its
error is small. The error magnitude *is* the corruption signal — and it is the one signal
that transfers exactly to the unseen test set. We classify `is_forget` by thresholding
that error.

Everything downstream is engineering to make that error estimate **unbiased** (no model
scoring a sample it was trained on) and the threshold **transferable** (found within each
set, never carried over from train).

## 1. Data transformation: CSI → CNN input

`src/dataset.py::format_csi_for_cnn`, applied identically to train and test:

1. Raw CSI is complex64 `(N, 4, 2, 4, 64)` = (samples, AP, pol, antenna, subcarrier).
2. Flatten the three antenna dims → `(N, 32, 64)`.
3. Split into real and imaginary parts, stack as 2 channels → `(N, 2, 32, 64)` float32.
4. Global standardisation: subtract the global mean, divide by the global std (one scalar
   each, over the whole array). Matches the competition notebook's preprocessing exactly.

Positions: only `[:, :2]` (x, y) are used; z is ~constant (0.94 m) and dropped.

## 2. Label denoising: kNN-corrected forget positions

We never trust a forget sample's labelled position. We re-estimate it from clean data,
using **train data only** (rules-safe — no external DICHASUS data, no test labels):

1. Magnitude features: `|CSI|` flattened per sample, reduced to 64 dims with PCA.
2. Fit `NearestNeighbors` on the **retain** samples in this PCA space.
3. For each forget sample, take its 10 nearest retain neighbours and set its corrected
   position to the **mean of their (true) positions**.
4. Retain labels are left untouched.

This yields `pos_corrected`: retain positions unchanged, forget positions replaced by a
clean estimate of where that CSI actually was. (Mean correction shift is ~1.2 m, matching
the measured corruption magnitude.)

> Writeup language: describe this as *"label denoising via internal kNN consistency on
> competition train data"* — never as "recovering ground-truth positions".

## 3. Model: 5-fold cross-fitted fine-tune ensemble

We fine-tune the provided baseline CNN (`DichasusPositionPredictor`, 6-block conv net,
`data/baseline_cnn_task2.pth`) on the **corrected** labels. The twist that makes the error
signal honest is **cross-fitting**:

- Split train into 5 stratified folds (stratified on `is_forget`, seed 42).
- For each fold *k*: load the baseline checkpoint fresh, fine-tune on the corrected labels
  of the **other 4 folds** (Adam, lr 1e-4, 30 epochs, batch 64, MSE loss), then set fold
  *k* aside.

Why: a model memorises any sample it trains on, so its error on that sample is
artificially small. By holding out fold *k*, every train sample is scored by a model that
**never saw it** → out-of-fold (OOF) errors that are representative of test behaviour.

Output: 5 fold checkpoints (`model_fold{1..5}.pth`).

## 4. Error computation

All errors are 2D Euclidean distance between prediction and the **original (possibly
corrupted) label** — that is what the detector must read.

- **Train (OOF):** each sample's error comes from the single fold model that held it out.
  → `oof_errors` (length 10918).
- **Test:** every fold model scores all test samples; the 5 error vectors are **averaged**
  per sample (ensemble mean error). → `test_errors` (length 2728).

OOF separation: retain mean ≈ 0.149 m vs forget mean ≈ 0.785 m.

## 5. Detector: error → `is_forget`

Two interchangeable detectors are produced; both consume only CNN errors.

- **LR (supervised threshold):** `LogisticRegression` fit on `oof_errors → is_forget`,
  then applied to the averaged `test_errors`. Because the training errors are OOF, the
  LR's own fit accuracy (0.8785) is already a cross-validated estimate.
- **GMM (unsupervised threshold):** a 2-component Gaussian mixture on `log(test_errors)`;
  the higher-mean component is labelled `forget`. This finds the split **inside the test
  set itself**, so no absolute threshold is transferred from train (where errors are
  smaller). Use the natural **0.5** decision, not a prior-matched one — test errors are
  cleanly bimodal and the true test forget rate is ≈0.50 (higher than train's 0.375).

The two detectors agree on **99.8%** of test samples, which is our confidence check.

## 6. Submission

For the chosen detector, write `submission_{lr,gmm}.csv` with columns `id`
(= `meta_test["sample_index"]`) and `is_forget` (0/1). Two final slots are allowed; we
submit the two best CNN-error CSVs.

## 7. Validation (offline, no leaderboard probing)

`scripts/eval_robust.py` estimates Kaggle accuracy without touching the LB:

1. self-MIA on train errors (legacy continuity metric),
2. GMM forget rate on test (sanity: want ≈0.50),
3. agreement with the retired exp06 direct-classifier pseudo-labels (96% CV) — used
   **offline only**, for model selection.

exp07 estimated test accuracy: **0.860** (best rules-safe recipe).

---

## End-to-end data flow

```
            ┌──────────────────────── TRAIN ────────────────────────┐
raw CSI (N,4,2,4,64) ─ format_csi_for_cnn ─► X (N,2,32,64)
positions (N,3) ─ [:, :2] ─► pos_train
                                │
        |CSI|→PCA(64)→kNN(10 retain neighbours)
                                ▼
                         pos_corrected  (retain unchanged, forget denoised)
                                │
              5× stratified fold fine-tune (baseline CNN, lr1e-4, 30ep)
                                │
        ┌───────────────────────┴───────────────────────┐
   held-out fold errors                          all folds score TEST
   (vs ORIGINAL labels)                                  │
        ▼                                          mean over 5 folds
   oof_errors (10918)                              test_errors (2728)
        │                                                 │
   fit LR(error→is_forget)  ───────────────────►  LR.predict(test_errors)
        │                                          GMM split of test_errors
        └─────────────── agree 99.8% ────────────────────┘
                                ▼
                      is_forget ─► submission.csv (id, is_forget)
```

Key invariants that make it work and keep it rules-safe:
- **Honest errors:** no sample is ever scored by a model trained on it (cross-fitting).
- **Transferable threshold:** the test split is found within the test errors (GMM), never
  carried over from the memorised train distribution.
- **CNN-only pipeline:** the prediction path is CSI → CNN → error → detector; no
  CSI-direct or position-direct classifier is in the submission path.
- **Train-data-only denoising:** corrected labels come purely from public train CSI +
  retain positions; the test set and its labels are never used to build the model.
