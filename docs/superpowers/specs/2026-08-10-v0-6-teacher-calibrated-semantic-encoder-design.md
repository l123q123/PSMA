# PASM v0.6 Teacher-Calibrated Semantic Encoder Design

## Goal

Retrain the frozen-DINO dual-image semantic encoder so that its 16x16 patch values reproduce the full geometric visibility-support heatmap, rather than concentrating a globally normalized probability distribution into only the strongest patches.

## Scope

- Keep the DINO backbone, DaD detector, LoMa descriptor, PairSemanticEncoder topology, and v0.5 fusion/inference path unchanged.
- Train a new semantic checkpoint from a fresh deterministic initialization.
- Replace only the v0.4 geometry value target representation and value-only loss.
- Do not add uniform-coverage, entropy-maximization, patch-budget, or joint-fusion objectives.

## Teacher target

For each source image and each patch, sample the existing regular stride-2 grid.  Let `M_i` be the number of source samples that project into the other image and pass the existing depth-visibility test, and let `N_i` be the number of sampled source points assigned to patch `i`.

`raw_support_i = M_i / N_i` is the new continuous target in `[0, 1]`.  It is stored alongside the legacy `normalized_target_i = M_i / sum_j M_j`; the latter is retained for diagnostics and the weak distribution-shape term only.

Every patch has its own target.  No thresholding converts the teacher to a binary mask and no target is globally normalized before the main loss.

## Model output

`PairSemanticEncoder` keeps its single per-patch value logit and sigmoid output.  Under the new mode, its sigmoid output is interpreted as predicted raw geometry support.  The shared semantic tokens still feed downstream v0.5 fusion unchanged.

## Loss

For predictions `p = sigmoid(z)`, raw teacher support `t`, and normalized teacher distribution `q`:

```text
L_local = BCEWithLogits(z, t)
L_reg   = SmoothL1(p, t)
L_shape = KL(q || normalize(p))
L_total = L_local + 0.25 * L_reg + 0.05 * L_shape
```

The local loss is the primary signal and treats all 256 patches independently.  The regression term calibrates the continuous heatmap.  The weak shape term preserves relative high-support ordering without reintroducing dominant global competition.

## Cache and checkpoint identity

Add target kind `geometry_calibrated`.  Its cache records `raw_target_a/b`, `distribution_target_a/b`, the existing pair and grid metadata, and a target-schema identifier.  New checkpoints record `loss_mode=calibrated_shape` and `target_kind=geometry_calibrated`; resume must reject incompatible caches/checkpoints.

## Training and evaluation

Use the same 16x16, one-cross-attention, 96-positive-pair setup, frozen backbones, optimizer, seed, and 60 epochs as the static-16 baseline.  Save epoch 20, 40, and 60 checkpoints.  Training logs expose local, regression, shape, and total loss.

Held-out diagnostics compare both raw support calibration and normalized heatmap agreement: raw MAE/Brier, effective patch count, visible-patch recall and precision, Spearman, and top-20 IoU.  The existing nine held-out pairs remain final evaluation data rather than being used for checkpoint selection.

## Non-goals

This change does not retrain DaD, change its point scores, alter v0.5 fusion-head architecture, make a global spatial-uniformity prior, or jointly fine-tune the semantic encoder and fusion head.
