# v0.6 Calibrated Shape Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the 16x16 dual-image semantic encoder against independent continuous geometry-support targets while retaining a weak normalized-shape term.

**Architecture:** Geometry target generation will expose per-patch visible fractions before legacy global normalization. A new calibrated cache kind stores both raw support and a normalized diagnostic distribution. The existing sigmoid value head is trained with independent local BCE, SmoothL1 calibration, and a weak KL shape loss; the encoder topology and v0.5 consumers stay unchanged.

**Tech Stack:** Python 3, NumPy, PyTorch, unittest, existing PASM v0.4 cache/training/checkpoint infrastructure.

## Global Constraints

- Use patch grid 16, stride 2, and the existing geometry depth-visibility test.
- Freeze DINO, DaD, and LoMa exactly as current `train_v04.py` does.
- Preserve current `geometry` and `quality` cache/checkpoint compatibility.
- New mode is named `geometry_calibrated` with loss mode `calibrated_shape`.
- Do not modify v0.5 fusion or inference behavior.

---

### Task 1: Store independent raw geometry support beside the diagnostic distribution

**Files:**

- Modify: `pasm_v0/geometry.py`
- Modify: `pasm_v0/v04_targets.py`
- Modify: `tests/test_pasm_v0_v04_targets.py`

**Interfaces:**

- Produces `projected_patch_support_fractions(...) -> tuple[np.ndarray, np.ndarray]`, each value finite and in `[0, 1]`.
- Produces `load_calibrated_patch_targets(...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` returning raw A/B then normalized A/B targets.

- [ ] **Step 1: Write the failing cache validation test**

```python
raw = np.zeros(config.patch_count, dtype=np.float32)
raw[[0, 1]] = [0.25, 0.75]
distribution = raw / raw.sum()
np.savez_compressed(path, raw_target_a=raw, raw_target_b=raw,
                    distribution_target_a=distribution, distribution_target_b=distribution,
                    target_kind=np.asarray("geometry_calibrated"), target_schema=np.asarray(2), ...)
loaded = load_calibrated_patch_targets(root, "demo", config)
np.testing.assert_allclose(loaded[0], raw)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_v04_targets -v`

Expected: import failure because `load_calibrated_patch_targets` does not exist.

- [ ] **Step 3: Implement raw support and cache helpers**

```python
def projected_patch_support_fractions(...):
    counts = _patch_support_counts(points[visible], layout)
    totals = _patch_support_counts(points, layout)
    return counts / np.maximum(totals, 1.0)

def load_calibrated_patch_targets(...):
    # Require raw_target_a/b, distribution_target_a/b, target_schema=2,
    # metadata identity, raw values in [0,1], and distributions summing to one.
```

`cache_targets(..., target_kind="geometry_calibrated")` saves both forms. Existing legacy helpers retain their existing names and behavior.

- [ ] **Step 4: Run target tests to verify they pass**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_v04_targets -v`

Expected: all target tests pass.

### Task 2: Add calibrated-shape value loss and checkpoint identity

**Files:**

- Modify: `pasm_v0/train_v04.py`
- Modify: `tests/test_pasm_v0_train_v04.py`

**Interfaces:**

- Produces `v04_calibrated_shape_loss(logits, raw_target, distribution_target) -> tuple[Tensor, Tensor, Tensor, Tensor]`.
- `train_model(..., target_kind="geometry_calibrated", loss_mode="calibrated_shape")` trains only the calibration objective and stores all three scalar losses in history.

- [ ] **Step 1: Write failing loss and checkpoint tests**

```python
logits = torch.tensor([0.0, 0.0], requires_grad=True)
raw = np.array([0.2, 0.8], dtype=np.float32)
distribution = np.array([0.2, 0.8], dtype=np.float32)
local, regression, shape, total = v04_calibrated_shape_loss(logits, raw, distribution)
total.backward()
assert logits.grad is not None
assert float(total) > float(local)
```

Also assert a calibrated checkpoint records `target_kind="geometry_calibrated"` and `loss_mode="calibrated_shape"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_train_v04 -v`

Expected: import failure because `v04_calibrated_shape_loss` does not exist.

- [ ] **Step 3: Implement the loss and training branch**

```python
predictions = torch.sigmoid(logits)
local = F.binary_cross_entropy_with_logits(logits, raw)
regression = F.smooth_l1_loss(predictions, raw)
shape = F.kl_div(F.log_softmax(logits, dim=0), distribution, reduction="batchmean")
total = local + 0.25 * regression + 0.05 * shape
```

In calibrated mode, bypass DaD point selection and `projected_match_targets`; load calibrated cache targets for both images, sum their calibrated losses, and log `local_loss`, `regression_loss`, and `shape_loss`. Extend CLI choices and checkpoint compatibility checks.

- [ ] **Step 4: Run training tests to verify they pass**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_train_v04 -v`

Expected: all training tests pass.

### Task 3: Expose calibration diagnostics and verify the executable path

**Files:**

- Modify: `pasm_v0/diagnose_v04_distribution.py`
- Modify: `tests/test_pasm_v0_train_v04.py`

**Interfaces:**

- Produces raw-support MAE, Brier score, effective-patch count, and legacy normalized heatmap metrics for calibrated caches/checkpoints.

- [ ] **Step 1: Write a failing metric test**

```python
metrics = calibrated_support_metrics(
    np.array([0.0, 1.0], dtype=np.float32),
    np.array([0.25, 0.75], dtype=np.float32),
)
assert metrics["raw_mae"] == 0.25
assert metrics["brier"] == 0.0625
```

- [ ] **Step 2: Run it to verify it fails**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_train_v04 -v`

Expected: import failure because `calibrated_support_metrics` does not exist.

- [ ] **Step 3: Implement metrics and calibrated target loading in the diagnostic CLI**

```python
def calibrated_support_metrics(prediction, target):
    difference = prediction - target
    return {"raw_mae": float(np.abs(difference).mean()),
            "brier": float(np.square(difference).mean())}
```

The diagnostic script additionally reports effective-patch counts plus its existing Spearman and top-k normalized-map metrics.

- [ ] **Step 4: Run focused and regression tests**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_v04_targets tests.test_pasm_v0_train_v04 tests.test_pasm_v0_evaluate_v04 -v`

Expected: all tests pass.

## Training command after implementation

```powershell
& $py -m pasm_v0.cache_v04_targets `
  --manifest $manifest `
  --output 'outputs\v0.6_ablation\geometry_calibrated_16_targets' `
  --target-kind geometry_calibrated `
  --patch-grid 16 `
  --device cuda

& $py -m pasm_v0.train_v04 `
  --manifest $manifest `
  --dino-weights $dino `
  --loma-weights $loma `
  --dad-weights $dad `
  --target-cache 'outputs\v0.6_ablation\geometry_calibrated_16_targets' `
  --target-kind geometry_calibrated `
  --output 'outputs\v0.6_ablation\geometry_calibrated_shape_16_60.pt' `
  --epochs 60 `
  --loss-mode calibrated_shape `
  --patch-grid 16 `
  --device cuda
```

