# PASM v0.5 Dense Point Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hard Patch-budget point selection with a learned residual dense score map that combines frozen DaD pixel-level features and the existing trained dual-image semantic encoder, then selects 1024 points per image with the existing DaD/LoMa inference flow.

**Architecture:** Expose the frozen DaD decoder context and dense detector response, project and upsample the semantic encoder's cross-attended token hidden states, and train a shared residual Fusion Head whose output is added to the DaD base logit. Train only the fusion/projection layers using geometry-derived candidate-level soft labels from a large frozen-DaD candidate pool; keep DaD, DINO, semantic encoder, DeDoDe, and LoMa frozen in v0.5.

**Tech Stack:** Python, PyTorch, NumPy, existing `pasm_v0` geometry/data/evaluation utilities, local LoMa/DaD weights, CUDA when available.

## Global Constraints

- Final inference selects exactly 1024 points per image.
- DaD is never trained or updated.
- Inference must not read depth, camera, pose, or other geometry files.
- LoMa matching, dual-softmax mutual filtering, and existing 3px/6px evaluation remain unchanged.
- No Patch activation, Patch cap, Patch budget, or Patch-local point restriction is used by v0.5.
- 16×16 and 37×37 semantic grids use the same Fusion Head implementation.
- Existing unrelated working-tree changes must not be staged or overwritten.

---

### Task 1: Expose frozen DaD dense features and candidate pools

**Files:**
- Modify: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/pasm_v0/dad.py`
- Test: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/tests/test_pasm_v0_dad.py`

**Interfaces:**
- `FrozenDaDDetector` locally reproduces the native decoder loop and captures the `scale="2"` 32-channel context without modifying the external LoMa checkout; the final `scale="1"` context is one channel and is not used as fusion input.
- `FrozenDaDDetector.detect()` returns `DaDDetections(dense_scores, nms_points, nms_scores, dense_logits, feature_map)` with all arrays detached and on CPU.
- Add `select_dad_candidate_pool(detections, candidate_count) -> tuple[np.ndarray, np.ndarray]`, selecting the highest-scoring unique NMS points without applying the final 1024-point limit.

- [ ] **Step 1: Write failing shape/API tests.**

```python
def test_dad_detections_expose_dense_logits_and_32_channel_features(fake_detector):
    detections = fake_detector.detect(np.zeros((3, 32, 32), dtype=np.float32))
    assert detections.dense_logits.shape == (32, 32)
    assert detections.feature_map.shape == (32, 32, 32)

def test_candidate_pool_is_score_sorted_and_unique():
    detections = make_test_detections()
    points, scores = select_dad_candidate_pool(detections, candidate_count=3)
    assert len(points) == 3
    assert np.all(scores[:-1] >= scores[1:])
    assert len(np.unique(points, axis=0)) == 3
```

- [ ] **Step 2: Run the focused tests and verify they fail because the fields/function do not exist.**

Run from the PASM worktree:

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest tests/test_pasm_v0_dad.py -q
```

Expected: failure for the missing `dense_logits`, `feature_map`, and candidate-pool interface.

- [ ] **Step 3: Extend `DaDDetections` and `FrozenDaDDetector.detect()`.**

Store `scoremap[0, 0]` as `dense_logits` and `context[0]` as `[H, W, 32]` `feature_map`, while retaining the existing normalized `dense_scores`, NMS points, and NMS scores. All returned tensors must be detached before conversion to NumPy.

- [ ] **Step 4: Implement `select_dad_candidate_pool()`.**

Validate `candidate_count > 0`, sort NMS candidates by descending `nms_scores`, remove duplicate integer pixel cells, and return at most the requested count. Raise a clear error if the detector produces fewer valid candidates than requested.

- [ ] **Step 5: Run the focused tests and a CUDA smoke check.**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest tests/test_pasm_v0_dad.py -q
& 'D:\anaconda3\envs\loma\python.exe' -c "from pasm_v0.dad import FrozenDaDDetector; print('dad import ok')"
```

Expected: all focused tests pass and the import smoke check prints `dad import ok`.

- [ ] **Step 6: Commit only the DaD API changes.**

```powershell
git add 'pasm_v0\dad.py' 'tests\test_pasm_v0_dad.py'
git commit -m "feat: expose frozen dad dense features"
```

### Task 2: Implement residual dense Fusion Head

**Files:**
- Create: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/pasm_v0/point_fusion.py`
- Test: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/tests/test_pasm_v0_point_fusion.py`

**Interfaces:**
- `DensePointFusion(semantic_dim=128, dad_channels=32, hidden_channels=64, alpha_init=0.0)` is a shared per-image module.
- `DensePointFusion.forward(dad_features, base_logits, semantic_tokens) -> torch.Tensor` returns `[B, 1, H, W]` final logits.
- `select_dense_topk(final_logits, num_points=1024, nms_size=3) -> tuple[torch.Tensor, torch.Tensor]` returns normalized/subpixel points and their scores using the same DaD sampling semantics.
- `point_candidate_logits(final_logits, points_xy, image_size) -> torch.Tensor` bilinearly samples logits at candidate coordinates.

- [ ] **Step 1: Write failing tests for shape, residual initialization, and grid alignment.**

```python
def test_dense_point_fusion_starts_as_dad_baseline():
    model = DensePointFusion(alpha_init=0.0)
    dad = torch.randn(1, 32, 32, 32)
    base = torch.randn(1, 1, 32, 32)
    tokens = torch.randn(1, 16 * 16, 128)
    output = model(dad, base, tokens)
    assert output.shape == base.shape
    torch.testing.assert_close(output, base)

def test_dense_point_fusion_supports_16_and_37_grids():
    model = DensePointFusion()
    dad = torch.randn(1, 32, 32, 32)
    base = torch.randn(1, 1, 32, 32)
    for grid in (16, 37):
        output = model(dad, base, torch.randn(1, grid * grid, 128))
        assert output.shape == (1, 1, 32, 32)
```

- [ ] **Step 2: Run the focused tests and verify failure.**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest tests/test_pasm_v0_point_fusion.py -q
```

Expected: failure because `point_fusion.py` does not yet exist.

- [ ] **Step 3: Implement semantic projection, bilinear upsampling, and residual fusion.**

The implementation must reshape tokens to the configured grid, project `semantic_dim→dad_channels`, interpolate to `(H, W)`, concatenate with the DaD feature map and one-channel base logit, and produce a one-channel delta. Initialize the learnable residual scale to `alpha_init` and return `base_logits + alpha * delta`.

- [ ] **Step 4: Implement candidate sampling helpers.**

Use the existing DaD NMS/top-k/subpixel behavior and return coordinates in the same fixed 518×518 pixel convention used by `pasm_v0.dad`. Do not introduce Patch IDs or Patch budgets in this module.

- [ ] **Step 5: Run tests and a CPU shape smoke test.**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest tests/test_pasm_v0_point_fusion.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the Fusion Head.**

```powershell
git add 'pasm_v0\point_fusion.py' 'tests\test_pasm_v0_point_fusion.py'
git commit -m "feat: add residual dense point fusion head"
```

### Task 3: Implement geometry-derived candidate soft labels

**Files:**
- Create: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/pasm_v0/point_teacher.py`
- Test: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/tests/test_pasm_v0_point_teacher.py`

**Interfaces:**
- `build_candidate_teacher(source_points, target_points, geometry, direction, sigma_px, search_radius_px, depth_tolerance) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returns `(soft_labels, visible_mask, nearest_distances)`.
- Labels satisfy `q = visible * exp(-d²/(2*sigma²))` only when a target candidate is within `search_radius_px`; otherwise `q=0` and distance is `inf`.
- The implementation uses existing `visible_projected_points()` and the 518×518 geometry coordinate system.

- [ ] **Step 1: Write deterministic synthetic geometry tests.**

```python
def test_candidate_teacher_assigns_one_to_exact_projected_neighbor(identity_geometry):
    source = np.array([[10.0, 10.0]], dtype=np.float32)
    target = np.array([[10.0, 10.0]], dtype=np.float32)
    labels, visible, distances = build_candidate_teacher(
        source, target, identity_geometry, "a_to_b", sigma_px=3.0,
        search_radius_px=12.0, depth_tolerance=0.05,
    )
    assert visible.tolist() == [True]
    np.testing.assert_allclose(labels, [1.0])
    np.testing.assert_allclose(distances, [0.0])
```

- [ ] **Step 2: Run the focused test and verify failure.**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest tests/test_pasm_v0_point_teacher.py -q
```

Expected: failure because the teacher module is not implemented.

- [ ] **Step 3: Implement visibility, projection, and chunked nearest-neighbor search.**

Call `visible_projected_points()` to obtain valid source indices and projected target coordinates. Compute nearest target distances in NumPy chunks rather than constructing an unbounded full pairwise matrix. Apply visibility, finite-distance, and search-radius masks before computing the Gaussian soft label.

- [ ] **Step 4: Add symmetric A→B/B→A validation tests.**

Test an occluded point, an out-of-bounds projection, and a point beyond the search radius; each must receive label 0 and `visible_mask=False` or `nearest_distance=inf` as appropriate.

- [ ] **Step 5: Run all teacher tests and commit.**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest tests/test_pasm_v0_point_teacher.py -q
git add 'pasm_v0\point_teacher.py' 'tests\test_pasm_v0_point_teacher.py'
git commit -m "feat: add geometry candidate point teacher"
```

### Task 4: Add v0.5 training and checkpointing

**Files:**
- Create: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/pasm_v0/train_v05.py`
- Modify: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/pasm_v0/config.py`
- Test: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/tests/test_pasm_v0_train_v05.py`

**Interfaces:**
- CLI entry point: `python -m pasm_v0.train_v05`.
- Required arguments: `--manifest`, `--dino-weights`, `--dad-weights`, `--semantic-checkpoint`, and `--output`; image, depth, and camera paths are read from the existing manifest records.
- Defaults: `patch_grid=37`, `total_points=1024`, `teacher_candidate_count=4096`, `teacher_sigma_px=3.0`, `teacher_search_radius_px=12.0`, `depth_tolerance=0.05`, frozen backbones, and CUDA `auto` device resolution.
- Checkpoint payload contains `fusion_head`, optimizer state, epoch/history, semantic checkpoint path, patch grid, teacher parameters, and model dimensions.

- [ ] **Step 1: Add config validation tests.**

```python
def test_v05_config_rejects_nonpositive_teacher_parameters():
    with pytest.raises(ValueError):
        PointFusionTrainConfig(teacher_candidate_count=0)
    with pytest.raises(ValueError):
        PointFusionTrainConfig(teacher_sigma_px=0.0)
```

- [ ] **Step 2: Run the config test and verify failure.**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest tests/test_pasm_v0_train_v05.py -q
```

- [ ] **Step 3: Implement one training-pair step.**

The step must load 518×518 images, obtain frozen DaD detections/features, obtain frozen DINO tokens, load the trained `PairSemanticEncoder`, compute `semantic_a/semantic_b`, build 4096-point symmetric teacher labels, compute fused dense logits, sample logits at candidate coordinates, and backpropagate only through the fusion head and semantic projection.

- [ ] **Step 4: Implement losses and checkpoint save/resume.**

Use weighted BCE on candidate labels plus pairwise ranking loss. Set DaD, DINO, and semantic encoder parameters to `requires_grad=False` for the first implementation. Save both latest and best checkpoints using the explicit v0.5 metadata fields.

- [ ] **Step 5: Add a one-pair CPU synthetic training test.**

The test must run two optimizer steps on synthetic DaD tensors, semantic tokens, candidate coordinates, and labels, then assert that Fusion Head parameters change while frozen-backbone tensors have no gradients.

- [ ] **Step 6: Run tests and a one-pair CUDA smoke training.**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest tests/test_pasm_v0_train_v05.py -q
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.train_v05 --help
```

Expected: tests pass and CLI help lists the semantic checkpoint, teacher pool, sigma, search radius, and patch-grid arguments.

- [ ] **Step 7: Commit training support.**

```powershell
git add 'pasm_v0\train_v05.py' 'pasm_v0\config.py' 'tests\test_pasm_v0_train_v05.py'
git commit -m "feat: train v0.5 dense point fusion"
```

### Task 5: Add inference and nine-pair evaluation

**Files:**
- Create: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/pasm_v0/infer_v05.py`
- Create: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/pasm_v0/evaluate_v05.py`
- Test: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/tests/test_pasm_v0_evaluate_v05.py`

**Interfaces:**
- `load_v05_runner(checkpoint, dino_weights, dad_weights, semantic_checkpoint, device) -> V05Runner`.
- `V05Runner.select(image_a, image_b) -> (points_a, points_b, scores_a, scores_b, diagnostics)`.
- CLI `python -m pasm_v0.evaluate_v05` evaluates the fixed nine held-out pairs and writes `summary.csv`, `aggregate.csv`, selected-point images, budget-free score heatmaps, and LoMa match images.

- [ ] **Step 1: Write inference shape and no-geometry tests.**

```python
def test_v05_runner_returns_1024_points_per_image(fake_runner, image_pair):
    points_a, points_b, scores_a, scores_b, diagnostics = fake_runner.select(*image_pair)
    assert points_a.shape == (1024, 2)
    assert points_b.shape == (1024, 2)
    assert scores_a.shape == (1024,)
    assert scores_b.shape == (1024,)
    assert diagnostics["used_geometry"] is False
```

- [ ] **Step 2: Implement runner loading and inference.**

Load the v0.5 checkpoint, frozen DaD, DINO extractor, and semantic checkpoint. Run semantic cross-attention on both images, generate final dense logits, and apply DaD-style NMS/top-k/subpixel selection. Do not accept geometry paths in the runner API.

- [ ] **Step 3: Implement nine-pair evaluation using existing metrics.**

Reuse `load_heldout_pairs`, `load_pair_geometry`, `_metric_row`, `save_static_point_distribution`, and `save_match_visualization`. The evaluator may load geometry only for metric computation after inference; it must never pass geometry into `V05Runner.select()`.

- [ ] **Step 4: Add CLI and output validation tests.**

Verify that `summary.csv` contains 9 rows for one target, `aggregate.csv` contains the expected means, and every pair output includes selected points, score heatmaps, and LoMa matches.

- [ ] **Step 5: Run focused tests and commit.**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest tests/test_pasm_v0_evaluate_v05.py -q
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.evaluate_v05 --help
git add 'pasm_v0\infer_v05.py' 'pasm_v0\evaluate_v05.py' 'tests\test_pasm_v0_evaluate_v05.py'
git commit -m "feat: evaluate v0.5 dense point fusion"
```

### Task 6: Run ablations and verify the hypothesis

**Files:**
- Create: `D:/研0新方向/论文调研/Loma实验/PASM-v0/.worktrees/v0.3-pasm-dad-static/outputs/v0.5_ablation/` (generated outputs only)
- Modify: none unless a test exposes a documented defect

**Interfaces:**
- Evaluation consumes the v0.5 checkpoints produced by Task 4 and uses the v0.5 evaluator from Task 5.
- All comparison rows use the same nine held-out pairs and existing geometry metric definitions.

- [ ] **Step 1: Train and evaluate the DaD-only baseline through the new runner.**

Set fusion residual scale to zero and verify that selected point statistics match the existing frozen DaD global baseline within numerical tolerance.

- [ ] **Step 2: Train 16×16 and 37×37 fusion heads.**

Use identical teacher-candidate, sigma, search-radius, optimizer, and final-1024 settings; only `patch_grid` changes.

- [ ] **Step 3: Run the nine-pair evaluator.**

Save separate output directories for `fusion_16`, `fusion_37`, and `dad_baseline`; never overwrite earlier v0.3/v0.4 outputs.

- [ ] **Step 4: Compare metrics and inspect visualizations.**

Report mutual matches, 3px/6px inlier counts/rates, geometric coverage, selected-point heatmaps, teacher Top-K means, and failure cases. A fusion model is accepted only if it does not regress against DaD-only and improves at least one primary metric against the corresponding hard-budget model.

- [ ] **Step 5: Run the complete test suite and record the final commit.**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pytest -q
git status --short
```

Expected: all tests pass; unrelated pre-existing working-tree modifications remain unstaged.
