# Semantic-Coverage DaD-Preserving Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the first direct feature-fusion route so frozen Patch semantic tokens improve region coverage while frozen DaD features preserve within-Patch point preference, with no inference-time Patch budget or hand-tuned selection threshold.

**Architecture:** Add a candidate-level conditional fusion module that samples the frozen DaD 32-channel feature map at the fixed DaD NMS candidate pool, applies FiLM-style modulation from the candidate's Patch semantic token, and returns one fused logit per candidate. Keep the existing global candidate Top-K inference path. Add two structured training terms: a differentiable Patch occupancy coverage loss anchored by geometry teacher Patch support, and a within-Patch DaD distribution-preservation loss.

**Tech Stack:** Python 3, PyTorch, existing `pasm_v0.point_fusion`, `pasm_v0.train_v05`, unittest test suite, existing DaD candidate pool and geometry teacher.

## Global Constraints

- Freeze DINO, DaD, and the semantic encoder; only the new fusion head is trainable.
- Use the fixed original DaD NMS candidate pool and existing global Top-K selection.
- Do not add Patch quotas, Patch point caps, inference thresholds, distance radii, or static budgets.
- Use geometry only to build training support labels; do not redefine the semantic encoder's task.
- Preserve existing `residual`, `contrastive`, `candidate_suppression`, and `patch_gate` modes.
- New mode must initialize near the original DaD ranking (`fused_logits ≈ base_logits`).

---

### Task 1: Add candidate-level semantic-conditioned fusion

**Files:**
- Modify: `pasm_v0/point_fusion.py`
- Test: `tests/test_pasm_v0_point_fusion.py`

**Interfaces:**
- Add `SemanticCoverageDaDFusion(nn.Module)` with constructor `(semantic_dim=128, dad_channels=32, hidden_dim=64)`.
- Add `forward(dad_features, base_logits, semantic_tokens, candidates_xy, image_size) -> fused_logits` where inputs match `CandidateSuppressionFusion.forward` and output is `[N]`.
- Add factory mode `semantic_coverage` without changing existing modes.

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
model = SemanticCoverageDaDFusion()
dad = torch.randn(1, 32, 8, 8)
base = torch.randn(1, 1, 16, 16)
tokens = torch.randn(1, 16 * 16, 128)
xy = torch.tensor([[2.0, 3.0], [10.0, 11.0]])
out = model(dad, base, tokens, xy, 16)
assert out.shape == (2,)
```

Also assert zero-initialized output preserves the original candidate logits and that changing only the semantic token can change the fused logits while changing candidate order within a Patch still follows DaD when semantic tokens are identical.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_point_fusion -v
```

Expected: FAIL because `SemanticCoverageDaDFusion` and factory mode do not yet exist.

- [ ] **Step 3: Implement the minimal module**

Reuse `_candidate_semantic_tokens`, `_sample_dense_features`, and `point_candidate_logits`. Implement:

```python
semantic = _candidate_semantic_tokens(semantic_tokens, candidates, height, width)
dad = _sample_dense_features(dad_features, candidates, height, width)
base = point_candidate_logits(base_logits, candidates, (height, width))
gamma_beta = self.semantic_modulation(semantic)
gamma, beta = gamma_beta.chunk(2, dim=1)
modulated = (1.0 + 0.1 * torch.tanh(gamma)) * dad + beta
delta = self.candidate_head(torch.cat((modulated, semantic, base[:, None]), dim=1)).squeeze(1)
return base + self.residual_scale * delta
```

Initialize `residual_scale` and the last candidate-head layer to zero so the new mode starts as DaD. Do not apply a nonnegative penalty or a manually chosen semantic score multiplier.

- [ ] **Step 4: Run focused tests and commit**

Run the same unittest command; expected: all point-fusion tests PASS. Commit only the module and its tests:

```powershell
git add pasm_v0/point_fusion.py tests/test_pasm_v0_point_fusion.py
git commit -m "feat: add semantic conditioned candidate fusion"
```

### Task 2: Implement differentiable Patch coverage and local DaD preservation losses

**Files:**
- Modify: `pasm_v0/train_v05.py`
- Test: `tests/test_pasm_v0_train_v05.py`

**Interfaces:**
- Add `patch_support_targets(labels, visible, candidates_xy, image_size, patch_grid) -> torch.Tensor` returning `[P]` nonnegative support weights and `[P]` validity mask.
- Add `soft_patch_occupancy(candidate_logits, candidates_xy, patch_grid, topk, temperature) -> torch.Tensor` returning `[P]` values in `[0,1]`.
- Add `semantic_coverage_loss(candidate_logits, candidates_xy, support, valid, patch_grid, topk, temperature) -> torch.Tensor`.
- Add `local_dad_preservation_loss(fused_logits, base_logits, candidates_xy, support, valid, patch_grid, temperature) -> torch.Tensor`.

- [ ] **Step 1: Write failing unit tests**

Test that a candidate set with one high score in each supported Patch has lower coverage loss than a set whose scores all lie in one Patch. Test that coverage loss ignores invalid Patches. Test that identical fused/base logits produce near-zero local preservation loss, while swapping the within-Patch order produces a positive loss. Test all outputs are finite for an empty-valid mask.

- [ ] **Step 2: Run the focused tests and verify they fail**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_train_v05 -v
```

Expected: FAIL with missing function errors.

- [ ] **Step 3: Implement the loss helpers**

Build `support` from the geometry teacher without a new physical radius: aggregate each candidate's existing soft `labels` by Patch using `amax` over visible candidates, mark Patches with no visible candidates invalid, and detach the target. Use a detached soft Top-K threshold:

```python
threshold = torch.topk(candidate_logits.detach(), min(topk, candidate_logits.numel())).values[-1]
selected = torch.sigmoid((candidate_logits - threshold) / temperature).clamp(0.0, 1.0)
log_not_selected = torch.log1p(-selected.clamp(max=1.0 - 1e-6))
patch_log_not_selected = torch.zeros(patch_grid * patch_grid, device=selected.device)
patch_log_not_selected.scatter_add_(0, patch_ids, log_not_selected)
occupancy = 1.0 - torch.exp(patch_log_not_selected)
```

For `semantic_coverage_loss`, normalize only over valid supported Patches and compute weighted negative log occupancy. For `local_dad_preservation_loss`, compute per-Patch softmax distributions of fused and base logits and weighted KL divergence, skipping Patches with fewer than two candidates or invalid support. Keep all reductions finite when no Patch contributes.

- [ ] **Step 4: Run focused tests and commit**

Expected: all training-loss tests PASS. Commit:

```powershell
git add pasm_v0/train_v05.py tests/test_pasm_v0_train_v05.py
git commit -m "feat: add patch coverage and DaD preservation losses"
```

### Task 3: Integrate the new mode into training and checkpoints

**Files:**
- Modify: `pasm_v0/train_v05.py`
- Modify: `pasm_v0/infer_v05.py`
- Modify: `tests/test_pasm_v0_train_v05.py`
- Test: `tests/test_pasm_v0_infer_v05.py`

**Interfaces:**
- Extend `PointFusionTrainConfig` with `coverage_weight`, `local_preservation_weight`, `coverage_temperature`, and `coverage_topk`.
- Extend `train_model` and CLI with `--fusion-mode semantic_coverage`, `--coverage-weight`, `--local-preservation-weight`, `--coverage-temperature`, and `--coverage-topk`.
- Checkpoint metadata must save and restore the new mode and weights.

- [ ] **Step 1: Write failing integration tests**

Add a test that `train_options_from_args` accepts the new flags, creates the new fusion mode, and rejects nonpositive coverage temperature/top-K. Add an inference test that a checkpoint with `fusion_mode="semantic_coverage"` routes through candidate scoring and does not call dense Top-K.

- [ ] **Step 2: Run focused tests and verify failure**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_train_v05 tests.test_pasm_v0_infer_v05 -v
```

Expected: FAIL because the CLI, training branch, and checkpoint dispatch do not know the new mode.

- [ ] **Step 3: Integrate the training branch**

For each image direction, call the new fusion module on the fixed candidate pool, compute `L_geo`, `L_cover`, and `L_local`, and optimize:

```python
total = geo + settings.coverage_weight * cover + settings.local_preservation_weight * local
```

Record `coverage_loss` and `local_preservation_loss` in the history JSON and progress display. Preserve the existing branches unchanged for all other modes.

- [ ] **Step 4: Integrate inference/checkpoint dispatch**

Load the new mode from checkpoint metadata, sample the frozen DaD feature map at the existing candidate coordinates, calculate fused candidate logits, and pass them to `select_candidate_topk`. Keep the existing border filtering and candidate count behavior.

- [ ] **Step 5: Run integration tests and commit**

Expected: focused tests PASS. Commit only integration files:

```powershell
git add pasm_v0/train_v05.py pasm_v0/infer_v05.py tests/test_pasm_v0_train_v05.py tests/test_pasm_v0_infer_v05.py
git commit -m "feat: train and infer semantic coverage fusion"
```

### Task 4: Full regression and diagnostic outputs

**Files:**
- Modify: `README.md`
- Test: `tests/` (no new behavior beyond regression coverage)

- [ ] **Step 1: Run the complete unit suite**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest discover -s tests -v
```

Expected: all existing and new tests PASS.

- [ ] **Step 2: Run a one-pair smoke training**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.train_v05 `
  --manifest $manifest `
  --dino-weights $dino `
  --dad-weights $dad `
  --semantic-checkpoint $semantic `
  --output outputs\v0.6_ablation\semantic_coverage_smoke.pt `
  --epochs 1 `
  --patch-grid 16 `
  --fusion-mode semantic_coverage `
  --coverage-weight 1.0 `
  --local-preservation-weight 0.25 `
  --coverage-temperature 0.5 `
  --coverage-topk 1024 `
  --device cuda `
  --max-pairs 1
```

Expected: loss, coverage loss, and local preservation loss are finite; checkpoint and history JSON are written.

- [ ] **Step 3: Update usage documentation**

Document the new CLI mode, its training-only weights, and the fact that inference remains fixed-candidate global Top-K. Explicitly state that `coverage-topk` and `coverage-temperature` affect only the differentiable training surrogate.

- [ ] **Step 4: Commit documentation and report handoff**

```powershell
git add README.md
git commit -m "docs: document semantic coverage fusion training"
```

After this, run the nine-pair relative-pose evaluation against `fusion-16-main6px` using identical candidate pool and matching settings, and compare success count, median pose error, AUC, Patch coverage, and within-Patch DaD rank correlation.
