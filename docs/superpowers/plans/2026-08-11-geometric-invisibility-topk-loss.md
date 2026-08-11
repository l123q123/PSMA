# Geometric-Invisibility Soft Top-K Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training-only loss that suppresses geometrically invisible candidates when they receive high Soft Top-1024 selection mass.

**Architecture:** Reuse `soft_patch_occupancy`'s detached Top-K threshold to create candidate selection weights. A new pure loss consumes those weights and the existing `visible` geometry-teacher mask. The `semantic_coverage` training branch averages the loss for both directions, records it in checkpoint history, and leaves inference unchanged.

**Tech Stack:** Python, PyTorch, unittest, existing `pasm_v0.train_v05` training loop.

## Global Constraints

- Penalize only candidates with `visible=False`.
- Do not penalize geometrically visible candidates merely because their soft correspondence label is low.
- Do not change fixed DaD candidates, global Top-1024 inference, Patch coverage, or DaD local preservation.
- New loss weight is training-only and must not be read by `infer_v05.py`.

---

### Task 1: Add and test the pure invisibility loss

**Files:**
- Modify: `pasm_v0/train_v05.py`
- Test: `tests/test_pasm_v0_train_v05.py`

**Interfaces:**
- Add `geometric_invisibility_loss(candidate_logits, visible, topk, temperature) -> torch.Tensor`.
- It returns a scalar and keeps gradients with respect to `candidate_logits`.

- [ ] **Step 1: Write failing tests**

```python
visible = torch.tensor([True, False])
good = torch.tensor([4.0, -4.0], requires_grad=True)
bad = torch.tensor([-4.0, 4.0], requires_grad=True)
assert geometric_invisibility_loss(good, visible, 1, 0.5) < geometric_invisibility_loss(bad, visible, 1, 0.5)
```

Also assert an all-visible mask returns zero and that a high invisible logit has positive gradient, so gradient descent reduces it.

- [ ] **Step 2: Run the focused test and verify it fails**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_train_v05 -v
```

Expected: FAIL because `geometric_invisibility_loss` does not exist.

- [ ] **Step 3: Implement the loss**

```python
threshold = torch.topk(logits.detach(), min(topk, logits.numel())).values[-1]
selected = torch.sigmoid((logits - threshold) / temperature)
invisible = ~visible_mask
return selected[invisible].sum() / selected.sum().clamp_min(1e-6)
```

Validate nonempty finite logits, same-shape boolean visibility, positive `topk`, and positive `temperature`. Return `logits.sum() * 0.0` if all candidates are visible.

- [ ] **Step 4: Run focused tests**

Run the same unittest command. Expected: PASS.

### Task 2: Integrate semantic-coverage training history and CLI

**Files:**
- Modify: `pasm_v0/train_v05.py`
- Modify: `README.md`
- Test: `tests/test_pasm_v0_train_v05.py`

**Interfaces:**
- Add `invisibility_weight: float = 1.0` to `PointFusionTrainConfig` and `train_model`.
- Add CLI `--invisibility-weight`.
- Add history field `invisibility_loss`.

- [ ] **Step 1: Write failing config tests**

```python
assert PointFusionTrainConfig(invisibility_weight=0.5).invisibility_weight == 0.5
with self.assertRaises(ValueError):
    PointFusionTrainConfig(invisibility_weight=-0.1)
```

- [ ] **Step 2: Run the focused test and verify it fails**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_train_v05 -v
```

Expected: FAIL because the config field is absent.

- [ ] **Step 3: Integrate the loss only in `semantic_coverage` mode**

Compute the average of `geometric_invisibility_loss(final_a, visible_a, ...)` and the reverse direction. Use:

```python
total = geo + coverage_weight * coverage + local_preservation_weight * local + invisibility_weight * invisibility
```

Set `invisibility` to a detached zero scalar in every existing non-semantic-coverage branch. Add it to the progress text and history record. Add parser/config validation and document the command-line flag as training-only.

- [ ] **Step 4: Run full regression and smoke test**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest discover -s tests -v
```

Then run one epoch with `--fusion-mode semantic_coverage --invisibility-weight 1.0 --max-pairs 1`; expected: finite `invisible_loss` in the saved history.
