# v0.6 Patch-Gated DaD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace candidate-level suppression with an independently trained Patch gate that preserves original DaD ordering within trusted Patches.

**Architecture:** Keep the frozen DaD 4096-candidate pool and the frozen pair semantic encoder. Replace `CandidateSuppressionFusion`'s DaD-feature and candidate-penalty branches with a single semantic-token gate head. During training, aggregate the existing candidate geometric teacher by Patch-wise maximum over reliably visible candidates and fit independent sigmoid gate probabilities with BCE. During inference, subtract a Patch-constant penalty from fixed DaD candidate logits before existing border filtering and global Top-1024.

**Tech Stack:** Python 3, PyTorch, NumPy, existing `unittest` suite.

## Global Constraints

- Keep DaD, DINOv2, and the pair semantic encoder frozen.
- Keep the original 3x3-NMS top-4096 DaD candidate pool and the 8-pixel border exclusion.
- Do not use Patch-budget allocation, power normalization, candidate ranking loss, or candidate-level point penalty in this mode.
- Keep legacy `residual`, `contrastive`, and existing `candidate_suppression` checkpoint loading behavior unchanged.

---

### Task 1: Define and test Patch-gate primitives

**Files:**
- Modify: `pasm_v0/point_fusion.py`
- Modify: `pasm_v0/train_v05.py`
- Test: `tests/test_pasm_v0_point_fusion.py`
- Test: `tests/test_pasm_v0_train_v05.py`

**Interfaces:**
- Produces `PatchGatedDaDFusion.forward(base_logits, semantic_tokens, candidates_xy, image_size) -> tuple[Tensor, Tensor]`, where outputs are `[N]` final candidate logits and `[P]` independent Patch gate probabilities (`P = patch_grid²`).
- Produces `patch_gate_targets(labels, visible, candidates_xy, image_size, patch_grid) -> Tensor`, returning `[patch_grid*patch_grid]` teacher values in `[0,1]` or `-1` for unknown Patches.
- Produces `patch_gate_loss(gates_a, targets_a, gates_b, targets_b) -> Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
def test_patch_gate_preserves_within_patch_dad_logit_differences_when_gate_is_shared(self) -> None:
    model = PatchGatedDaDFusion()
    with torch.no_grad():
        model.semantic_gate_head.weight.zero_()
        model.semantic_gate_head.bias.fill_(20.0)
    base = torch.zeros(1, 1, 16, 16)
    base[0, 0, 1, 1] = 3.0
    base[0, 0, 1, 2] = 1.0
    tokens = torch.zeros(1, 2 * 2, 128)
    candidates = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
    final, gates = model(base, tokens, candidates, image_size=16)
    torch.testing.assert_close(final[0] - final[1], torch.tensor(2.0))
    self.assertGreater(float(gates.min()), 0.99)

def test_patch_gate_targets_take_visible_candidate_maximum_and_ignore_unknown_patch(self) -> None:
    targets = patch_gate_targets(
        labels=torch.tensor([0.2, 0.9, 1.0]),
        visible=torch.tensor([True, True, False]),
        candidates_xy=torch.tensor([[1.0, 1.0], [2.0, 2.0], [14.0, 14.0]]),
        image_size=16,
        patch_grid=2,
    )
    torch.testing.assert_close(targets, torch.tensor([0.9, -1.0, -1.0, -1.0]))
```

- [ ] **Step 2: Run the two new tests to verify RED**

Run:

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_point_fusion tests.test_pasm_v0_train_v05 -v
```

Expected: FAIL because `PatchGatedDaDFusion`, `patch_gate_targets`, and `patch_gate_loss` do not exist.

- [ ] **Step 3: Implement the minimal Patch gate**

```python
class PatchGatedDaDFusion(nn.Module):
    def __init__(self, semantic_dim: int = 128, penalty_scale: float = 3.0, quality_gamma: float = 2.0) -> None:
        super().__init__()
        self.semantic_gate_head = nn.Linear(semantic_dim, 1)
        self.penalty_scale = float(penalty_scale)
        self.quality_gamma = float(quality_gamma)

    def forward(self, base_logits, semantic_tokens, candidates_xy, image_size):
        gates = torch.sigmoid(self.semantic_gate_head(semantic_tokens[0])).squeeze(1)
        candidate_patch_ids = _candidate_patch_ids(candidates_xy, gates.numel(), image_size)
        base = point_candidate_logits(base_logits, candidates_xy, image_size)
        return base - self.penalty_scale * (1.0 - gates[candidate_patch_ids]).pow(self.quality_gamma), gates
```

Implement `patch_gate_targets` by assigning each reliably visible candidate to its semantic Patch and using `scatter_reduce_(amax)` with an accompanying observed mask. Implement BCE only over `targets >= 0` and average the two views in `patch_gate_loss`.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run:

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_point_fusion tests.test_pasm_v0_train_v05 -v
```

Expected: PASS, including the new Patch-gate behavior tests.

### Task 2: Connect Patch-gated training and checkpoint configuration

**Files:**
- Modify: `pasm_v0/train_v05.py`
- Modify: `tests/test_pasm_v0_train_v05.py`

**Interfaces:**
- Adds fusion mode `patch_gate` to `PointFusionTrainConfig` and CLI `--fusion-mode` choices.
- Adds configuration fields `gate_penalty_scale` default `3.0` and `gate_quality_gamma` default `2.0`.
- For `patch_gate`, training history stores `total_loss` and `point_loss` as the gate BCE; legacy loss fields remain zero for backward-compatible JSON shape.

- [ ] **Step 1: Write the failing CLI/config test**

```python
def test_training_config_accepts_patch_gate_and_rejects_invalid_gate_parameters(self) -> None:
    self.assertEqual(PointFusionTrainConfig(fusion_mode="patch_gate").fusion_mode, "patch_gate")
    with self.assertRaises(ValueError):
        PointFusionTrainConfig(gate_penalty_scale=0.0)
    with self.assertRaises(ValueError):
        PointFusionTrainConfig(gate_quality_gamma=0.0)
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_train_v05.PointFusionTrainingTests.test_training_config_accepts_patch_gate_and_rejects_invalid_gate_parameters -v
```

Expected: FAIL because the mode and settings are absent.

- [ ] **Step 3: Implement the training branch**

In `train_model`, retain `build_candidate_teacher` exactly as-is. When `fusion_mode == "patch_gate"`, convert both view labels to Patch targets using `source_config.patch_rows`, call the gate with base logits, semantic tokens, candidate coordinates, and image size, then train the returned `[P]` gate vectors with only `patch_gate_loss`. Do not prepare DaD feature tensors for this branch. Persist the two gate parameters in `settings`; expose them as CLI flags and pass them to `create_point_fusion` through a dedicated constructor branch.

- [ ] **Step 4: Run the training test module to verify GREEN**

Run:

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_train_v05 -v
```

Expected: PASS.

### Task 3: Connect Patch-gated inference and run regression tests

**Files:**
- Modify: `pasm_v0/infer_v05.py`
- Modify: `README.md`

**Interfaces:**
- `load_v05_runner` recognizes `fusion_mode == "patch_gate"` and recreates the gate using persisted settings.
- `_select_suppressed_candidates` accepts either legacy candidate suppression or Patch-gate fusion and always returns fixed-pool global Top-1024.

- [ ] **Step 1: Implement inference mode dispatch and documentation**

Keep candidate-pool selection, border filtering, and `select_candidate_topk` unchanged. Dispatch `patch_gate` to `fusion(base_logits, semantic_tokens, candidate_tensor, image_size)` and legacy `candidate_suppression` to its existing five-argument call. Document a fresh training command that includes `--fusion-mode patch_gate`, `--gate-penalty-scale 3.0`, and `--gate-quality-gamma 2.0`.

- [ ] **Step 2: Run the focused complete regression suite**

Run:

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_point_fusion tests.test_pasm_v0_train_v05 tests.test_pasm_v0_infer_v05 -v
```

Expected: PASS with all existing and newly added focused tests.
