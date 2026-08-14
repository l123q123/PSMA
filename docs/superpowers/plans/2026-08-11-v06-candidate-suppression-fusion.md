# v0.6 Candidate Suppression Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and infer a feature-level fusion model that only downweights fixed original DaD candidates instead of generating a new dense response map.

**Architecture:** A new candidate scorer consumes sampled DaD features, base logits, and the dual-image semantic token at each fixed DaD NMS candidate. Its non-negative penalty is subtracted from the base logit. The train and infer pipelines use the same fixed-candidate selection path.

**Tech Stack:** Python, PyTorch, existing PASM v0.5 DaD/DINO/semantic backbones, unittest.

## Global Constraints

- DaD, DINO, and semantic encoder stay frozen.
- The new fusion path must never increase an original DaD score.
- Candidate coordinates must originate only from original DaD NMS.
- No Patch budget or per-Patch candidate quota is introduced.
- Existing residual and contrastive fusion modes remain usable.

---

### Task 1: Candidate suppression scorer

**Files:**
- Modify: `pasm_v0/point_fusion.py`
- Test: `tests/test_pasm_v0_point_fusion.py`

**Interfaces:**
- Produces `CandidateSuppressionFusion.forward(dad_features, base_logits, semantic_tokens, candidates, image_size) -> (final_logits, penalties)`.
- `final_logits` and `penalties` have shape `[N]`; `penalties >= 0`; `final_logits <= sampled_base_logits`.

- [ ] Write failing tests for non-negative penalties, no score increase, and weaker suppression at high semantic quality.
- [ ] Run the focused unittest and confirm the new class is missing.
- [ ] Implement minimal candidate feature sampling, token lookup, and constrained penalty parameterization.
- [ ] Run the focused unittest and confirm it passes.

### Task 2: Candidate-label suppression loss

**Files:**
- Modify: `pasm_v0/train_v05.py`
- Test: `tests/test_pasm_v0_train_v05.py`

**Interfaces:**
- Produces `candidate_suppression_loss(final_logits, penalties, labels, ...) -> (total, keep, suppress, rank)`.
- Ignores teacher-unknown candidates while preserving symmetric A/B training.

- [ ] Write failing tests for positive keep loss, negative suppression loss, ignored labels, and rank ordering.
- [ ] Run the focused unittest and confirm the function is missing.
- [ ] Implement the three-term loss and CLI/config validation for `fusion_mode=candidate_suppression`.
- [ ] Run focused tests and existing v0.5 trainer tests.

### Task 3: Fixed-candidate inference path

**Files:**
- Modify: `pasm_v0/infer_v05.py`
- Modify: `pasm_v0/evaluate_v05.py`
- Test: `tests/test_pasm_v0_point_fusion.py`

**Interfaces:**
- Candidate suppression mode uses original DaD NMS candidates, configurable candidate-pool size, and `border_margin_px` before global Top-K.
- Returned points are a subset of the original candidate positions and contain no point in the hard border band.

- [ ] Write failing tests for candidate-only output and border masking.
- [ ] Run the focused unittest and confirm the behavior is absent.
- [ ] Implement candidate-pool construction, scorer invocation, filtering, and Top-K selection.
- [ ] Run focused inference tests and the full point-fusion test module.

### Task 4: Checkpoint and command-line compatibility

**Files:**
- Modify: `pasm_v0/train_v05.py`
- Modify: `pasm_v0/infer_v05.py`
- Test: `tests/test_pasm_v0_train_v05.py`

**Interfaces:**
- Candidate-suppression checkpoints record its format, fusion mode, candidate-pool size, border margin, and suppression-loss settings.
- Existing residual/contrastive checkpoints retain their current loading behavior.

- [ ] Write failing checkpoint metadata and loading validation tests.
- [ ] Run tests and confirm they fail for the missing metadata/branch.
- [ ] Implement metadata persistence and mode-specific loading.
- [ ] Run all v0.5-related unit tests and compile the PASM package.

### Task 5: User-run training/evaluation handoff

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents one PowerShell training command using `main_6px_1024_60.pt` and one nine-pair evaluation command.

- [ ] Add concise commands and parameter definitions for candidate suppression mode.
- [ ] Run `--help` for training and evaluation commands.
- [ ] Report the exact output locations expected from the nine-pair run.
