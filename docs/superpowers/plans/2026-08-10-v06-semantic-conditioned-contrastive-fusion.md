# v0.6 Semantic-Conditioned Contrastive Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checkpoint-compatible contrastive dense point fusion mode that modulates DaD local contrast with paired-image semantic features.

**Architecture:** Preserve the v0.5 residual fusion class. Add an independent fusion class plus a mode factory; train and infer select the class using checkpoint metadata. Replace the vanishing hinge ranking only when the new ranking mode is explicitly selected.

**Tech Stack:** Python 3, PyTorch, unittest.

## Global Constraints

- Keep existing residual fusion checkpoints loadable without conversion.
- Keep DaD, DINO, and semantic encoder frozen during v0.6 fusion training.
- All spatial convolutions that need padding use `padding_mode="reflect"`.
- Default CLI behavior remains v0.5 residual + hinge ranking.

---

### Task 1: Contrastive fusion primitives

**Files:**
- Modify: `pasm_v0/point_fusion.py`
- Test: `tests/test_pasm_v0_point_fusion.py`

**Interfaces:**
- Produces `local_logit_zscore(logits, window_size=5, eps=1e-6) -> Tensor`.
- Produces `ContrastiveDensePointFusion(...).forward(dad_features, base_logits, semantic_tokens) -> Tensor`.
- Produces `create_point_fusion(mode) -> nn.Module`.

- [ ] Write failing tests for finite local z-scores, contrast direction, identity initialization, and factory validation.
- [ ] Run the point-fusion tests and confirm the new tests fail because the new symbols do not exist.
- [ ] Implement the z-score helper, contrastive fusion class, and factory.
- [ ] Re-run the point-fusion tests and confirm they pass.

### Task 2: Persistent soft pairwise rank loss and metadata

**Files:**
- Modify: `pasm_v0/train_v05.py`
- Test: `tests/test_pasm_v0_train_v05.py`

**Interfaces:**
- Extends `PointFusionTrainConfig` with `fusion_mode`, `ranking_mode`, and `ranking_temperature`.
- Extends `point_fusion_loss(..., ranking_mode="hinge", ranking_temperature=1.0)`.
- Stores the three settings in each checkpoint.

- [ ] Write failing tests that soft pairwise loss is positive and differentiable despite a satisfied hinge margin, and invalid modes/temperatures are rejected.
- [ ] Run the training tests and confirm expected failure.
- [ ] Implement rank-mode dispatch, soft pairwise loss, CLI options, fusion factory selection, and checkpoint metadata.
- [ ] Re-run the training tests and confirm they pass.

### Task 3: Checkpoint-driven inference selection

**Files:**
- Modify: `pasm_v0/infer_v05.py`
- Test: `tests/test_pasm_v0_evaluate_v05.py`

**Interfaces:**
- `load_v05_runner` uses checkpoint `fusion_mode`; missing field maps to `residual`.

- [ ] Write a failing loader test that records the requested factory mode for a legacy checkpoint and a contrastive checkpoint.
- [ ] Run the evaluation tests and confirm expected failure.
- [ ] Implement checkpoint mode validation and factory-based model construction.
- [ ] Run all v0.5 unit tests and byte-compile the touched modules.
