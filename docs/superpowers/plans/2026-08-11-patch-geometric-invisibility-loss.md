# Patch-Geometric-Invisibility Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert only the new invisibility loss from candidate-level suppression to fully-invisible-Patch occupancy suppression.

**Architecture:** Reuse `soft_patch_occupancy` and derive per-Patch candidate/visibility masks from the fixed DaD candidate coordinates. The training loop continues to average both image directions and retains existing reporting and CLI fields.

**Tech Stack:** Python, PyTorch, unittest.

## Global Constraints

- Leave `L_geo`, `L_cover`, `L_local`, inference, and all existing loss weights unchanged.
- Penalize only Patches with at least one candidate and zero geometrically visible candidates.
- Reuse the existing Soft Top-K temperature and K.

---

### Task 1: Replace the new loss with Patch-level occupancy supervision

**Files:**
- Modify: `pasm_v0/train_v05.py`
- Modify: `tests/test_pasm_v0_train_v05.py`
- Modify: `README.md`

**Interfaces:**
- `geometric_invisibility_loss(candidate_logits, candidates_xy, visible, image_size, patch_grid, topk, temperature) -> torch.Tensor`

- [ ] Write tests proving a fully invisible Patch with high Soft Top-K occupancy has higher loss, a mixed Patch is exempt, and an occupied fully invisible Patch receives a positive score gradient.
- [ ] Run `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_train_v05 -v` and verify the old candidate-level interface fails those tests.
- [ ] Implement the new loss by reusing `soft_patch_occupancy`, excluding empty Patches, and averaging occupancy only over eligible fully invisible Patches.
- [ ] Pass candidate coordinates, image size, and Patch grid from the `semantic_coverage` training branch; keep the existing total-loss expression and weight unchanged.
- [ ] Update README wording and run focused plus full tests, then one one-pair CUDA smoke run with `--fusion-mode semantic_coverage`.
