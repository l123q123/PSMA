# Nine-Pair Relative Pose Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score saved nine-pair LoMa matches with reproducible essential-matrix relative-pose metrics.

**Architecture:** A standalone evaluator reads existing `result.npz` files and held-out pair metadata, estimates pose from normalized matched coordinates, and saves per-pair and aggregate CSV files. It never calls the point selector, matcher, depth projection, or model checkpoint loader.

**Tech Stack:** Python 3, NumPy, OpenCV, unittest.

## Global Constraints

- Pose estimation may use only matched image coordinates and camera intrinsics.
- Ground-truth camera poses are used only to score estimated poses.
- Failed estimates are retained as explicit 180-degree failures.

---

### Task 1: Pure relative-pose metrics

**Files:**
- Create: `pasm_v0/evaluate_pose.py`
- Create: `tests/test_pasm_v0_evaluate_pose.py`

- [ ] Write failing tests for identity pose recovery, rotation/translation angular error, and AUC with explicit failures.
- [ ] Run the new test module and observe its expected import failure.
- [ ] Implement coordinate normalization, essential-matrix recovery, angular errors, and AUC.
- [ ] Run the new test module and confirm it passes.

### Task 2: Saved-output nine-pair runner

**Files:**
- Modify: `pasm_v0/evaluate_pose.py`
- Modify: `tests/test_pasm_v0_evaluate_pose.py`

- [ ] Write a failing temporary-directory test that saves two `result.npz` files, executes the runner, and verifies `pose_summary.csv` and `pose_aggregate.csv`.
- [ ] Run the test to confirm expected failure.
- [ ] Implement directory mapping, CSV output, and CLI arguments.
- [ ] Run the pose tests, then run the evaluator over the selected nine-pair fusion output.
