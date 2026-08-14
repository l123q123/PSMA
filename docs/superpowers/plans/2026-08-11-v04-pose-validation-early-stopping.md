# v0.4 位姿验证早停 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于固定 held-out 验证集的 `pose_auc_10deg` 为 v0.4 语义头选择最佳 checkpoint 并实现可恢复的早停。

**Architecture:** `train_v04.py` 在指定 epoch 保存 `.last.pt` 后复用现有 `evaluate_v03.evaluate_pairs` 和 `evaluate_pose.evaluate_saved_matches`。从 `pose_aggregate.csv` 读取 AUC，并将最佳指标和连续未提升次数写入 checkpoint。

**Tech Stack:** Python、PyTorch、现有 PASM v0.3 静态评测、OpenCV RANSAC。

## Global Constraints

- 默认验证间隔为 2 epoch，默认 patience 为 5 次验证，默认指标为 `pose_auc_10deg`。
- 未传验证图像目录时必须完全保留现有按训练 loss 选最佳模型的行为。
- 验证目录必须为 `<output_stem>_validation/epoch_XXX/`。
- 不修改训练损失或融合头。

---

### Task 1: 早停决策与 checkpoint 元数据

**Files:**
- Modify: `pasm_v0/train_v04.py`
- Test: `tests/test_pasm_v0_train_v04.py`

**Interfaces:**
- Produces: `update_validation_early_stopping(metric, best_metric, no_improvement_count, patience) -> tuple[bool, float, int, bool]`。
- Produces: `save_v04_checkpoint(..., best_validation_metric, validation_no_improvement)` 和对应恢复逻辑。

- [ ] **Step 1: Write the failing test**

```python
improved, best, count, stop = update_validation_early_stopping(0.31, 0.30, 2, 5)
self.assertEqual((improved, best, count, stop), (True, 0.31, 0, False))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_train_v04.PasmV04TrainTests.test_validation_early_stopping_resets_on_auc_improvement -v`

Expected: FAIL because `update_validation_early_stopping` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def update_validation_early_stopping(metric, best_metric, no_improvement_count, patience):
    if metric > best_metric:
        return True, metric, 0, False
    count = no_improvement_count + 1
    return False, best_metric, count, count >= patience
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_train_v04.PasmV04TrainTests.test_validation_early_stopping_resets_on_auc_improvement -v`

Expected: PASS.

### Task 2: 训练内验证与 CLI

**Files:**
- Modify: `pasm_v0/train_v04.py`
- Test: `tests/test_pasm_v0_train_v04.py`

**Interfaces:**
- Consumes: `evaluate_pairs(...)` 和 `evaluate_saved_matches(...)` 的现有输出。
- Produces: CLI 参数 `--validation-image-root`、`--validation-every`、`--early-stop-patience`、`--validation-max-pairs`、`--validation-ransac-threshold-px`。

- [ ] **Step 1: Write the failing test**

```python
with patch("sys.argv", ["train_v04", "--help"]), redirect_stdout(stdout):
    with self.assertRaises(SystemExit):
        train_v04_main()
self.assertIn("--validation-image-root", stdout.getvalue())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_train_v04.PasmV04TrainTests.test_v04_cli_exposes_pose_validation_options -v`

Expected: FAIL because the option is absent.

- [ ] **Step 3: Write minimal implementation**

```python
validation_root = output.with_name(f"{output.stem}_validation") / f"epoch_{epoch + 1:03d}"
evaluate_pairs(..., output=validation_root, ...)
evaluate_saved_matches(..., match_output=validation_root, output=validation_root / "pose", result_subdir="pasm_dad")
```

- [ ] **Step 4: Run focused and module tests**

Run: `D:\anaconda3\envs\loma\python.exe -m unittest tests.test_pasm_v0_train_v04 -v`

Expected: PASS.
