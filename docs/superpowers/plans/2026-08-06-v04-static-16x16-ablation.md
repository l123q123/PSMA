# v0.4 静态 16×16 Patch 对照实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 在不改变 v0.4 单轮静态训练和推理流程的前提下，增加 16×16 Patch 网格实验入口，用于与现有 37×37 模型做严格对照。

**Architecture:** 保留现有一层 PairSemanticEncoder、geometry `value_only` 损失、DaD 预算内选点和冻结 LoMa matcher。仅让配置、教师缓存、训练 CLI 和评测 CLI 接受 Patch 网格参数，默认值继续为 37。

**Tech Stack:** Python、PyTorch、NumPy、unittest、现有 PASM v0.4 模块。

## Global Constraints

- 默认 `patch_grid=37`，现有 v0.4 命令和 checkpoint 行为不变。
- 新实验使用 `patch_grid=16`、`total_points=1024`、`max_points_per_patch=4`、`budget_score_power=1.0`。
- 几何教师缓存使用 `geometry` 和 `stride=2`，16×16 缓存单独保存。
- 不修改 DINOv2、DaD、LoMa 权重、匹配规则、损失公式或 Cross-Attention 层数。
- 代码改动保持最小，不新增独立模型模块。

---

### Task 1: 增加可参数化的静态 Patch 配置

**Files:**
- Modify: `pasm_v0/config.py`
- Test: `tests/test_pasm_v0_train_v04.py`

**Interfaces:**
- `dino_token_static_config(patch_grid: int = 37) -> PasmV0Config` 返回指定正方形网格的单轮静态配置。
- `patch_grid=37` 保持当前配置完全一致；`patch_grid=16` 返回 `patch_rows=patch_cols=16`，其余参数保持当前 v0.4 值。
- 非 16/37 的网格值抛出 `ValueError`，避免误把像素 Patch 尺度当成网格数量。

- [ ] **Step 1: Write the failing test**

```python
def test_static_config_supports_16_patch_grid(self) -> None:
    from pasm_v0.config import dino_token_static_config

    config = dino_token_static_config(16)
    self.assertEqual((config.patch_rows, config.patch_cols), (16, 16))
    self.assertEqual(config.iterations, 1)
    self.assertEqual(config.total_points, 1024)

def test_static_config_rejects_unknown_patch_grid(self) -> None:
    from pasm_v0.config import dino_token_static_config

    with self.assertRaises(ValueError):
        dino_token_static_config(20)
```

- [ ] **Step 2: Run the focused test and confirm it fails because the function has no grid parameter.**

Run: `python -m unittest tests.test_pasm_v0_train_v04 -v`

- [ ] **Step 3: Implement the parameterized factory with the existing 37 default and the new 16 branch.**

- [ ] **Step 4: Run the focused tests and confirm both branches pass.**

- [ ] **Step 5: Commit:** stage the two files, then run `git commit -m "feat: parameterize static patch grid"`.

### Task 2: 让教师缓存和训练入口使用 16×16 配置

**Files:**
- Modify: `pasm_v0/v04_targets.py`
- Modify: `pasm_v0/cache_v04_targets.py`
- Modify: `pasm_v0/train_v04.py`
- Test: `tests/test_pasm_v0_v04_targets.py`
- Test: `tests/test_pasm_v0_train_v04.py`

**Interfaces:**
- `cache_targets(..., patch_grid: int = 37)` 用指定网格计算和校验 Patch 几何真值。
- `train_model(..., patch_grid: int = 37)` 用指定网格训练并把完整配置写入 checkpoint。
- 两个 CLI 都增加 `--patch-grid`，默认 37。

- [ ] **Step 1: Write a failing test asserting a 16×16 target cache has 256 values and metadata `patch_rows=16`, `patch_cols=16`.**
- [ ] **Step 2: Run the focused target tests and confirm the new parameter is missing.**
- [ ] **Step 3: Thread `patch_grid` into `dino_token_static_config`, cache metadata validation, training configuration, and CLI calls.**
- [ ] **Step 4: Add a checkpoint test asserting the saved config is 16×16 when `patch_grid=16`.**
- [ ] **Step 5: Run `python -m unittest tests.test_pasm_v0_v04_targets tests.test_pasm_v0_train_v04 -v`.**
- [ ] **Step 6: Commit:** stage the five files, then run `git commit -m "feat: train v04 on configurable patch grid"`.

### Task 3: 让 v0.4 评测按 checkpoint 网格工作

**Files:**
- Modify: `pasm_v0/evaluate_v04.py`
- Test: `tests/test_pasm_v0_evaluate_v04.py`

**Interfaces:**
- `evaluate_ablation(..., patch_grid: int = 37)` 使用所选网格构造布局和 DINO token 网格。
- CLI 增加 `--patch-grid`，并要求待比较 checkpoint 的配置与该网格一致。

- [ ] **Step 1: Write a failing test that passes `patch_grid=16` to the evaluation configuration helper and expects a 256-Patch layout.**
- [ ] **Step 2: Run the focused evaluation tests and confirm the parameter is missing.**
- [ ] **Step 3: Thread `patch_grid` through `evaluate_ablation` and CLI without改变已有 summary 字段。**
- [ ] **Step 4: Run `python -m unittest tests.test_pasm_v0_evaluate_v04 -v`.**
- [ ] **Step 5: Commit:** stage the two files, then run `git commit -m "feat: evaluate static patch grid ablation"`.

### Task 4: 完成最小验证并给出实验命令

**Files:**
- Verify: `pasm_v0/config.py`, `pasm_v0/v04_targets.py`, `pasm_v0/cache_v04_targets.py`, `pasm_v0/train_v04.py`, `pasm_v0/evaluate_v04.py`

- [ ] **Step 1: Run the complete relevant test set:** `python -m unittest tests.test_pasm_v0_sampling tests.test_pasm_v0_cli tests.test_pasm_v0_v04_targets tests.test_pasm_v0_train_v04 tests.test_pasm_v0_evaluate_v04 -v`.
- [ ] **Step 2: Run `python -m compileall -q pasm_v0`.
- [ ] **Step 3: Verify `--help` shows `--patch-grid` for cache, train, and evaluate commands.**
- [ ] **Step 4: Provide the user commands for generating 16×16 geometry cache and training; do not launch the long training automatically.**
