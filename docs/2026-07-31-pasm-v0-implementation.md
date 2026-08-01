# PASM v0 初版实施记录

> **迁移说明：** 本文保留开发时的原始路径和测试步骤，现已迁移至独立的 `PASM-v0` 项目。请以项目根目录 `README.md` 的当前命令和目录结构为准。

> **供执行代理使用：** 当前用户已要求在本会话内直接执行本计划；按任务顺序执行，每一步使用测试先行。

**目标：** 在独立的 `PSMA/pasm_v0` 目录实现每图 512 点、Patch 管预算、全局点级匹配和 Patch 内重采样的 PASM v0。

**架构：** `sampling.py` 只处理 Patch 网格、预算和坐标重采样；`matching.py` 只处理双图语义条件化和全局点级边分数；`pipeline.py` 将二者串成三轮流程。DINOv2 和 LoMa 作为冻结特征来源，通过 `backbones.py` 接入，不复用旧实现中的 Patch 对候选逻辑。

**技术栈：** Python、NumPy、PyTorch、Pillow、已安装的 LoMa 包和本地 DINOv2/LoMa 权重。

## 全局约束

- 所有新代码位于 `PSMA/pasm_v0`，不修改旧 `PSMA/pasm` 的候选 Patch 对策略。
- 每个 Python 文件开头使用简短中文模块注释；每个函数和方法使用一句中文 docstring。
- 输入默认缩放为 `518×518`，管理 Patch 网格固定为 `16×16`。
- 第 0 轮每 Patch 2 点；第 1 轮每 Patch 至少 1 点；第 2 轮起允许 0 点。
- 每图每轮精确使用 512 点，Patch 点数上限、探索比例和扰动范围必须可配置。
- 不实现 Patch 对、Top-K 候选、dustbin、拒绝头、教师和长期点状态。
- 单元测试只覆盖预算、边分数、重采样边界、三轮流程和训练目标的关键行为。
- 当前目录不是 Git 仓库；不执行提交。

---

### 任务 1：建立配置、Patch 网格和采样器

**文件：**

- 新建：`PSMA/pasm_v0/__init__.py`
- 新建：`PSMA/pasm_v0/config.py`
- 新建：`PSMA/pasm_v0/sampling.py`
- 新建：`PSMA/tests/test_pasm_v0_sampling.py`

**接口：**

- `PasmV0Config` 提供 `image_size`、`patch_rows`、`patch_cols`、`total_points`、`round_min_points`、`max_points_per_patch`、`exploration_ratio`、`jitter_ratio`。
- `PatchLayout(config)` 提供 `patch_ids(points_xy)`、`bounds(patch_id)`、`initial_points(seed)`。
- `allocate_patch_budget(values, config, round_index)` 返回长度为 256、和为 512 的整数预算。
- `resample_in_patches(layout, previous_points, previous_patch_ids, point_quality, budgets, config, seed)` 返回新的 `[512,2]` 坐标和 `[512]` Patch id。

- [ ] **步骤 1：写失败测试**

```python
def test_initial_sampling_gives_two_points_to_every_patch(self):
    config = PasmV0Config(image_size=32, patch_rows=4, patch_cols=4, total_points=32)
    layout = PatchLayout(config)
    points, patch_ids = layout.initial_points(seed=7)
    self.assertEqual(points.shape, (32, 2))
    np.testing.assert_array_equal(np.bincount(patch_ids, minlength=16), np.full(16, 2))

def test_later_budget_can_eliminate_low_value_patches(self):
    config = PasmV0Config(image_size=32, patch_rows=4, patch_cols=4, total_points=32, max_points_per_patch=4)
    values = np.zeros(16, dtype=np.float32)
    values[3] = 1.0
    budget = allocate_patch_budget(values, config, round_index=2)
    self.assertEqual(int(budget.sum()), 32)
    self.assertEqual(int(budget[0]), 0)
    self.assertGreater(int(budget[3]), 0)
```

- [ ] **步骤 2：运行测试并确认失败**

运行：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest PSMA.tests.test_pasm_v0_sampling -v
```

预期：因 `PSMA.pasm_v0` 尚不存在而失败。

- [ ] **步骤 3：实现最小采样逻辑**

```python
@dataclass(frozen=True)
class PasmV0Config:
    image_size: int = 518
    patch_rows: int = 16
    patch_cols: int = 16
    total_points: int = 512
    round_min_points: tuple[int, ...] = (2, 1, 0)
    max_points_per_patch: int = 4

def allocate_patch_budget(values, config, round_index):
    minimum = config.round_min_points[min(round_index, len(config.round_min_points) - 1)]
    # 先分配轮次下限，再按归一化价值分配余量，并遵守每 Patch 上限。
```

实现分层随机初始采样；重采样时在每个 Patch 保留探索点，其余位置以本 Patch 高质量旧点为中心加入扰动，并将全部坐标裁剪在各自 Patch 边界内。

- [ ] **步骤 4：运行测试并确认通过**

运行：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest PSMA.tests.test_pasm_v0_sampling -v
```

预期：预算精确为 32，首轮均匀覆盖，第三轮可分配 0 点，重采样不离开所属 Patch。

### 任务 2：实现双图语义条件化与全局点级匹配

**文件：**

- 新建：`PSMA/pasm_v0/matching.py`
- 新建：`PSMA/tests/test_pasm_v0_matching.py`

**接口：**

- `PairSemanticEncoder(token_dim, hidden_dim, heads)` 是一层共享的双向 Cross-Attention，输入 `[B,256,C]`，输出两张图的条件化特征和每个 Patch 的语义价值。
- `sample_feature_map(feature_map, points_xy, image_size)` 从 `[C,H,W]` LoMa 特征图双线性取点描述子。
- `global_edge_weights(descriptors_a, descriptors_b, semantic_a, semantic_b, patch_ids_a, patch_ids_b, temperature)` 返回 `[N,N]` dual-softmax 软边权。
- `patch_feedback(edge_weights, patch_ids, patch_count)` 返回每个 Patch 的点级边质量聚合值。

- [ ] **步骤 1：写失败测试**

```python
def test_identity_descriptors_make_diagonal_edges_strongest(self):
    descriptors = np.eye(3, dtype=np.float32)
    semantics = np.eye(3, dtype=np.float32)
    weights = global_edge_weights(descriptors, descriptors, semantics, semantics,
                                  np.arange(3), np.arange(3), temperature=0.1)
    np.testing.assert_array_equal(weights.argmax(axis=1), np.arange(3))

def test_patch_feedback_aggregates_point_level_edges(self):
    weights = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    feedback = patch_feedback(weights, np.array([0, 0, 1]), patch_count=2)
    self.assertGreater(feedback[0], feedback[1])
```

- [ ] **步骤 2：运行测试并确认失败**

运行：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest PSMA.tests.test_pasm_v0_matching -v
```

预期：因匹配模块尚不存在而失败。

- [ ] **步骤 3：实现最小匹配逻辑**

```python
class PairSemanticEncoder(nn.Module):
    def forward(self, tokens_a, tokens_b):
        # 投影后进行 A←B 和 B←A 的单层 Cross-Attention，并输出每个 Patch 价值。

def global_edge_weights(...):
    # LoMa 余弦相似度与所属 Patch 条件化语义余弦相加后执行双向 softmax。
```

语义只作为点级边的软支持，绝不把边筛成预设 Patch 对；所有当前点对均参与矩阵计算。

- [ ] **步骤 4：运行测试并确认通过**

运行：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest PSMA.tests.test_pasm_v0_matching -v
```

预期：对角模拟对应为最高边，Patch 反馈来自点级边而非 Patch 对。

### 任务 3：串联三轮流程与几何训练目标

**文件：**

- 新建：`PSMA/pasm_v0/pipeline.py`
- 新建：`PSMA/pasm_v0/training.py`
- 新建：`PSMA/tests/test_pasm_v0_pipeline.py`

**接口：**

- `PasmV0(model, config).run(tokens_a, tokens_b, descriptor_grid_a, descriptor_grid_b, seed)` 返回 `PasmV0Result`，包含最终点、最终边权和每轮 `RoundRecord`。
- `point_match_targets(points_a, points_b, geometry_a, geometry_b, threshold_px)` 返回可监督的目标列及有效行掩码。
- `patch_support_targets(geometry_points, layout)` 返回归一化的 Patch 正区域分布。
- `pasm_v0_loss(edge_weights, target_columns, valid_rows, patch_values, patch_targets)` 返回点级匹配损失与 Patch 价值损失之和。

- [ ] **步骤 1：写失败测试**

```python
def test_pipeline_keeps_512_points_and_allows_zero_budget_on_last_round(self):
    config = PasmV0Config(image_size=32, patch_rows=4, patch_cols=4, total_points=32)
    model = PairSemanticEncoder(token_dim=8, hidden_dim=8, heads=2)
    result = PasmV0(model, config).run(
        tokens_a=torch.randn(1, 16, 8), tokens_b=torch.randn(1, 16, 8),
        descriptor_grid_a=np.random.randn(4, 8, 8).astype(np.float32),
        descriptor_grid_b=np.random.randn(4, 8, 8).astype(np.float32), seed=3,
    )
    self.assertEqual(result.points_a.shape, (32, 2))
    self.assertEqual(len(result.rounds), 3)
    self.assertTrue(np.any(result.rounds[-1].budget_a == 0))
```

- [ ] **步骤 2：运行测试并确认失败**

运行：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest PSMA.tests.test_pasm_v0_pipeline -v
```

预期：因三轮流程和训练目标函数尚不存在而失败。

- [ ] **步骤 3：实现最小流程与损失**

```python
for round_index in range(config.iterations):
    semantic_a, semantic_b, value_a, value_b = model(tokens_a, tokens_b)
    weights = global_edge_weights(...)
    budget_a = allocate_patch_budget(value_a + patch_feedback(...), config, round_index + 1)
    points_a, ids_a = resample_in_patches(...)
```

第 0 轮使用固定初始化；第 1 轮最低预算为 1；第 2 轮最低预算为 0。位置采样不反向传播，但 Cross-Attention、边分数和 Patch 价值在同一训练步骤使用几何标签更新。

- [ ] **步骤 4：运行测试并确认通过**

运行：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest PSMA.tests.test_pasm_v0_pipeline -v
```

预期：三轮均保持固定总点数，最终轮允许低价值 Patch 为 0，损失对有效几何行可计算。

### 任务 4：接入冻结骨干、提供命令行并编写中文说明

**文件：**

- 新建：`PSMA/pasm_v0/backbones.py`
- 新建：`PSMA/pasm_v0/infer.py`
- 新建：`PSMA/pasm_v0/train.py`
- 新建：`PSMA/pasm_v0/README.md`
- 新建：`PSMA/tests/test_pasm_v0_cli.py`

**接口：**

- `FrozenFeatureExtractor(dino_weights, loma_weights, device)` 读取两张归一化 RGB 图，返回聚合至 `16×16` 的 DINO token 和 LoMa 描述子图。
- `infer.py` 接受 `--image-a`、`--image-b`、`--dino-weights`、`--loma-weights`、`--output`、`--device`。
- `train.py` 接受 MegaDepth manifest、两个权重路径、输出检查点、轮数、学习率和设备。

- [ ] **步骤 1：写失败测试**

```python
def test_inference_command_exposes_required_arguments(self):
    script = Path(__file__).parents[1] / "pasm_v0" / "infer.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--image-a", result.stdout)
    self.assertIn("PASM v0", result.stdout)
```

- [ ] **步骤 2：运行测试并确认失败**

运行：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest PSMA.tests.test_pasm_v0_cli -v
```

预期：因推理命令尚不存在而失败。

- [ ] **步骤 3：实现骨干、命令行和说明**

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="运行 PASM v0 的三轮点级联合重采样")
    parser.add_argument("--image-a", type=Path, required=True)
    parser.add_argument("--image-b", type=Path, required=True)
    # 输出 npz：最终坐标、置信度、每轮预算与采样点历史。
```

`README.md` 使用中文说明安装环境、两条权重路径、推理命令、训练命令、输出字段，以及 v0 尚未支持的功能。

- [ ] **步骤 4：运行测试并确认通过**

运行：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest PSMA.tests.test_pasm_v0_cli -v
```

预期：`--help` 无需加载权重即可成功显示中文命令说明。

### 任务 5：回归验证

**文件：**

- 验证：`PSMA/tests/test_pasm_v0_*.py`
- 验证：`PSMA/tests/*.py`

- [ ] **步骤 1：运行 v0 定向测试**

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m unittest discover -s PSMA\tests -p 'test_pasm_v0_*.py' -v
```

预期：所有新测试通过。

- [ ] **步骤 2：运行现有完整回归**

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'D:\anaconda3\envs\loma\python.exe' -m unittest discover -s PSMA\tests -v
```

预期：既有几何工具和新 v0 测试全部通过。

## 自检结果

- 规格覆盖：任务 1 覆盖固定预算与轮次下限；任务 2 覆盖双图语义和全局点级边；任务 3 覆盖预算迁移、Patch 内重采样和单阶段几何损失；任务 4 覆盖真实权重接入、训练/推理入口与中文文档。
- 占位内容：无；所有文件、函数名、测试命令和行为均已确定。
- 接口一致性：`PasmV0Config` 和 `PatchLayout` 由采样、流程、训练目标共享；`PairSemanticEncoder` 的语义输出同时供匹配与 Patch 价值使用。
