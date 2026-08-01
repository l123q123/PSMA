# PASM v0.1 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐项执行。步骤使用复选框追踪。

**Goal:** 以直接几何投影替换稀疏缓存点监督，并让高置信互相匹配的点对跨轮原样保留。

**Architecture:** 新增独立的 `geometry.py`，负责 MegaDepth 相机/深度读取、直接投影、双向点级标签和固定网格 Patch 支持目标。匹配模块从全局边权中选择受双侧 Patch 容量约束的锚点；采样模块以锚点作为预算下限并原样保留其坐标，主流程仅负责连接这些纯函数。

**Tech Stack:** Python 3.12、NumPy、PyTorch、OpenEXR 3.4.13、Pillow、`unittest`。

## 全局约束

- 仅修改 `D:\研0新方向\论文调研\Loma实验\PASM-v0`。
- 所有新增文档、注释、类和函数 docstring 使用中文。
- DINOv2 与 LoMa 描述子始终冻结；只训练 `PairSemanticEncoder`。
- 不构造 Patch 对；仍对当前全部采样点做全局点级匹配。
- 不加入新的匹配 Transformer、dustbin、matchability 头或连续坐标优化。
- 训练期几何真值只生成损失标签，不能直接决定锚点；推理期不能读取深度和位姿。
- 项目当前没有 Git 仓库；不创建提交。
- 保持现有 `unittest` 风格，只添加覆盖新增核心行为的测试。

## 文件结构

- 新增 `pasm_v0/geometry.py`：相机/深度读取、坐标缩放、投影、点级双向标签和 Patch 投影支持。
- 修改 `pasm_v0/config.py`：加入锚点阈值和几何监督参数。
- 修改 `pasm_v0/data.py`：清单记录深度与相机文件，并构造 `PairGeometry`。
- 修改 `pasm_v0/matching.py`：从互相最佳边中选择满足两侧 Patch 上限的锚点。
- 修改 `pasm_v0/sampling.py`：支持锚点保留、锚点预算下限和锚点附近采样。
- 修改 `pasm_v0/pipeline.py`：将锚点选择、预算和保留式重采样串联，记录锚点统计。
- 修改 `pasm_v0/train.py`：使用投影生成的点级与 Patch 价值监督，不再读取缓存几何点。
- 修改 `pasm_v0/infer.py`：在结果中保存每轮锚点数量。
- 修改 `README.md`：更新 v0.1 的流程、训练真值来源和输出字段。
- 新增或修改 `tests/test_pasm_v0_geometry.py`、`tests/test_pasm_v0_sampling.py`、`tests/test_pasm_v0_pipeline.py`、`tests/test_pasm_v0_matching.py`。

---

### 任务 1：实现直接几何投影与 MegaDepth 几何读取

**Files:**

- Create: `pasm_v0/geometry.py`
- Modify: `pasm_v0/config.py`
- Modify: `pasm_v0/data.py`
- Test: `tests/test_pasm_v0_geometry.py`

**Interfaces:**

- 消费：`PairRecord` 中的 RGB、EXR 深度、Safetensors 相机路径和 `PasmV0Config`。
- 产出：

```python
@dataclass(frozen=True)
class PairGeometry:
    depth_a: np.ndarray
    depth_b: np.ndarray
    intrinsics_a: np.ndarray
    intrinsics_b: np.ndarray
    cam2world_a: np.ndarray
    cam2world_b: np.ndarray
    original_size_a: tuple[int, int]  # (width, height)
    original_size_b: tuple[int, int]  # (width, height)

def load_pair_geometry(pair: PairRecord, image_size: int) -> PairGeometry: ...

def projected_match_targets(
    points_a: np.ndarray,
    points_b: np.ndarray,
    geometry: PairGeometry,
    threshold_px: float,
    depth_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

def projected_patch_support_targets(
    geometry: PairGeometry,
    layout: PatchLayout,
    stride: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]: ...
```

- `projected_match_targets` 依次返回 `targets_a`、`valid_a`、`targets_b`、`valid_b`，目标索引为 `-1` 的位置不参与点级损失。

- [ ] **步骤 1：先写失败测试**

在 `tests/test_pasm_v0_geometry.py` 构造 `32×32`、恒定深度 `2.0`、单位内参、单位 `cam2world` 的 `PairGeometry`。覆盖以下三件事：

```python
def test_identity_geometry_labels_mutual_nearest_sampled_points(self) -> None:
    geometry = self._identity_geometry()
    points_a = np.array([[4.0, 4.0], [20.0, 20.0]], dtype=np.float32)
    points_b = np.array([[4.5, 4.0], [20.0, 20.5], [30.0, 30.0]], dtype=np.float32)
    targets_a, valid_a, targets_b, valid_b = projected_match_targets(
        points_a, points_b, geometry, threshold_px=1.0, depth_tolerance=0.01
    )
    torch.testing.assert_close(targets_a, torch.tensor([0, 1]))
    torch.testing.assert_close(valid_a, torch.tensor([True, True]))
    torch.testing.assert_close(targets_b, torch.tensor([0, 1, -1]))
    torch.testing.assert_close(valid_b, torch.tensor([True, True, False]))

def test_occluded_projection_does_not_create_positive_target(self) -> None:
    geometry = self._identity_geometry(depth_b=1.0)
    targets_a, valid_a, _, _ = projected_match_targets(
        np.array([[8.0, 8.0]], dtype=np.float32),
        np.array([[8.0, 8.0]], dtype=np.float32),
        geometry, threshold_px=1.0, depth_tolerance=0.01,
    )
    torch.testing.assert_close(targets_a, torch.tensor([-1]))
    torch.testing.assert_close(valid_a, torch.tensor([False]))

def test_projection_support_counts_visible_patch_coverage(self) -> None:
    support_a, support_b = projected_patch_support_targets(
        self._identity_geometry(), PatchLayout(self.config), stride=4, depth_tolerance=0.01
    )
    self.assertAlmostEqual(float(support_a.sum()), 1.0)
    self.assertAlmostEqual(float(support_b.sum()), 1.0)
```

再用临时 JSON 清单验证 `load_manifest` 能读取 `depth_A`、`depth_B`、`camera_A`、`camera_B`，并且正图像对缺少其中任一字段时抛出带字段名的 `ValueError`。

- [ ] **步骤 2：运行失败测试**

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_geometry -v
```

预期：因 `pasm_v0.geometry` 和投影函数尚不存在而失败。

- [ ] **步骤 3：实现最小几何模块和清单字段**

在 `PasmV0Config` 加入固定默认值：

```python
anchor_threshold: float = 0.02
geometry_match_threshold_px: float = 3.0
geometry_depth_tolerance: float = 0.01
geometry_support_stride: int = 4
```

同时在 `__post_init__` 验证 `anchor_threshold >= 0`、`geometry_match_threshold_px > 0`、`geometry_depth_tolerance >= 0`、`geometry_support_stride > 0`，并保持已有预算约束不变。

在 `PairRecord` 加入可空的 `depth_a`、`depth_b`、`camera_a`、`camera_b`。`load_manifest` 对 `label == 1` 的记录要求四个路径同时存在且为文件；负图像对允许这些字段为 `None`。

在 `geometry.py` 使用下列规则实现：

```python
target_from_source = np.linalg.inv(target_cam2world) @ source_cam2world
source_xyz = source_depth[:, None] * (np.linalg.inv(source_intrinsics) @ pixels.T).T
target_xyz = source_xyz @ target_from_source[:3, :3].T + target_from_source[:3, 3]
projected = (target_intrinsics @ target_xyz.T).T
target_xy = projected[:, :2] / projected[:, 2:3]
```

源点深度读取时，把模型坐标按 `original_size / image_size` 转到源深度图，使用最近像素；目标深度使用投影坐标的双线性读取。只有源深度有限且大于零、目标相机深度大于零、投影坐标在图内、目标深度有限大于零、并满足：

```python
abs(projected_depth - observed_target_depth) <= depth_tolerance * max(observed_target_depth, 1e-6)
```

的点才是可见投影。

对 A→B 和 B→A 分别投影；只保留两边都选择对方、且两个投影误差都不超过 `threshold_px` 的点对。最近邻使用平方欧氏距离的 NumPy 广播实现，避免引入新依赖。

`projected_patch_support_targets` 在模型输入坐标生成步长为 `stride` 的固定网格，分别做 A→B、B→A 可见投影，并按源点所在 Patch 计数、归一化。某方向无有效投影时返回全零数组。

`read_camera_metadata` 使用 8 字节小端头长度解析 Safetensors JSON，严格读取 `cam2world` 的 `F64[4,4]` 和 `intrinsics` 的 `F64[3,3]`；`read_exr_depth` 惰性导入 `OpenEXR`/`Imath`，优先读取 `Y` 通道、回退 `R` 通道。

- [ ] **步骤 4：运行几何测试**

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_geometry -v
```

预期：全部通过，单位相机产生互相对应，遮挡深度产生零正样本，固定网格产生归一化 Patch 支持。

---

### 任务 2：让训练损失完全使用投影监督

**Files:**

- Modify: `pasm_v0/train.py`
- Modify: `tests/test_pasm_v0_pipeline.py`
- Modify: `pasm_v0/data.py`

**Interfaces:**

- 消费：任务 1 的 `PairGeometry`、`projected_match_targets`、`projected_patch_support_targets`。
- 产出：`_result_loss(rounds, geometry, layout, config)`，仍返回一个标量 `torch.Tensor`。

- [ ] **步骤 1：改写失败测试**

将现有的 `test_geometry_targets_and_loss_use_only_reliable_sampled_pairs` 替换为直接投影版本：

```python
def test_projected_geometry_targets_drive_point_and_patch_losses(self) -> None:
    geometry = make_identity_geometry(32)
    targets_a, valid_a, _, _ = projected_match_targets(
        np.array([[2.0, 2.0], [20.0, 20.0]], dtype=np.float32),
        np.array([[2.0, 2.0], [20.0, 20.0]], dtype=np.float32),
        geometry, threshold_px=0.5, depth_tolerance=0.01,
    )
    support_a, _ = projected_patch_support_targets(
        geometry, PatchLayout(self.config), stride=4, depth_tolerance=0.01
    )
    loss = pasm_v0_loss(
        torch.full((2, 2), 0.5, requires_grad=True), targets_a, valid_a,
        torch.full((16,), 0.5, requires_grad=True), support_a,
    )
    self.assertGreater(float(loss.detach()), 0.0)
```

同时断言 `_result_loss` 不再接受 `geometry_a`、`geometry_b` 两个缓存数组。

- [ ] **步骤 2：运行失败测试**

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_pipeline -v
```

预期：旧 `_result_loss` 签名和缓存标签调用不满足新测试。

- [ ] **步骤 3：接入训练调用链**

在 `train_model` 中将训练集筛选改为：

```python
pairs = [pair for pair in load_manifest(manifest_path) if pair.label == 1]
```

每次处理正图像对时调用：

```python
geometry = load_pair_geometry(pair, config.image_size)
loss = _result_loss(result.rounds, geometry, layout, config)
```

将 `_result_loss` 改为：先调用一次 `projected_patch_support_targets`，再对每一轮调用一次 `projected_match_targets`，并像现有版本一样将 A→B、B→A 两个 `pasm_v0_loss` 加入 `losses` 后求平均。`pasm_v0_loss` 的计算公式不修改。

删除 `scaled_pair_geometry`、`point_match_targets`、`patch_support_targets` 在 `train.py` 中的导入和调用；保留旧函数不删除，避免破坏其他历史脚本。

将 CLI 描述改为“使用 MegaDepth 深度和位姿训练 PASM v0.1”。当正图像对缺少几何文件时，报错应直接指出图像对 ID 和缺少路径字段。

- [ ] **步骤 4：运行训练目标测试**

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_geometry tests.test_pasm_v0_pipeline -v
```

预期：所有投影目标和损失测试通过；现有三轮预算测试仍通过。

---

### 任务 3：实现锚点选择、保留预算和 Patch 内锚点采样

**Files:**

- Modify: `pasm_v0/matching.py`
- Modify: `pasm_v0/sampling.py`
- Modify: `tests/test_pasm_v0_matching.py`
- Modify: `tests/test_pasm_v0_sampling.py`

**Interfaces:**

- 产出：

```python
@dataclass(frozen=True)
class AnchorMatches:
    indices_a: np.ndarray
    indices_b: np.ndarray
    scores: np.ndarray

def select_anchor_matches(
    edge_weights: torch.Tensor,
    patch_ids_a: np.ndarray,
    patch_ids_b: np.ndarray,
    threshold: float,
    max_points_per_patch: int,
) -> AnchorMatches: ...

def allocate_patch_budget(
    values: np.ndarray,
    config: PasmV0Config,
    round_index: int,
    reserved: np.ndarray | None = None,
) -> np.ndarray: ...

def resample_in_patches(
    layout: PatchLayout,
    previous_points: np.ndarray,
    previous_patch_ids: np.ndarray,
    point_quality: np.ndarray,
    budgets: np.ndarray,
    anchor_points: np.ndarray,
    anchor_patch_ids: np.ndarray,
    anchor_scores: np.ndarray,
    config: PasmV0Config,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...
```

- [ ] **步骤 1：先写失败测试**

在匹配测试加入容量约束的锚点选择：

```python
def test_anchor_selection_keeps_mutual_edges_under_both_patch_caps(self) -> None:
    weights = torch.tensor([[0.90, 0.01], [0.02, 0.80]], dtype=torch.float32)
    anchors = select_anchor_matches(
        weights,
        np.array([0, 0], dtype=np.int64),
        np.array([1, 1], dtype=np.int64),
        threshold=0.1,
        max_points_per_patch=1,
    )
    np.testing.assert_array_equal(anchors.indices_a, np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(anchors.indices_b, np.array([0], dtype=np.int64))
```

在采样测试加入两个行为：

```python
def test_reserved_anchors_raise_patch_budget_lower_bound(self) -> None:
    reserved = np.zeros(16, dtype=np.int64)
    reserved[3] = 3
    budget = allocate_patch_budget(np.ones(16), self.config, 2, reserved=reserved)
    self.assertGreaterEqual(int(budget[3]), 3)

def test_resampling_retains_anchor_coordinates_exactly(self) -> None:
    anchor = np.array([[9.0, 9.0]], dtype=np.float32)
    sampled, sampled_ids = resample_in_patches(
        self.layout, previous_points, previous_patch_ids, quality, budget,
        anchor, np.array([5], dtype=np.int64), np.array([0.9], dtype=np.float32),
        self.config, seed=9,
    )
    self.assertTrue(np.any(np.all(sampled == anchor[0], axis=1)))
    self.assertEqual(int(sampled_ids[np.flatnonzero(np.all(sampled == anchor[0], axis=1))[0]]), 5)
```

- [ ] **步骤 2：运行失败测试**

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_matching tests.test_pasm_v0_sampling -v
```

预期：新增锚点接口不存在，且旧重采样签名不接受锚点输入。

- [ ] **步骤 3：实现锚点和保留式采样**

`select_anchor_matches` 先调用现有 `mutual_match_indices`，取出边权后按从高到低排序。按顺序扫描每条边，只有当 A 点所属 Patch 和 B 点所属 Patch 的已选锚点数都小于 `max_points_per_patch` 时才保留。返回排序后的 NumPy 索引和分数，空结果统一返回形状为 `(0,)` 的数组。

`allocate_patch_budget` 将 `reserved` 规范为长度等于 Patch 数的非负整数数组；其起始预算为：

```python
minimum = config.minimum_for_round(round_index)
budget = np.maximum(np.full(config.patch_count, minimum, dtype=np.int64), reserved)
```

然后沿用现有的容量限制最大余数法分配剩余预算。若 `reserved` 超过每块上限或总和超过总预算，抛出清晰的 `ValueError`。

`resample_in_patches` 对每个 Patch 先直接写入该 Patch 的锚点坐标，再处理剩余名额：

```python
remaining = count - len(patch_anchors)
explore_count = ceil(remaining * config.exploration_ratio)
local_count = remaining - explore_count
```

`explore_count` 个点在 Patch 内均匀采样。存在锚点时，`local_count` 按锚点边权为概率选择中心并添加现有 `jitter_ratio` 扰动；没有锚点时，沿用旧点质量作为中心概率；旧点也为空时，全部剩余名额均匀采样。所有扰动坐标继续裁剪在 Patch 内。

- [ ] **步骤 4：运行锚点与采样测试**

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_matching tests.test_pasm_v0_sampling -v
```

预期：锚点只在双侧容量允许时被选中，锚点 Patch 预算不低于其保留数，锚点原始坐标出现在下一轮点集中，所有点仍归属各自 Patch。

---

### 任务 4：接入主流程、记录锚点统计并更新说明

**Files:**

- Modify: `pasm_v0/pipeline.py`
- Modify: `pasm_v0/infer.py`
- Modify: `README.md`
- Modify: `tests/test_pasm_v0_pipeline.py`
- Modify: `tests/test_pasm_v0_cli.py`

**Interfaces:**

- `RoundRecord` 新增 `anchor_indices_a`、`anchor_indices_b`、`anchor_scores`，它们描述从本轮点集保留到下一轮的锚点边。
- `result.npz` 新增 `round_anchor_count_a`、`round_anchor_count_b`，形状均为 `[iterations, patch_count]`。

- [ ] **步骤 1：先写失败测试**

在流程测试中构造 `anchor_threshold=0.0` 的小配置，并验证任一记录的锚点会完整出现在下一轮：

```python
for round_index, record in enumerate(result.rounds[:-1]):
    next_record = result.rounds[round_index + 1]
    for index in record.anchor_indices_a:
        self.assertTrue(np.any(np.all(next_record.points_a == record.points_a[index], axis=1)))
    for index in record.anchor_indices_b:
        self.assertTrue(np.any(np.all(next_record.points_b == record.points_b[index], axis=1)))
```

在 CLI 测试中，以小数组构造 `PasmV0Result` 或读取临时 `.npz`，断言 `round_anchor_count_a` 与 `round_anchor_count_b` 被保存。

- [ ] **步骤 2：运行失败测试**

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m unittest tests.test_pasm_v0_pipeline tests.test_pasm_v0_cli -v
```

预期：`RoundRecord` 没有锚点字段，推理输出没有锚点统计字段。

- [ ] **步骤 3：连接锚点、预算和重采样**

在每轮得到 `edge_weights` 后、进入下一轮前执行：

```python
anchors = select_anchor_matches(
    edge_weights, patch_ids_a, patch_ids_b,
    self.config.anchor_threshold, self.config.max_points_per_patch,
)
anchor_counts_a = np.bincount(patch_ids_a[anchors.indices_a], minlength=self.config.patch_count)
anchor_counts_b = np.bincount(patch_ids_b[anchors.indices_b], minlength=self.config.patch_count)
anchor_support_a = anchor_counts_a / max(int(anchor_counts_a.max()), 1)
anchor_support_b = anchor_counts_b / max(int(anchor_counts_b.max()), 1)
```

将下一轮预算分数改为：

```python
scores_a = value_a[0].detach().cpu().numpy() + feedback_a.detach().cpu().numpy() + anchor_support_a
scores_b = value_b[0].detach().cpu().numpy() + feedback_b.detach().cpu().numpy() + anchor_support_b
budget_a = allocate_patch_budget(scores_a, self.config, round_index + 1, reserved=anchor_counts_a)
budget_b = allocate_patch_budget(scores_b, self.config, round_index + 1, reserved=anchor_counts_b)
```

随后将 `points_a[anchors.indices_a]`、`patch_ids_a[anchors.indices_a]`、`anchors.scores` 传入 A 图重采样；B 图使用对应的 B 索引。最后一轮不选择下一轮锚点，保存形状为 `(0,)` 的空锚点字段。

`infer.py` 不改变最终匹配定义，只在 `.npz` 中新增固定形状的锚点计数：

```python
round_anchor_count_a=np.stack([
    np.bincount(record.patch_ids_a[record.anchor_indices_a], minlength=runner.config.patch_count)
    for record in result.rounds
]),
```

B 图同理。README 更新为：训练通过深度与位姿直接投影生成标签；重采样保留高置信锚点；输出文件包含逐轮锚点计数。

- [ ] **步骤 4：运行完整测试和最小真实连通性验证**

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m unittest discover -s tests -v
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.train `
  --manifest '..\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --output '.\outputs\pasm_v0_smoke\v0_1.pt' `
  --epochs 1 `
  --max-pairs 1 `
  --device cuda
```

预期：全部单元测试通过；最小训练能读取 EXR/Safetensors、完成一次前向/反向传播并输出检查点与历史文件。完整训练轮数、采样轮数、总预算和 Patch 上限的对比实验不包含在本实现任务中。

## 计划自检

- 设计文档中的直接投影、双向一致性、目标深度检查、固定网格 Patch 支持、锚点容量约束、预算保留和三类重采样均有对应任务。
- 保留式采样只使用模型预测锚点，训练期没有向采样流程泄漏深度真值。
- 新接口的名称、参数和调用顺序在任务 1 至任务 4 保持一致。
- 未引入匹配器、Patch 对、候选点阶段或额外实验变量，符合 v0.1 的最小范围。
