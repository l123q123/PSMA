# PASM v0.1：直接几何投影与点级联合重采样

本项目实现当前确认的 PASM 第一版：每张图每轮固定 1024 个实际采样点，所有点全局匹配；Patch 只管理采样预算和 Patch 内重采样，不构造 Patch 对。

## 当前流程

1. 将图像缩放为 `518×518`，建立 `16×16` 管理 Patch 网格。
2. 第 0 轮每个 Patch 固定采样 4 点，合计 1024 点。
3. 使用冻结 DINOv2 的双向 Cross-Attention 得到双图条件化 Patch 语义。
4. 使用冻结 LoMa 稠密描述子和联合语义，对全部 `1024×1024` 点对计算 dual-softmax 边权。
5. 从全局互相最佳边中选出高置信锚点；锚点原样保留，并占用两侧所属 Patch 的预算。
6. 将点级边质量、Patch 联合语义价值和锚点支持共同用于下一轮预算。
7. 第 1 轮每 Patch 至少 1 点；第 2 轮允许低价值 Patch 为 0 点；每 Patch 最多 20 点。
8. 重采样始终在 Patch 内执行；除锚点外，其余名额混合均匀探索和锚点邻域采样。点级投影在 6px 内视为当前采样对齐，以减少随机采样造成的正样本稀疏。

## 强化移动采样

默认配置在不改变总点数、Patch 上限、轮数和匹配结构的前提下，强化从低价值区域向高价值区域的迁移：

- 仅保留置信度至少为 `0.05` 的锚点，减少中等置信点占用下一轮预算；
- 锚点和每 Patch 最低点数扣除后，剩余预算按 Patch 价值的平方（`budget_score_power=2.0`）分配；
- Patch 内局部重采样扰动比例由 `0.20` 增至 `0.35`，扩大围绕高质量点的探索范围。

新训练的检查点会保存上述配置。旧检查点因未保存 `budget_score_power`，推理时自动使用原先的线性分配（`1.0`），因此不会被本次更新悄然改变结果；要使用强化策略，请从旧权重续训或重新训练并使用新输出检查点。

若只想用旧权重验证强化采样本身，可在推理命令末尾加入 `--strengthened-sampling`。该开关仅覆盖锚点阈值、预算分数幂和扰动比例，不会修改检查点权重。

v0 暂不包含 Patch 对筛选、Top-K 候选、dustbin、拒绝头、动态语义教师和自定义双图局部描述子。

## 环境与权重

默认 Python 环境：

```powershell
D:\anaconda3\envs\loma\python.exe
```

当前可用权重：

```text
DINOv2：C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth
LoMa：C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth
```

## 训练

以下命令以 MegaDepth mini 的深度图、相机内参和位姿训练一层双向 Cross-Attention 和 Patch 价值头。当前采样点会直接投影到另一张图，并以双向最近邻和深度一致性生成正样本；冻结 DINOv2 与 LoMa 不参与训练。

默认实验配置为每图 1024 点、首轮每 Patch 4 点、点级投影对齐阈值 6px、每 Patch 上限 20 点，并启用强化移动采样。以下是当前主实验的 96 对、10 轮训练命令：

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.train `
  --manifest '..\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --output '.\outputs\pasm_v0_1024\main_6px_1024.pt' `
  --epochs 10 `
  --device cuda
```

首次建议追加 `--max-pairs 4` 做连通性验证。完整训练会重复提取冻结特征，因此较慢；当前 GPU 实测约需 30 分钟。

## 续训与检查点

每完成一轮，`输出.last.pt` 会覆盖保存最新模型、Adam 优化器状态和完整训练历史，用于精确续训；`输出.pt` 仅在该轮平均 loss 低于历史最佳值时更新，用于推理和保留最佳模型。

已完成 1 轮的旧格式检查点不含优化器状态，但可加载模型权重并从同名历史文件继续。以下命令从 `repair_validation.pt` 训练至总共 10 轮，第一轮恢复时会重新初始化 Adam：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.train `
  --manifest '..\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --output '.\outputs\pasm_v0_1024\main_6px_1024.pt' `
  --resume '.\outputs\pasm_v0_1024\repair_validation.pt' `
  --epochs 10 --device cuda
```

## 推理

训练得到检查点后，使用两张图输出最终匹配和三轮历史：

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.infer `
  --image-a '图像A路径.jpg' `
  --image-b '图像B路径.jpg' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --checkpoint '.\outputs\pasm_v0\checkpoint.pt' `
  --output '.\outputs\pasm_v0\实验名称' `
  --device cuda
```

## 软质量教师可视化验证

这个独立脚本不训练任何模型，也不加载 PASM 或 DINO 权重。它用 MegaDepth 的深度、内参和位姿筛掉不可见点，再以冻结 LoMa 描述子的“正确投影相似度相对最强错误位置的优势”作为软质量分数；其目的是先肉眼判断这种教师是否能选出更好的局部点。

固定设置为 `518×518` 输入、`16×16` 管理 Patch、每 Patch 4 点，因此随机基线与教师选点均为每图 1024 点。默认只处理清单中的前 10 个正图像对：

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0'
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.visualize_quality_teacher `
  --manifest '..\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --output '.\outputs\quality_teacher_10pairs' `
  --device cuda
```

输出根目录包含 `summary.csv`，每个图像对另有独立子目录：

- `random_points.png`、`teacher_points.png`：左右图点分布；教师图从黄色（低分）到红色（高分）。
- `random_matches.png`、`teacher_matches.png`：冻结 LoMa 的互为最近邻连线图，仅显示相似度最高的 100 条边以避免遮挡。
- `summary.csv`：`coverage_3px/6px` 为 A→B、B→A 两方向的可见投影覆盖率平均值；`mutual_matches` 为互为最近邻数量；`inlier_3px/6px` 为全部 A→B 互为最近邻边中满足几何投影阈值的比例。

这里的教师分数不是“位姿价值”的人工真值，只表示该位置在当前几何下可见，且 LoMa 描述子更容易与真实投影区分开；是否适合作为 v0.2 局部质量头的监督，要结合这 10 对的图片和 CSV 对比后再决定。

## 输出字段

每次 `--output` 对应一个独立实验文件夹，固定包含：

- `result.npz`：数值结果；
- `round_1_points.png`、`round_2_points.png`、`round_3_points.png`：三轮采样点分布；
- `matches.png`：最终互相匹配连线图。
- `high_confidence_matches.png`：置信度不低于 `0.6` 的最终匹配连线图。

其中 `result.npz` 含以下关键字段：

- `keypoints_a`、`keypoints_b`：最终 mutual matching 的原图缩放坐标；
- `confidence`：对应边的 dual-softmax 置信度；
- `patch_ids_a`、`patch_ids_b`：每个最终点的管理 Patch；
- `round_budget_a`、`round_budget_b`：三轮的每 Patch 预算；
- `round_points_a`、`round_points_b`：三轮的全部采样点坐标；
- `round_anchor_count_a`、`round_anchor_count_b`：每轮每个 Patch 被原样保留的锚点数。
