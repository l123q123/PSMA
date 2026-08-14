# PASM v0.1：直接几何投影与点级联合重采样

本项目实现当前确认的 PASM 第一版：每张图每轮固定 1024 个实际采样点，所有点全局匹配；Patch 只管理采样预算和 Patch 内重采样，不构造 Patch 对。

## v0.3：静态 Patch 预算与 DaD 局部选点

v0.3 取消三轮锚点、反馈和重采样，改为一次静态推理：PASM 双图语义价值头只给 `16×16` 管理 Patch 分配 `0～20` 点的预算；冻结 DaD 在各自 Patch 内按原生 `3×3` NMS 分数选择具体坐标；最后把两图各 `1024` 个点交给完整冻结 LoMa-B128 做描述和匹配。

两个模式使用同一组图像、同一个 PASM v0.2 检查点、同一个 DaD 和完整 LoMa 匹配器：

- `dad_global`：DaD 在整图全局选 1024 点，不使用 PASM Patch 预算，是单图检测基线。
- `pasm_dad`：PASM 根据两图联合语义分配 Patch 预算，再由 DaD 在对应 Patch 内选点，是本版本的联合预算方法。

这两个模式都只运行一次选点和一次 LoMa 匹配，因而可以直接比较“联合 Patch 预算”本身的作用。v0.3 不需要重新训练，复用已有的 PASM 检查点；完整 LoMa-B128 使用 `dedode_b` 的 128 维局部描述子和 256 维匹配 Transformer，并依赖本地 DINOv2 权重。

## v0.4：DINO 原始 37×37 Patch 的监督消融

v0.4 保持当前一层双向 Cross-Attention 编码器不变，只把管理网格改为 DINOv2 的原始 `37×37` token 网格。每个 Patch 约对应 `14×14` 像素；每图总预算仍为 `1024`，每个 Patch 预算为 `0～4`。选点逻辑复用 v0.3：先由双图 Patch 价值分配预算，再由冻结 DaD 在每个 Patch 内按预算选点，最后由冻结完整 LoMa 匹配。

这次只比较 Patch 价值监督真值：

- `geometry`：`stride=2` 的深度和位姿几何共同可见性；
- `quality`：`stride=2` 的几何可见性与 LoMa 正负描述子差值形成的质量教师。

训练时不会调用完整 LoMa Transformer 反向传播，而是在 DaD 实际选出的点上读取冻结 LoMa 稠密描述子，用现有可微的局部描述子加双图语义边权计算 `L_match`；价值头使用对应缓存计算 `L_value`。两组训练除 `target_kind` 外完全一致。

### 生成几何 Patch 目标缓存

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0\.worktrees\v0.3-pasm-dad-static'
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.cache_v04_targets `
  --manifest 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --output '.\outputs\v0.4_ablation\targets_geometry_stride2' `
  --target-kind geometry `
  --device cuda
```

### 生成 LoMa 质量 Patch 目标缓存

这一步是离线计算，使用每隔 2 个像素的密集网格，耗时会明显高于 geometry 缓存，但只需生成一次。

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0\.worktrees\v0.3-pasm-dad-static'
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.cache_v04_targets `
  --manifest 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --output '.\outputs\v0.4_ablation\targets_quality_stride2' `
  --target-kind quality `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --device cuda
```

### 训练两组 v0.4 模型

先各运行 `--epochs 1` 检查连通性，确认无误后把轮数改为 `20`。

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0\.worktrees\v0.3-pasm-dad-static'
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.train_v04 `
  --manifest 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --dad-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dad.pth' `
  --target-cache '.\outputs\v0.4_ablation\targets_geometry_stride2' `
  --target-kind geometry `
  --output '.\outputs\v0.4_ablation\geometry_20.pt' `
  --epochs 20 `
  --device cuda `
  --seed 0
```

质量教师模型只需将上面命令中的 `target-cache`、`target-kind` 和 `output` 改为：

```text
--target-cache .\outputs\v0.4_ablation\targets_quality_stride2
--target-kind quality
--output .\outputs\v0.4_ablation\quality_20.pt
```

若继续训练到 60 轮，使用 `.last.pt` 而不是最佳 checkpoint：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.train_v04 `
  --manifest 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --dad-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dad.pth' `
  --target-cache '.\outputs\v0.4_ablation\targets_geometry_stride2' `
  --target-kind geometry `
  --output '.\outputs\v0.4_ablation\geometry_20.pt' `
  --resume '.\outputs\v0.4_ablation\geometry_20.last.pt' `
  --epochs 60 `
  --device cuda `
  --seed 0
```

质量教师续训时只需同步替换 `target-cache`、`target-kind`、`output` 和 `resume` 路径。

### 统一评测 A/B

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.evaluate_v04 `
  --image-root 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\extracted\0285\0' `
  --training-manifest 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --dad-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dad.pth' `
  --checkpoint-geometry '.\outputs\v0.4_ablation\geometry_20.pt' `
  --checkpoint-quality '.\outputs\v0.4_ablation\quality_20.pt' `
  --output '.\outputs\v0.4_ablation\heldout_eval' `
  --max-pairs 9 `
  --device cuda
```

评测输出包括 `summary.csv`、`aggregate.csv`、两种监督各自的 Patch 预算热图、DaD 选点图和 LoMa 匹配图。

### v0.3 推理命令

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0\.worktrees\v0.3-pasm-dad-static'
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.infer `
  --image-a 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\extracted\0285\0\3297850508_67cf0aae1c_b.jpg.jpg' `
  --image-b 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\extracted\0285\0\3390973770_759850a96e_o.jpg.jpg' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --dad-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dad.pth' `
  --checkpoint 'D:\研0新方向\论文调研\Loma实验\PASM-v0\outputs\pasm_v0_1024\main_6px_1024_60.pt' `
  --output '.\outputs\v0.3\pasm_dad_0285' `
  --device cuda `
  --mode pasm_dad
```

把最后的 `--mode pasm_dad` 改成 `--mode dad_global` 即可运行单图 DaD 基线。每次实验目录包含 `result.npz`、`points.png`、`matches.png` 和 `high_confidence_matches.png`。`result.npz` 中的 `selected_points_a/b` 必须各有 1024 个点；`pasm_dad` 还保存 `patch_budget_a/b` 与 `selected_patch_ids_a/b`，可直接验证每个 Patch 的实际点数。

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

## v0.6 候选抑制融合头

`candidate_suppression` 不再让融合头生成一张新的稠密响应图。它先固定原始 DaD 的 3×3 NMS 候选池，再在每个候选处融合 DaD 的 32 维特征、原始 DaD logit 和双图语义 token；融合头只输出非负抑制量，因此最终候选分数只能低于或等于原始 DaD 分数。训练中，几何可见且在另一图邻近 DaD 候选的点作为正样本保留；可见但没有邻近候选的点作为负样本压低；无法可靠投影的点不参与损失。推理时固定排除图像四周 `8px` 边界带，避免边界伪响应进入最终 Top-1024。

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0\.worktrees\v0.3-pasm-dad-static'
$py = 'D:\anaconda3\envs\loma\python.exe'
$manifest = 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json'
$dino = 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth'
$dad = 'C:\Users\HP\.cache\torch\hub\checkpoints\dad.pth'
$semantic = 'D:\研0新方向\论文调研\Loma实验\PASM-v0\outputs\pasm_v0_1024\main_6px_1024_60.pt'

& $py -m pasm_v0.train_v05 `
  --manifest $manifest `
  --dino-weights $dino `
  --dad-weights $dad `
  --semantic-checkpoint $semantic `
  --output 'outputs\v0.6_ablation\candidate_suppression_16_main6.pt' `
  --epochs 20 `
  --learning-rate 1e-4 `
  --patch-grid 16 `
  --fusion-mode candidate_suppression `
  --teacher-candidate-count 4096 `
  --teacher-sigma-px 3.0 `
  --teacher-search-radius-px 12.0 `
  --suppression-weight 1.0 `
  --suppression-margin 1.0 `
  --ranking-weight 0.5 `
  --ranking-mode soft_pairwise `
  --border-margin-px 8 `
  --device cuda
```

完成训练后，固定九组评测：

```powershell
$loma = 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth'
$images = 'D:\研0新方向\论文调研\Loma实验\LoMa-main\LoMa-main\datasets\megadepth_mini\extracted\0285\0'

& $py -m pasm_v0.evaluate_v05 `
  --image-root $images `
  --training-manifest $manifest `
  --dino-weights $dino `
  --loma-weights $loma `
  --dad-weights $dad `
  --fusion-checkpoint 'outputs\v0.6_ablation\candidate_suppression_16_main6.pt' `
  --semantic-checkpoint $semantic `
  --output 'outputs\v0.6_ablation\eval_candidate_suppression_16_main6' `
  --max-pairs 9 `
  --device cuda
```

### v0.6 Patch 门控 DaD

`patch_gate` 仅判断每个双图 Patch 是否应整体信任原始 DaD；它不再生成稠密响应，也不改变同一 Patch 内的 DaD 排序。训练时，对每个 Patch 取其可靠可见 DaD 候选几何教师的最大值；推理时低门控 Patch 的所有候选统一降分，再从固定候选池全局 Top-1024。

```powershell
& $py -m pasm_v0.train_v05 `
  --manifest $manifest `
  --dino-weights $dino `
  --dad-weights $dad `
  --semantic-checkpoint $semantic `
  --output 'outputs\v0.6_ablation\patch_gate_16_main6.pt' `
  --epochs 20 `
  --learning-rate 1e-4 `
  --patch-grid 16 `
  --fusion-mode patch_gate `
  --teacher-candidate-count 4096 `
  --teacher-sigma-px 3.0 `
  --teacher-search-radius-px 12.0 `
  --gate-penalty-scale 3.0 `
  --gate-quality-gamma 2.0 `
  --border-margin-px 8 `
  --device cuda
```

### v0.6 语义覆盖、DaD 局部保真融合

`semantic_coverage` 在固定原始 DaD NMS 候选上直接融合 Patch 双图语义 token、DaD 32 通道像素特征和原始 DaD logit。它不使用 Patch 预算、每 Patch 点数上限或推理阈值：推理仍对融合后的固定候选做全局 Top-1024。

训练损失由四部分组成：原有候选几何 BCE 与排序损失保证跨图匹配质量；`coverage` 项让几何上有可靠支持的共同可见 Patch 不会在最终点集中完全漏选；`local preservation` 项保持可靠 Patch 内融合前后的 DaD 相对排序；`invisibility` 项只压制“含 DaD 候选但没有任何几何可见候选”的完整 Patch 所获得的 Soft Top-K 占用。混合可见边界 Patch 不受该项约束。因此语义主要影响区域间竞争，DaD 主要决定区域内选哪个具体像素点。

```powershell
& $py -m pasm_v0.train_v05 `
  --manifest $manifest `
  --dino-weights $dino `
  --dad-weights $dad `
  --semantic-checkpoint $semantic `
  --output 'outputs\v0.6_ablation\semantic_coverage_16_main6.pt' `
  --epochs 20 `
  --learning-rate 1e-4 `
  --patch-grid 16 `
  --fusion-mode semantic_coverage `
  --teacher-candidate-count 4096 `
  --teacher-sigma-px 3.0 `
  --teacher-search-radius-px 12.0 `
  --ranking-weight 0.5 `
  --ranking-mode soft_pairwise `
  --coverage-weight 1.0 `
  --local-preservation-weight 0.25 `
  --coverage-temperature 0.5 `
  --coverage-topk 1024 `
  --invisibility-weight 1.0 `
  --border-margin-px 8 `
  --device cuda
```

`--coverage-temperature` 与 `--coverage-topk` 只用于训练中构造可微的 Soft Top-K 代理；`--invisibility-weight` 也仅训练使用，且只惩罚完全没有几何可见候选的 Patch。它们都不会进入推理选点逻辑；推理始终使用固定候选池和全局 Top-1024。
