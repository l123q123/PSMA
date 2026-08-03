# PASM v0.2：双图条件化密集质量采样

PASM v0.2 在全图 `128×128` 的双图条件化质量场上反复采样。Patch 只管理跨区域预算和语义；匹配、锚点保留和淘汰始终发生在点级，不构造 Patch 对，也不先产生固定 Top-K 特征点池。

## 当前流程

1. 两图缩放到 `518×518`，划分为 `16×16` 个管理 Patch；每 Patch 有 `8×8` 个局部质量格。
2. 冻结 DINOv2 的 Patch token 经共享 Self-Attention 与两层共享双向 Cross-Attention 得到双图 Patch 语义和 Patch 价值。
3. 冻结 LoMa 稠密特征在全部质量格中心读取，局部 Query 对另一图所有 256 个 Patch 语义做 Cross-Attention，得到联合局部描述子和每图 `128×128` 质量 logits。
4. 每 Patch 内对 64 个 logits 做 softmax；首轮每 Patch 无放回抽 4 个质量格，格内再均匀生成连续坐标。
5. 每轮所有 1024 点做全局 dual-softmax 点级匹配；高置信互相匹配边作为锚点原样保留，并占下轮 Patch 预算。
6. Patch 预算由 Patch 价值、点级边反馈和锚点支持分配。非锚点每轮重新从该 Patch 的完整质量分布无放回抽样，因此跨轮可重新抽到旧位置，但同轮不会重复同一质量格。
7. 三轮中的质量图对同一图像对保持不变；只有锚点、Patch 预算和随机抽样改变。

## 训练监督

训练总损失为：

```text
L_total = L_base + L_quality
L_base = 三轮的 (L_match + L_value) 平均
L_quality = BCEWithLogits(q_A, T_A) + BCEWithLogits(q_B, T_B)
```

- `L_match`：由 MegaDepth 深度、内参和位姿直接投影生成当前采样点的点级正对应。
- `L_value`：由固定投影网格的可见覆盖生成 Patch 价值目标。
- `L_quality`：离线软教师 `T` 由“几何可见性 × LoMa 真投影相似度相对最强错误位置的优势”生成。

DINOv2 与 LoMa 在训练中始终冻结。v0.2 暂不包含相对位姿损失、RANSAC、dustbin、拒绝头、连续偏移、候选池或 NMS。

## 环境与权重

```powershell
D:\anaconda3\envs\loma\python.exe
```

```text
DINOv2: C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth
LoMa:   C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth
```

以下命令需要在 v0.2 工作目录执行：

```powershell
Set-Location 'D:\研0新方向\论文调研\Loma实验\PASM-v0\.worktrees\v0.2-quality-sampling'
```

`main` 工作目录仍保留 v0.1；v0.1 与 v0.2 检查点不能混用。完成开发后若将 v0.2 合并到 `main`，同样使用仓库根目录下的命令即可。

## 1. 生成离线质量教师缓存

先以 MegaDepth 的正图像对生成一次缓存。这个步骤读取深度、相机位姿和 LoMa；训练和推理不再读取缓存以外的几何真值。

建议先只跑 4 对确认连通性：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.cache_quality_teacher `
  --manifest '..\..\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --output '.\outputs\quality_cache_4pairs' `
  --max-pairs 4 `
  --device cuda
```

完整缓存只需删除 `--max-pairs 4` 并使用新输出目录。每个正图像对对应一个 `<pair_id>.npz`，其中保存两张 `128×128` 软教师图及元数据。

## 2. 训练 v0.2

使用上一步同一批图像对的缓存训练。进度条会显示当前和平均 `total loss`，以及当前 `base loss`、`quality loss`。

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.train `
  --manifest '..\..\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --quality-cache '.\outputs\quality_cache_4pairs' `
  --output '.\outputs\pasm_v02_smoke\checkpoint.pt' `
  --epochs 1 `
  --max-pairs 4 `
  --device cuda
```

完整训练时，先为完整 manifest 建立缓存，再移除 `--max-pairs 4`。每轮保存：

- `checkpoint.last.pt`：最新模型、Adam 状态和完整历史，用于续训。
- `checkpoint.pt`：平均 total loss 最低的模型，用于推理。
- `checkpoint.history.json`：每轮 total、base、quality loss。

续训示例：

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.train `
  --manifest '..\..\LoMa-main\LoMa-main\datasets\megadepth_mini\manifests\demo_pairs.json' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --quality-cache '.\outputs\quality_cache_full' `
  --output '.\outputs\pasm_v02_full\checkpoint.pt' `
  --resume '.\outputs\pasm_v02_full\checkpoint.last.pt' `
  --epochs 20 `
  --device cuda
```

## 3. 推理与输出

```powershell
& 'D:\anaconda3\envs\loma\python.exe' -m pasm_v0.infer `
  --image-a '图像A路径.jpg' `
  --image-b '图像B路径.jpg' `
  --dino-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth' `
  --loma-weights 'C:\Users\HP\.cache\torch\hub\checkpoints\loma_B128.pth' `
  --checkpoint '.\outputs\pasm_v02_smoke\checkpoint.pt' `
  --output '.\outputs\pasm_v02_smoke\inference_001' `
  --device cuda
```

每次 `--output` 是独立实验文件夹，包含：

- `quality_a.png`、`quality_b.png`：两图条件化质量热图。
- `round_1_points.png`、`round_2_points.png`、`round_3_points.png`：三轮点分布。
- `matches.png`、`high_confidence_matches.png`：最终点级匹配连线图。
- `result.npz`：匹配点、置信度、三轮预算、锚点数、质量 logits 和每轮质量格 ID。

推理只使用两张 RGB 图像、DINO/LoMa 权重和 v0.2 检查点；不读取深度、相机位姿或质量教师缓存。
