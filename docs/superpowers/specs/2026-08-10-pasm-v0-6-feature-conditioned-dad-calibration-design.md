# PASM v0.6：特征条件化 DaD 分数校准设计

## 目标

PASM v0.5 已证明双图语义可以提升精度，但当前无约束残差融合会把粗粒度语义偏好放大为区域级否决：在 held-out 教堂对中，static-16 模型将下方拱门区域的 344 个 DaD global 候选压至 0 个最终点。v0.6 的目标是在不改变 DaD、DINO、双图语义编码器、LoMa、3×3 NMS 与全局 Top-1024 的前提下，让高维双图语义特征调制 DaD 的像素级特征，并显式保留“DaD 原始强点应得到更大提升”的单调关系。

推理不得读取深度、位姿或其他几何信息；几何信息仅用于训练教师与 held-out 指标。

## 已知依据

在相同 reflect-padding、九组 held-out 协议下：

| 模型 | mean matches | mean 3px inlier | mean 6px inlier | mean 3px coverage | mean 6px coverage |
|---|---:|---:|---:|---:|---:|
| v0.5 static-16 | 722.8 | 87.44% | 94.21% | 73.63% | 87.34% |
| v0.5 main_6px-16 | 683.0 | 89.35% | 95.15% | 72.06% | 85.02% |

`main_6px` 使精度提高，但点数和覆盖略低。教堂第 2 对中，static-16 在拱门主体（`y >= 320`）选取 0 点；`main_6px` 恢复为 37 点。这表明当前 static 语义头的区域分布过于集中，但也表明仅更换语义头不能同时解决精度与覆盖问题。

## 非目标

- 不恢复 Patch 预算、Patch 容量或全局均匀撒点。
- 不训练或微调 DaD、DINO、双图语义编码器。
- 不在第一版同时改变候选池、Top-K 损失或 LoMa；这些作为后续单独消融。
- 不把双图语义特征提前压缩为共同可见性单概率图。

## 架构

设 `S` 为冻结双图语义头输出的 `[B, N, 128]` token，`D` 为冻结 DaD 的 32 通道 dense feature，`z` 为冻结 DaD 的单通道 dense logit。`N` 可为 `16×16` 或 `37×37`，实现必须从 token 数动态恢复方形网格。

### 语义场与 FiLM 调制

1. `S -> LayerNorm -> Linear(128, 64)`，reshape 为 Patch 网格并双线性上采样，得到像素级语义场 `S'`，形状 `[B, 64, H, W]`。
2. 从 `S'` 用 1×1 卷积产生 `gamma`、`beta`，均为 `[B, 32, H, W]`。
3. 将 `D` 双线性上采样到 `[B, 32, H, W]`，然后计算：

   ```text
   D' = (1 + 0.5 * tanh(gamma)) * D + beta
   ```

语义特征因此逐位置、逐 DaD 通道地调制局部点特征，而非只输出一个 Patch 标量。

### 受限的双分支分数校准

将 `[D', S', z]` 输入两个 3×3 局部卷积层（GroupNorm、GELU），输出 `boost` 和 `veto` 两张单通道图。它们是高维融合后的最终控制量，不是语义头本身的概率。

从 DaD 原始 logit 构造严格单调的图内质量因子：

```text
q = sigmoid((z - mean(z)) / (T * std(z)))
```

`T` 为可学习正温度。最终 logit 为：

```text
z_final = z
        + alpha_pos * q * sigmoid(boost)
        - alpha_neg * sigmoid(veto)
```

其中 `alpha_pos = 1.5 * sigmoid(theta_pos)`、`alpha_neg = 1.5 * sigmoid(theta_neg)`。上限防止模型再次对整片区域施加无界压制。

此结构保证：当融合特征判定区域适合匹配时，DaD 高分点因 `q` 更大而获得更大的正增益；DaD 弱点增益有限；当融合特征判定区域不适合时，`veto` 可对全部候选施加抑制。

## 训练

冻结 DaD、DINO 与语义编码器。仅训练语义投影、FiLM 参数头、局部融合层、`boost/veto` 输出头、`T` 与两个 `alpha` 参数。

第一版沿用 v0.5 训练候选域：每侧取 DaD 前 4096 个 NMS 候选；以几何投影、深度可见性、对侧最近 DaD 候选距离构造软标签。训练损失：

```text
L = L_point_BCE + ranking_weight * L_rank + lambda_delta * mean(abs(z_final - z))
```

`L_point_BCE` 与当前 v0.5 相同。`L_rank` 初版保留，尽管必须记录其是否再次过早归零。`L_delta` 为小权重最小改动正则，迫使模型在教师没有证据时保留 DaD 原始排序。

初版不引入显式 Patch 可见性 BCE；几何与跨图候选关系只通过现有点教师监督最终校准结果，避免把 128 维双图语义提前坍缩为标量掩码。

## 推理

```text
双图 RGB
-> 冻结 DaD 得到 D, z
-> 冻结 DINO 与双图语义头得到 S
-> FiLM 调制 D 为 D'
-> boost/veto 与受限校准得到 z_final
-> 全图 softmax、3×3 NMS、全局 Top-1024
-> 冻结 LoMa 匹配
```

推理路径不读取任何几何文件，且不包含 Patch 配额。

## 可视化与诊断

每一 held-out 对除已有输出外，保存：

- `semantic_feature_norm_{a,b}.png`：`S'` 的通道范数；
- `dad_quality_{a,b}.png`：`q`；
- `boost_{a,b}.png`、`veto_{a,b}.png`；
- `calibration_delta_{a,b}.png`：`z_final-z`；
- 原有 DaD / final score 图、选点图、匹配图。

汇总统计必须包含：高/中/低 `q` 分位的平均正增益、平均 veto、DaD Top-1024 与最终 Top-1024 的保留率、每个 Patch 的 DaD 点数与最终点数。诊断的目标是验证高 DaD 质量是否确实对应更高正校准，而不是再次只观察伪彩热图。

## 验证与消融

所有结果使用 reflect padding、固定九组 held-out、相同 DaD/LoMa 权重和相同几何评测协议。核心矩阵为：

| 语义 checkpoint | 融合器 |
|---|---|
| static-16 | v0.5 |
| main_6px-16 | v0.5 |
| static-16 | v0.6 |
| main_6px-16 | v0.6 |

每个 v0.6 模型训练 20 轮，并保存 e10、e15、e20。比较平均/全局 3px、6px 内点率，coverage、mutual matches，以及教堂对与其他典型对的点分布。

成功的最低标准是：在不以全局均匀撒点为代价的前提下，改善或保持 v0.5 的精度，并消除共同可见且 DaD 高响应区域被系统性清零的现象。若 v0.6 覆盖提高而精度下降，下一步只消融 `veto` 上限与 `q` 温度；若精度提高但覆盖仍显著不足，下一步再独立扩展教师候选域。

## 实施范围

新增独立 v0.6 模块、训练入口、推理入口、评测入口与单元测试；不得修改 v0.5 checkpoint 格式或覆盖既有输出。实现前需给出单独的实施计划，覆盖张量形状、checkpoint 字段、CLI、诊断输出、单元测试及九组评测命令。
