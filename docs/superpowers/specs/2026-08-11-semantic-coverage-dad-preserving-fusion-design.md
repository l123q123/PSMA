# Semantic-Coverage and DaD-Preserving Fusion

## 目标

保留第一版“Patch 语义特征 + DaD 像素特征直接融合、端到端输出最终候选分数”的方向，但明确两类特征的职责：

- 双图语义特征负责区域级覆盖先验，避免共同可见区域被完全漏掉；
- DaD 的稠密像素特征和原始分数负责 Patch 内具体点的稳定性、重复性和排序；
- 融合头学习二者的协调方式，不引入 Patch 配额、每 Patch 上限、推理阈值或静态预算。

最终推理仍固定使用 DaD 候选池和全局 Top-K 选点。

## 结构

只对冻结 DaD NMS 候选进行重打分，避免融合头在非 DaD 候选位置制造新的局部峰。

对候选点 `i`，取其 DaD 32 维特征 `D_i`、原始 DaD logit `b_i`，以及所属 Patch 的双图语义 token `S_p`。语义 token 经过小型调制网络生成 `gamma_p` 和 `beta_p`，再以 FiLM 形式调制 DaD 特征：

```text
gamma_p, beta_p = MLP(S_p)
D'_i = gamma_p * D_i + beta_p
s_i = b_i + MLP([D'_i, b_i])
```

其中 `s_i` 是最终用于全局 Top-K 的候选分数。该结构允许语义信息改变候选的区域级竞争力，但保留 DaD 对具体像素点的辨别能力。

## 训练监督

### 几何候选损失

保留当前候选级软教师监督：根据深度、位姿、可见投影以及目标图 DaD 候选邻近性得到候选软标签 `q_i`，使用原有点级 BCE 和正负候选排序损失：

```text
L_geo = L_point + lambda_rank * L_rank
```

它负责匹配质量和几何可靠性，不负责点集空间分散。

### 语义覆盖损失

使用几何教师在训练阶段生成 Patch 支持权重 `w_p`，但语义 token 是覆盖预测的输入，不重新训练语义头。对最终候选分数构造可微的 Soft Top-K 选择权重 `a_i`，并计算每个 Patch 至少被一个候选代表的软占用：

```text
o_p = 1 - product_{i in p}(1 - a_i)
L_cover = - sum_p w_p * log(o_p + eps) / (sum_p w_p + eps)
```

该项只要求可靠共同可见 Patch 不被完全漏掉；同一 Patch 继续增加候选不会线性获得额外奖励，因此不会形成 Patch 配额。

### DaD 局部保真损失

在语义可靠且有候选支持的 Patch 内，约束融合后候选的相对分布不要偏离原始 DaD：

```text
pi_dad(i|p) = softmax(b_i / T)
pi_fused(i|p) = softmax(s_i / T)
L_local = sum_p rho_p * KL(pi_dad(.|p) || pi_fused(.|p))
```

这允许融合头整体抬高或压低一个 Patch，但不应无依据地打乱 Patch 内 DaD 的角点偏好。

### 总损失

```text
L_total = L_geo + lambda_cover * L_cover + lambda_local * L_local
```

所有 lambda 仅用于训练，不出现在推理逻辑中；第一版实验使用固定的小权重，先验证方向，再考虑自动损失平衡。

## 推理流程

```text
DaD dense features/logits
        + frozen semantic Patch tokens
        -> conditional fusion head
        -> candidate scores s_i
        -> existing global Top-K selection
        -> matching and pose estimation
```

不添加 Patch 预算、Patch 上限、人工阈值、点间距离半径或全局均匀化步骤。

## 非目标

- 不重新训练双图语义编码器；
- 不用融合头重新学习几何可见性定义；
- 不要求所有语义 Patch 都有点；没有可靠几何候选的 Patch 应被忽略；
- 不用覆盖损失替代 DaD 的局部点质量判断。

## 验收标准

在相同候选池、Top-K 和匹配流程下，至少检查：

1. 高语义支持但原版漏选的区域，Patch 被代表的比例提高；
2. Patch 内点的 DaD 排序相关性不明显下降；
3. 局部堆点减少，但不是通过全局均匀撒点实现；
4. 九组相对位姿成功率、中位姿态误差和 AUC 不劣于 `fusion-16-main6px` 基线。
