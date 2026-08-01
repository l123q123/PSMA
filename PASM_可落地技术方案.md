# Pair-Adaptive Sparse Matching（PASM）可落地方案

## 1. 目标

设计一个面向稀疏图像匹配的端到端框架。与传统“单图检测关键点 → 单图提取描述子 → 两图匹配”不同，本方法从一开始就使用图像对的联合信息，并通过多轮自适应采样完成以下任务：

1. 判断当前图像对中哪些 DINO patch 值得分配匹配计算。
2. 在高价值 patch 内动态增加采样点，在低价值 patch 内减少采样点。
3. 保留高置信匹配，删除失败点，并在 patch 内重新采样。
4. 输出数量可变的像素级稀疏匹配。
5. 当有效 patch 和有效匹配持续衰减时，拒绝当前图像对。
6. 利用 VPR 教师和动态语义教师提供弱监督，但最终由两张图的联合信息修正教师先验。

暂定名称：

> **Pair-Adaptive Sparse Matching（PASM）**

---

## 2. 核心思想

传统方法的点是由单张图决定的，而 PASM 中的点是由当前图像对共同决定的。

完整流程：

```text
Image A + Image B
        ↓
DINO patch 特征
        ↓
双向 Cross-Attention
        ↓
Pair-conditioned Patch Value
        ↓
每个 patch 分配初始采样预算
        ↓
patch 内采样局部点
        ↓
pair-conditioned 局部匹配
        ↓
保留成功点 / 删除失败点
        ↓
根据匹配反馈重新分配各 patch 预算
        ↓
patch 内重采样
        ↓
迭代若干轮
        ↓
最终稀疏匹配 / 图像对拒绝
```

关键限制：

- 点只在所属 patch 内采样，不跨 patch 移动。
- 不采用“一个粗匹配只生成一个精确点”的设计。
- 一个 patch 最终可以输出 0、1 或多个像素级匹配。
- 区域选择、采样预算、局部匹配和拒绝共同优化。

---

# 3. 模型架构

## 3.1 输入

输入两张 RGB 图像：

\[
I_A \in \mathbb{R}^{H_A \times W_A \times 3}
\]

\[
I_B \in \mathbb{R}^{H_B \times W_B \times 3}
\]

建议第一版统一缩放到固定尺寸，例如：

```text
518 × 518
```

这样便于直接使用 DINOv2 的 14×14 patch。

---

## 3.2 双分支特征编码器

### 3.2.1 语义特征分支

使用冻结或轻量微调的 DINOv2 / DINOv3：

```text
I_A → DINO → F_A^sem
I_B → DINO → F_B^sem
```

例如输入为 518×518、patch size 为14时：

```text
F_A^sem, F_B^sem ∈ R^(37×37×C_sem)
```

语义分支负责：

- 地点结构识别；
- 大视角和光照变化鲁棒性；
- 判断哪些 patch 在当前图像对中可能共同存在；
- 为 patch 分配初始计算预算。

### 3.2.2 局部细节分支

使用轻量 CNN 或 FPN：

```text
I_A → Local Encoder → F_A^local
I_B → Local Encoder → F_B^local
```

建议输出 1/4 或 1/2 分辨率特征：

```text
F^local ∈ R^(H/4 × W/4 × C_local)
```

局部分支负责：

- 边缘；
- 角点；
- 局部纹理；
- 像素级匹配定位。

第一版可选：

- XFeat backbone；
- SuperPoint encoder；
- 轻量 ResNet-FPN；
- 自定义 MobileNet/FPN。

---

## 3.3 教师先验模块

教师不直接决定最终关注区域，只提供单图弱先验。

### 3.3.1 VPR 教师

可选：

- SALAD；
- BoQ；
- NetVLAD；
- Pair-VPR 中的单图 backbone。

输出单图地点辨识热力图：

\[
T_A^{place}, T_B^{place} \in [0,1]^{H_p \times W_p}
\]

作用：

- 提高建筑结构、固定招牌、门窗边缘等区域的初始权重；
- 降低天空、无纹理背景等无辨识区域的权重。

### 3.3.2 动态语义教师

使用语义分割模型得到动态类别概率：

\[
T_A^{dyn}, T_B^{dyn}
\]

动态类别可包括：

```text
person, car, bus, bicycle, motorcycle, animal
```

构造教师先验：

\[
T^{teacher}=T^{place}\cdot(1-T^{dyn})
\]

教师先验必须是软约束，不能作为硬 mask。

---

## 3.4 Pair-conditioned Patch Encoder

对两张图的 DINO patch token 进行双向交叉注意力：

```text
F_A^sem → Self-Attention → Cross-Attention(B)
F_B^sem → Self-Attention → Cross-Attention(A)
```

输出：

\[
\hat F_A^{sem}, \hat F_B^{sem}
\]

每个 patch token 同时包含：

- 自身语义；
- 另一张图中可能对应结构的信息；
- 当前图像对的全局上下文。

可额外加入全局 pair token：

\[
g_{AB}
\]

用于表示：

- 图像对整体相似性；
- 预计重叠程度；
- 场景是否可能一致。

---

## 3.5 Patch Pair Scoring

对 A 中 patch \(i\) 和 B 中 patch \(j\) 计算联合分数：

\[
S_{ij}^{patch}
=
f_{patch}
\left(
\hat f_i^A,
\hat f_j^B,
\hat f_i^A \odot \hat f_j^B,
|\hat f_i^A-\hat f_j^B|,
g_{AB},
T_i^A,
T_j^B
\right)
\]

输出：

1. patch 对应分数；
2. patch 可匹配性；
3. patch 初始采样预算；
4. patch 无匹配概率 dustbin。

第一版不要对所有 \(N_A \times N_B\) patch 对全部展开。

建议：

1. 用余弦相似度得到每个 A patch 的 B 侧 Top-R 候选；
2. 只对 Top-R 候选用 MLP/Transformer 精细评分；
3. 保留双向一致的 patch 对。

推荐：

```text
R = 3 或 5
```

---

## 3.6 Patch 采样预算头

对每个候选 patch 对 \(P_{ij}\) 预测采样预算：

\[
n_{ij}^{0}\in\{0,1,\dots,n_{max}\}
\]

第一轮建议只分配少量点：

```text
n_min = 1
n_max = 8
```

初始预算由以下信息决定：

- 当前 patch 对联合语义分数；
- 教师先验；
- patch 区分度；
- 全局重叠预测；
- 当前总预算约束。

全图设置固定最大预算：

```text
N_total_max = 512 或 1024
```

通过 softmax budget allocation 分配：

\[
n_{ij}
\propto
\operatorname{softmax}(V_{ij}/\tau)
\]

这样模型必须在有限预算下选择重点区域。

---

## 3.7 Patch 内点采样器

点不移动，只在 patch 内进行采样和重采样。

### 初始采样

每个 patch 内可采用：

- 均匀网格；
- 分层随机采样；
- 由局部显著性图进行概率采样。

推荐第一版：

```text
第一轮：均匀或 Sobol 采样
后续轮：根据局部采样概率图重采样
```

对于 patch \(i\)，维护一个局部概率图：

\[
P_i^t(x,y)
\]

由以下部分构成：

\[
P_i^t
\propto
P_i^{teacher}
\cdot
P_i^{pair}
\cdot
P_i^{local}
\cdot
P_i^{unexplored}
\]

含义：

- `teacher`：教师认为值得关注；
- `pair`：两图联合语义支持；
- `local`：局部边缘/纹理强度；
- `unexplored`：远离已保留成功点。

---

## 3.8 Pair-conditioned 局部特征

对采样点从局部特征图双线性采样：

\[
l_k^A=\operatorname{sample}(F_A^{local},p_k^A)
\]

\[
l_r^B=\operatorname{sample}(F_B^{local},p_r^B)
\]

再使用对应 patch 的联合语义特征进行条件化：

\[
\tilde l_k^A=f_{cond}(l_k^A,z_{ij}^{sem})
\]

\[
\tilde l_r^B=f_{cond}(l_r^B,z_{ij}^{sem})
\]

这里的局部特征不是传统固定单图描述子，而是：

> 当前图像对条件下的局部特征表示。

---

## 3.9 Patch 内局部匹配器

对 patch 对内部的采样点建立小型匹配矩阵：

\[
M_{kr}
=
f_{match}
\left[
\tilde l_k^A,
\tilde l_r^B,
\tilde l_k^A\odot\tilde l_r^B,
|\tilde l_k^A-\tilde l_r^B|,
z_{ij}^{sem}
\right]
\]

建议匹配头：

```text
MLP pair scorer
+ dual-softmax 或 Sinkhorn
+ dustbin
```

必须支持：

- 一对一匹配；
- 无匹配；
- 双向一致性；
- 多个点来自同一个 patch；
- 不同 patch 中点数不同。

匹配结果输出：

```text
point_A
point_B
match_confidence
uniqueness
mutual_score
```

---

## 3.10 点保留、删除与重采样

每轮结束后，对每个点对进行分类：

### 保留

满足：

```text
match_confidence > τ_keep
mutual consistency = true
uniqueness > τ_unique
```

保留点进入下一轮，不再重复采样其邻域。

### 删除

满足任一条件：

```text
confidence < τ_drop
进入 dustbin
多对一冲突
局部重复
```

删除后释放采样预算。

### 重采样

在同一 patch 内重新采样，优先选择：

- 尚未探索的位置；
- 局部高纹理区域；
- 教师与联合语义共同支持区域；
- 远离已保留点的位置。

---

## 3.11 动态预算更新

第 \(t\) 轮后，patch 对价值：

\[
V_{ij}^{t}
=
\alpha S_{ij}^{pair}
+
\beta R_{ij}^{good}
+
\gamma C_{ij}^{match}
-
\delta A_{ij}^{amb}
-
\eta U_{ij}^{waste}
\]

其中：

- \(S^{pair}\)：联合语义分数；
- \(R^{good}\)：有效匹配比例；
- \(C^{match}\)：高置信匹配质量；
- \(A^{amb}\)：匹配歧义；
- \(U^{waste}\)：连续多轮无效采样比例。

下一轮预算：

```text
高价值：增加采样
中等价值：保持采样
低价值：减少采样
连续低价值：预算归零
```

第一版可将预算更新设为三分类：

```text
减少 / 保持 / 增加
```

---

## 3.12 图像对拒绝头

拒绝机制不应只看最终点数，而是融合：

- 全局 pair token；
- 高价值 patch 数量；
- 有效匹配数量；
- 有效匹配衰减曲线；
- 平均置信度；
- 匹配空间覆盖；
- dustbin 比例；
- patch 预算衰减比例。

输入特征示例：

\[
z_{reject}=
[
g_{AB},
N_{match}^{t},
N_{patch}^{t},
r_{dustbin}^{t},
\Delta N_{match}^{t},
coverage,
mean\_confidence
]
\]

输出：

\[
p_{accept}
\]

拒绝条件不要只写死阈值。建议：

```text
Early soft rejection：
仅减少后续预算，不立即终止

Final hard rejection：
最终输出 no-match
```

---

# 4. 训练标签

## 4.1 图像对标签

三分类优于简单二分类：

```text
0：不同场景
1：同地点但低重叠/不可估计位姿
2：同地点且可可靠匹配
```

第一版可简化为：

```text
positive：有足够有效重叠
negative：无足够有效重叠
```

---

## 4.2 教师热力图标签

教师生成：

```text
place relevance heatmap
dynamic probability map
```

构造：

\[
T^{teacher}=T^{place}(1-T^{dynamic})
\]

学生 patch 注意力与教师进行软蒸馏：

\[
L_{teacher}
=
KL(S_A^{student}\|T_A)
+
KL(S_B^{student}\|T_B)
\]

训练后期降低权重。

---

## 4.3 Patch 对应标签

若训练集有深度、内参和相对位姿：

1. 将 A 中 patch 中心或采样像素反投影到3D；
2. 投影到 B；
3. 找到落入的目标 patch；
4. 做边界、深度和遮挡检查；
5. 生成 patch 对应标签。

标签支持：

```text
positive patch pair
negative patch pair
dustbin
ignore
```

若没有深度，可使用：

- SfM tracks；
- LoMa/RoMa teacher；
- teacher matches + RANSAC/MAGSAC 验证。

---

## 4.4 点级匹配标签

对每个采样点：

```text
valid match：存在几何对应
no match：不可见、遮挡、出视野
ignore：深度或几何监督缺失
```

有效对应坐标：

\[
p_B^*=\pi_B(T_{BA}\pi_A^{-1}(p_A,D_A(p_A)))
\]

匹配标签可以转为：

- assignment matrix；
- dustbin 标签；
- 坐标误差；
- 双向一致性。

---

## 4.5 预算标签

第一版不直接人工标注下一轮应采多少点。

可根据本轮反馈生成伪标签：

```text
good ratio 高 + 仍有未探索区域 → 增加
good ratio 中等 → 保持
连续低匹配率 / 高歧义 → 减少
```

后期改为任务驱动优化：

- 最大化正确匹配数量；
- 最大化内点率；
- 最小化位姿误差；
- 最小化采样计算量。

---

# 5. 损失函数

总损失：

\[
L=
\lambda_1L_{pair}
+\lambda_2L_{teacher}
+\lambda_3L_{patch}
+\lambda_4L_{match}
+\lambda_5L_{dustbin}
+\lambda_6L_{budget}
+\lambda_7L_{reject}
+\lambda_8L_{diversity}
+\lambda_9L_{compute}
\]

## 5.1 图像对分类损失

\[
L_{pair}=CE(\hat y_{pair},y_{pair})
\]

## 5.2 教师蒸馏损失

\[
L_{teacher}=KL(S^{student}\|T^{teacher})
\]

## 5.3 Patch 对应损失

\[
L_{patch}=CE(S^{patch},Y^{patch})
\]

## 5.4 局部匹配分配损失

\[
L_{match}=CE(M,Y^{match})
\]

或 Sinkhorn assignment loss。

## 5.5 Dustbin 损失

监督不可匹配点进入无匹配类别。

## 5.6 预算更新损失

\[
L_{budget}=CE(\hat b,b^*)
\]

## 5.7 拒绝损失

\[
L_{reject}=BCE(p_{accept},y_{accept})
\]

## 5.8 多样性损失

防止采样点聚集：

\[
L_{diversity}
=
\sum_{m\neq n}
\exp\left(
-\frac{\|p_m-p_n\|^2}{2\sigma^2}
\right)
\]

## 5.9 计算预算损失

\[
L_{compute}
=
\frac{N_{sampled}}{N_{total\_max}}
\]

防止模型通过无限增加采样获得更低匹配损失。

---

# 6. 训练流程

## 阶段A：数据准备

数据集建议：

```text
MegaDepth
ScanNet
Aachen Day-Night
RobotCar Seasons
可选动态场景数据
```

每个样本至少包含：

```python
{
    "image_a": ...,
    "image_b": ...,
    "pair_label": ...,
    "intrinsics_a": ...,
    "intrinsics_b": ...,
    "pose_a": ...,
    "pose_b": ...,
    "depth_a": ...,
    "depth_b": ...,
    "teacher_heatmap_a": ...,
    "teacher_heatmap_b": ...,
    "dynamic_mask_a": ...,
    "dynamic_mask_b": ...
}
```

负样本类型：

1. 不同场景、语义差异大；
2. 不同场景、语义相似；
3. 同地点但无重叠；
4. 同地点但变化严重；
5. 动态物体占主导；
6. 重复纹理困难样本。

---

## 阶段B：预计算教师缓存

离线生成：

```text
DINO patch 特征（可选）
VPR 热力图
动态语义概率图
几何 patch 标签
像素投影 GT
```

保存为：

```text
.npz / .pt / .pkl
```

避免训练时重复运行大型教师。

---

## 阶段C：第一阶段训练

固定：

```text
每个 patch 固定采样 2 个点
只训练 patch scoring + local matching
不启用动态预算
不启用早停拒绝
```

目标：

- 学会两图联合 patch 评分；
- 学会 patch 内局部匹配；
- 学会 dustbin。

---

## 阶段D：第二阶段训练

启用：

```text
点保留
失败点删除
patch 内重采样
动态预算：增加 / 保持 / 减少
```

训练 2～3 轮迭代。

建议一开始使用 teacher forcing：

- 根据 GT/teacher 决定部分预算更新；
- 再逐渐切换为模型自身预测。

---

## 阶段E：第三阶段训练

加入：

- 困难负样本；
- pair rejection；
- 预算衰减；
- compute loss；
- 动态区域抑制。

---

## 阶段F：几何任务微调

使用最终匹配执行：

```text
MAGSAC / RANSAC
Fundamental / Essential Matrix
Relative Pose
```

优化：

- 内点率；
- Sampson误差；
- Pose AUC；
- 匹配空间覆盖。

第一版不需要可微 RANSAC，可先将几何指标仅用于评估或伪标签更新。

---

# 7. 推理流程

```text
1. 输入图像对
2. 提取 DINO 语义特征和局部高分辨率特征
3. 双向 Cross-Attention 得到 pair-conditioned patch 特征
4. 预测图像对整体可匹配性
5. 得到候选 patch 对与初始采样预算
6. 每个 patch 内少量采样
7. 进行 patch 内局部匹配
8. 保留高质量匹配，删除失败点
9. 更新各 patch 预算
10. 在 patch 内重新采样
11. 重复 T 轮
12. 汇总全部保留匹配
13. 全局去重和空间 NMS
14. 计算最终拒绝分数
15. 若接受：输出稀疏匹配
16. 若拒绝：输出 no-match
```

推荐参数：

```text
T = 3
Top patch pairs = 32～64
每 patch 初始点数 = 1～2
每 patch 最大点数 = 8
全图最大采样数 = 512 或 1024
```

---

# 8. 输出格式

```python
{
    "accepted": bool,
    "pair_score": float,
    "keypoints_a": Tensor[N, 2],
    "keypoints_b": Tensor[N, 2],
    "match_confidence": Tensor[N],
    "patch_ids_a": Tensor[N],
    "patch_ids_b": Tensor[N],
    "num_iterations": int,
    "sampling_history": {
        "num_points": [...],
        "num_active_patches": [...],
        "mean_confidence": [...]
    }
}
```

---

# 9. 评估指标

## 匹配质量

```text
匹配数量
几何内点数
几何内点率
Sampson误差
匹配精度
匹配召回率
空间覆盖率
```

## 位姿估计

```text
Pose AUC@5°
Pose AUC@10°
Pose AUC@20°
Rotation Error
Translation Direction Error
```

## 拒绝能力

```text
AUROC
AUPRC
F1
False Accept Rate
False Reject Rate
```

## 效率

```text
总采样点数量
总局部点对比较次数
推理时间
显存
每个正确内点的计算成本
```

## 对比方法

```text
SuperPoint + LightGlue
XFeat
LoFTR
Efficient LoFTR
CasMTR
RoMa
LoMa
```

---

# 10. 消融实验

必须包含：

1. 无教师；
2. 只有 VPR 教师；
3. VPR + 动态语义教师；
4. 无 Cross-Attention，仅单图 patch 分数；
5. 固定采样预算；
6. 动态采样预算；
7. 无重采样；
8. 无 dustbin；
9. 无拒绝机制；
10. 描述子余弦相似度 vs pair-conditioned MLP 匹配；
11. 1轮、2轮、3轮、4轮迭代；
12. 不同总采样预算。

最重要的核心消融：

> 单图决定采样 vs 两图联合决定采样。

---

# 11. MVP 最小可行版本

第一版不要一次实现全部功能。

## MVP v0

```text
DINOv2 冻结
+ 双向 Cross-Attention
+ Top-K patch 对
+ 每 patch 固定采样4点
+ 局部描述子余弦相似度
+ dual-softmax
+ dustbin
```

目标：验证 pair-conditioned patch 选择是否有效。

## MVP v1

加入：

```text
成功点保留
失败点删除
patch 内重采样
3轮迭代
```

## MVP v2

加入：

```text
动态预算：增加 / 保持 / 减少
计算量约束
```

## MVP v3

加入：

```text
VPR 教师
动态语义教师
pair rejection
```

---

# 12. 推荐代码结构

```text
pair_adaptive_match/
├── configs/
│   ├── pasm_v0.yaml
│   ├── pasm_v1.yaml
│   └── pasm_v2.yaml
├── datasets/
│   ├── megadepth.py
│   ├── scannet.py
│   ├── pair_sampler.py
│   └── geometry_labels.py
├── teachers/
│   ├── vpr_teacher.py
│   ├── dynamic_teacher.py
│   └── cache_teacher.py
├── models/
│   ├── semantic_encoder.py
│   ├── local_encoder.py
│   ├── pair_patch_transformer.py
│   ├── patch_pair_head.py
│   ├── point_sampler.py
│   ├── local_matcher.py
│   ├── budget_head.py
│   ├── rejection_head.py
│   └── pasm.py
├── losses/
│   ├── teacher_loss.py
│   ├── patch_loss.py
│   ├── matching_loss.py
│   ├── budget_loss.py
│   ├── rejection_loss.py
│   └── total_loss.py
├── train.py
├── evaluate.py
├── infer.py
└── README.md
```

---

# 13. Codex 首轮开发任务建议

可直接交给 Codex：

```text
请先实现 PASM 的 MVP v0：

1. 使用冻结 DINOv2 提取两张图的 patch token。
2. 实现双向 Cross-Attention。
3. 对每个 A patch 在 B 中选 Top-3 候选。
4. 使用一个 MLP 输出 patch pair score。
5. 选择全局 Top-32 patch pairs。
6. 每个 patch 内均匀采样4个点。
7. 从一个轻量 CNN 的1/4特征图双线性采样局部描述子。
8. 在每个 patch pair 内计算16×16点对相似度矩阵。
9. 使用 dual-softmax + mutual nearest + dustbin 输出匹配。
10. 输出原图像素坐标和匹配置信度。
11. 暂时不实现动态预算、教师和拒绝模块。
12. 添加单元测试，保证坐标转换和patch边界正确。
```

当 v0 跑通后，再按 v1、v2、v3 逐步扩展。

---

# 14. 预期创新点表述

可以暂时表述为：

1. 提出一种图像对条件下的自适应稀疏匹配框架，不再由单张图独立决定关键点。
2. 以 DINO patch 为计算分配单元，通过联合语义和匹配反馈动态增加、减少或停止局部采样。
3. 在 patch 内保留高质量匹配、删除失败点并进行迭代重采样，实现数量可变的稀疏对应发现。
4. 将有效匹配和有效区域的衰减过程用于图像对拒绝，使匹配与拒绝自然统一。
5. 使用单图 VPR/动态语义教师提供初始关注先验，同时利用双图交叉注意力超越单图教师限制。

---

# 15. 主要风险

1. 动态预算属于离散决策，训练可能不稳定。
2. 教师热力图可能存在偏差，学生容易过度模仿。
3. DINO patch 语义强但局部定位弱，必须保留高分辨率局部分支。
4. 若总预算约束不足，模型可能倾向于所有 patch 都增加采样。
5. 负样本过于简单时，拒绝头会退化成语义分类器。
6. patch 对选择错误后，局部匹配无法恢复，因此必须保留 Top-R 候选和 dustbin。
7. 与 LoMa 比较时，应重点强调单位计算量、动态场景和拒绝能力，而不只比较总匹配数。

---

# 16. 第一阶段成功标准

MVP 不要求立即超过 LoMa。先验证：

1. 联合 patch 选择优于单图显著性；
2. 相同采样预算下，内点率高于均匀采样；
3. 动态物体上的匹配比例降低；
4. 低重叠和负图像对中，有效点数量自然下降；
5. 迭代重采样后，正确内点数持续增加；
6. 推理计算量明显低于全图 dense matching。

只要这些趋势成立，方法就值得继续扩展。
