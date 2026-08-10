# v0.6 语义条件化 DaD 局部对比度融合设计

## 目标

在保持 DaD 像素级候选能力的前提下，用冻结的双图语义特征抑制非共同可见区域，并且只增强局部 DaD 响应中真正突出的点，避免普通残差把一个高可见 Patch 内的低质量像素一起抬高。

## 模型

新增 `ContrastiveDensePointFusion`，不替换已有 `DensePointFusion`。对 DaD 的 32 通道特征 `d`、投影后的 32 通道双图语义特征 `s`，以及逐元素交互 `d*s` 做 reflect-padded 卷积，预测逐像素的非负 `gain` 与 `penalty`。

令 `z=(base_logit-local_mean)/(local_std+eps)`，其中局部统计采用 5x5 reflect padding。最终输出：

`final_logit = base_logit + gain*z - penalty`

`gain` 与 `penalty` 由 softplus 产生，末层偏置初始化为 -5，使训练起点近似于原始 DaD；卷积全部使用 reflect padding，避免重新引入边缘伪影。

## 训练

候选点软 BCE 维持不变。原先的正负均值 hinge ranking 在满足 margin 后会恒为零，改为带标签差值权重的软 pairwise ranking：对标签 >= 0.5 的候选点和标签 <= 0.1 的候选点分别排序，计算所有正负组合的 `softplus(-(positive_logit-negative_logit)/temperature)`，并以标签差值加权。它在已有较大间隔时仍保留微小但非零的梯度。

训练入口加入：

- `--fusion-mode {residual,contrastive}`，默认 `residual`；
- `--ranking-mode {hinge,soft_pairwise}`，默认 `hinge`；
- `--ranking-temperature`，默认 1.0。

## 兼容性与检查点

checkpoint 保存 `fusion_mode`、`ranking_mode`、`ranking_temperature`。加载器依据 `fusion_mode` 创建模型；旧 checkpoint 没有该字段时默认为 `residual`，保证已有 v0.5 模型可评测。

## 验证

单元测试覆盖：局部 z-score 的符号与常量图有限性；接近 identity 的初始化；正 gain 时高于局部均值的 logit 上升、低于局部均值的 logit 下降；soft pairwise loss 在明显 margin 已满足时仍为正并能反传；checkpoint 模式工厂对旧格式兼容。
