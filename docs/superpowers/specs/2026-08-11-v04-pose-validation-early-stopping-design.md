# v0.4 几何教师验证早停设计

## 目标

在不改变 `geometry_calibrated_shape` 训练损失的前提下，以固定 9 组 held-out 图像对的几何教师损失选择 v0.4 语义头 checkpoint，并在连续多次未下降时停止训练。

## 范围

- 验证流程固定为：DINO → 双图语义头 → 与 stride=2 的 held-out 几何教师逐 Patch 比较。
- 默认每 2 个 epoch 验证一次，指标为与训练完全一致的 `calibrated_shape` total loss，连续 5 次未下降则停止。
- 未提供 `--validation-image-root` 时，训练行为保持原样，仍按训练总 loss 保存最佳权重。
- 提供验证集时，主 `--output` 保存最低验证教师损失权重；`.last.pt` 始终保存最后训练状态。

## 约束

- 固定九组已经反复参与模型选择，只能称为开发验证集，不能再视为最终泛化测试集。
- 验证不生成匹配图片或下游实验输出。
- 不改融合头、DaD、LoMa 匹配、预算分配、训练教师或任何训练损失。
