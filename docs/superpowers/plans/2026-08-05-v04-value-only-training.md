# v0.4 纯 Patch 价值训练实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 在保持 v0.4 的 37×37 几何教师、静态预算和推理结构不变的前提下，增加纯 `L_value` 训练模式，用于验证 Patch 价值头能否独立拟合几何教师。

**架构：** 继续使用现有双图语义编码器，不新增网络层。训练时仍可计算点级匹配指标用于日志，但 `value-only` 模式的反向传播只使用 Patch 价值损失；原有联合损失模式保留，避免破坏已有实验。

**技术栈：** Python、PyTorch、unittest、现有 PASM v0.4 训练代码。

## 全局约束

- Patch 网格保持 37×37。
- 几何教师仍读取现有 `targets_geometry_stride2` 缓存。
- 总点数保持 1024，预算上限保持 4，`budget_score_power` 保持 1.0。
- 不修改 DINOv2、LoMa、DaD 权重和推理流程。
- 原有联合 `match + value` 训练模式保持可用。
- 新实验建议训练 60 轮，输出到新的检查点路径。

### 任务 1：增加纯价值损失的可测试开关

**文件：**

- 修改：`pasm_v0/train_v04.py`
- 修改：`tests/test_pasm_v0_train_v04.py`

**接口：**

- `v04_loss_terms(..., loss_mode="match_value")` 保持原返回值 `(match_loss, value_loss, total_loss)`。
- `loss_mode="value_only"` 时返回的 `total_loss` 必须等于 `value_loss`，且 `total_loss.backward()` 不产生 `edge_weights.grad`。
- `train_model(..., loss_mode="value_only")` 记录该模式并使用纯价值损失训练。
- CLI 增加 `--value-only`，不改变旧命令默认行为。

**步骤：**

- [ ] 先新增一个断言纯价值模式不让点级边权产生梯度的单元测试，并运行确认失败。
- [ ] 给损失函数和训练入口增加 `loss_mode` 参数。
- [ ] 在 CLI 中把 `--value-only` 映射为 `loss_mode="value_only"`。
- [ ] 在检查点中记录 `loss_mode`，续训时拒绝模式不一致的检查点。
- [ ] 运行 v0.4 训练单元测试和编译检查。

### 任务 2：验证纯价值训练行为

**文件：**

- 使用：`pasm_v0/train_v04.py`
- 使用：`pasm_v0/diagnose_v04_distribution.py`

**步骤：**

- [ ] 用 60 轮、`--value-only` 训练新的几何检查点。
- [ ] 记录训练集 value loss、教师熵和 KL 差距。
- [ ] 重新生成第三组、第五组 Patch 教师与模型原始价值热图。
- [ ] 用相同静态推理流程评估 9 组 held-out 图像的预算集中度、3px/6px 内点率和覆盖率。

### 任务 3：结果判定

- [ ] 若训练热图明显贴合教师且 held-out 改善，确认原联合损失存在干扰。
- [ ] 若训练集仍无法贴合教师，下一步只讨论价值输出形式或局部空间结构，不增加匹配模块。
- [ ] 若价值热图改善但匹配效果下降，再单独设计匹配分支，不让同一个价值编码器承担两个任务。
