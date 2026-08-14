# v0.4 多层 Cross-Attention 模型实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 在不加入 Self-Attention、不改变采样和损失流程的前提下，为 v0.4 增加 2 层双向 Cross-Attention 与轻量 FFN。

**架构：** 保留 DINOv2 token 投影和 Patch 价值头；新模型连续执行 2 层共享双向 Cross-Attention，每层后使用 `Linear → GELU → Linear` 非线性残差模块。旧 checkpoint 缺少层数字段时仍按原 1 层结构加载。

**范围：** 只修改 `matching.py`、`train_v04.py`、`infer.py` 和对应测试；不修改 DINO、LoMa、DaD、Patch 预算、损失定义或推理匹配逻辑。

### 任务 1：测试多层结构与 checkpoint 元数据

- [ ] 增加 2 层编码器结构测试，确认有 2 个 Cross-Attention 和 2 个 FFN，且没有 Self-Attention 模块。
- [ ] 增加 v0.4 checkpoint 保存/读取层数元数据测试。
- [ ] 运行测试确认先失败。

### 任务 2：实现多层 Cross-Attention 与 FFN

- [ ] 为 `PairSemanticEncoder` 增加 `cross_attention_layers` 参数，默认 1 以兼容旧模型。
- [ ] 当层数为 2 时增加共享双向 Cross-Attention、LayerNorm 和 FFN 残差块。
- [ ] 为 v0.4 训练命令增加 `--cross-attention-layers` 并写入 checkpoint。
- [ ] 推理从 checkpoint 读取层数，旧权重缺失时默认为 1。

### 任务 3：验证

- [ ] 运行新增测试、全量测试和 Python 编译检查。
- [ ] 给出新的 2 层 value-only 训练命令，输出到新 checkpoint 路径。
