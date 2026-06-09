# Phase 1 设计纠偏记录

> 本文记录 Phase 1 实施过程中由人工 review 纠正的关键设计决策与取舍问题。它不是流水账，也不是学习笔记，而是阶段内方法论审计：哪些地方曾经判断不严谨、最终如何修正、这些修正为什么影响后续 Phase 2/3。
>
> 跨阶段摘要见 [`../../PROJECT_DECISION_CORRECTIONS.md`](../../PROJECT_DECISION_CORRECTIONS.md)；概念学习与技术问答沉淀见 [`../../LEARNING_LOG.md`](../../LEARNING_LOG.md)。

---

## 1. 计时口径：Latency 与 Throughput 不能混用

**原问题**：曾倾向使用“批量录制 CUDA Events 后统一 synchronize”的方式作为 latency 主口径。

**纠正**：这更接近 throughput 测法，不适合作为单请求 latency 主口径。

**最终口径**：

- `latency`：默认模式，逐次 CUDA Event 记录单次 forward 延迟。
- `throughput`：可选模式，批量 enqueue 后统一同步，报告 FPS。
- JSON 必须记录 `measurement_mode`。

**影响**：避免把吞吐优化口径误写成实时推理 latency 结论。

---

## 2. NVTX 与同步边界：NVTX 不是计时工具

**原问题**：早期表述容易让人误解为 NVTX range 边界附近可以加入 `torch.cuda.synchronize()`。

**纠正**：NVTX 只负责结构标注，不负责计时；range 内同步会破坏异步执行并污染 Nsight profiling。

**最终口径**：

- NVTX range 内只做 `range_push/range_pop`。
- `torch.cuda.synchronize()` 只允许出现在 warmup/measure 边界，以及 CUDA Event 读取处。
- 端到端 latency 以 CUDA Events 为准。
- 组件耗时以 Nsight SQLite attribution 为准，不直接用 NVTX range 的 `end-start`。

**影响**：保证 Nsight 归因结果反映原始执行形态，而不是被人为同步改造后的执行形态。

---

## 3. Plan D sanity check：必须保留 patch 前原始参照

**原问题**：曾出现“先 monkey-patch，再对 patched 路径做前后检查”的伪检查风险。

**纠正**：如果没有 patch 前的原始输出参照，就无法证明 monkey-patch 没有改变模型语义。

**最终口径**：

- patch 前通过 hook 捕获 LiteMLA 原始输入/输出。
- patch 后用同一输入单独跑 patched forward。
- 逐模块用 `torch.allclose(atol=1e-5, rtol=1e-5)` 校验。
- JSON 记录每个模块的 `max_abs_diff` / `mean_abs_diff` / pass 状态。

**影响**：保证 Plan D 的 NVTX patch 是 profiling 注释层，而不是隐式改变计算图的实现层。

---

## 4. 权重策略：正式 baseline 不允许随机权重伪装

**原问题**：曾考虑无权重时加载 ImageNet backbone + 随机 SegHead。

**纠正**：这种部分加载对 latency 无明显收益，还会引入 missing/unexpected keys、语义 warning 和结果解释混乱。

**最终口径**：

- 正式 baseline 必须使用 Cityscapes B0 真实权重。
- 无权重只能显式 `--allow-random-weights` 做 smoke test。
- smoke JSON 必须标记 `is_smoke_test=true`，不能写入正式性能结论。

**影响**：避免把“脚本链路验证”误当成“正式权重性能基线”。

---

## 5. MACs / FLOPs：工具和输入分辨率口径要匹配

**原问题**：曾考虑新增 `thop`，且引用过不适合 Cityscapes 全分辨率的 MACs 数值。

**纠正**：项目已有 `torchprofile`；ImageNet 输入下的 MACs 不能混用到 `1024x2048` Cityscapes 输入。

**最终口径**：

- MACs 统计为可选 `--profile-macs`。
- 优先使用项目已有依赖或更贴近现有工具链的实现。
- 报告必须注明输入分辨率对应的 MACs 口径。

**影响**：避免报告中出现“数值看似专业但输入口径错误”的硬伤。

---

## 6. JSON Schema：性能数字必须带可复现溯源

**原问题**：早期 JSON 侧重 latency / memory，环境和输入/权重溯源不足。

**纠正**：性能基线必须能复现、能对比、能追责。

**最终口径**：

- 记录 device / torch / cuda / cudnn / cuDNN flags。
- 记录 weights path / sha256 / status。
- 记录 input path / sha256 / dummy seed。
- 记录 script version 与 env patches。

**影响**：把一次本地 benchmark 变成可被 Phase 2/3 继续引用的正式实验记录。

---

## 7. Plan A/B/C/D：每一档只回答一个层级的问题

**原问题**：Plan B/C 的职责曾有过混淆，Plan C 曾被描述得过宽或过窄。

**纠正**：profiling plan 必须按证据链分层。

**最终口径**：

| Plan | 回答的问题 | 用途 |
|---|---|---|
| A | 原生 PyTorch 整体多快？ | 干净端到端 latency baseline |
| B | 全模型哪个大区域最慢？ | stage/head 级归因 |
| C | 热点区域内部是什么组件慢？ | stage0 / stage2 / head 组件归因 |
| D | stage2 LiteMLA 内部哪段适合融合？ | Phase 3 Plugin 边界细化 |

**影响**：防止用一个 profiling 视图承担所有结论，导致证据链混乱。

---

## 8. EfficientViT 源码事实：stage2/3 是先 context 再 local

**原问题**：曾把 stage2/3 内部 `context_module` 与 `local_module` 的顺序说反。

**纠正**：按源码，stage2/3 block 内是先 `context_module`，再 `local_module`。

**最终口径**：

- `context_module` 对应 LiteMLA 路径。
- `local_module` 对应 MBConv / local convolution 路径。
- stage2 中 context 比 local 更耗时，是 LiteMLA 进入 Phase 3 候选的重要证据之一。

**影响**：避免把 LiteMLA 和 MBConv 的 attribution 解释反。

---

## 9. Phase 3 候选排序：最大耗时不等于最适合 Plugin

**原问题**：曾容易被 Plan C 中 stage0/head 的高耗时占比带偏，把最大耗时模块直接等同于 Plugin 主线。

**纠正**：Plugin 候选要同时看端到端收益、非标准程度、TensorRT/cuDNN 是否已有成熟优化、实现风险和展示价值。

**最终口径**：

- `stage0/head/stage2-local`：标准 Conv / MBConv 为主，是重要工程热点，但不一定优先写 Plugin。
- `stage2/context` LiteMLA：不是全模型最大耗时，但更符合自定义 Plugin 的高区分度主线。
- 报告中必须区分“端到端收益排序”和“Plugin 展示价值排序”。

**影响**：让 Phase 3 目标从“哪里最大就写哪里”升级为“哪里最能体现自定义算子价值”。

---

## 10. Plan D 范围：优先聚焦 stage2 LiteMLA

**原问题**：Plan D 曾倾向覆盖所有 LiteMLA。

**纠正**：根据 Plan B/C 证据链，stage2 是更关键的 LiteMLA 热点；stage3 LiteMLA 绝对耗时较小。

**最终口径**：

- Plan D 第一版只针对 `stage2/context` LiteMLA。
- 目标是细化 Phase 3 Plugin 边界，不是做全模型递归 profiler。

**影响**：减少 NVTX 扰动和分析噪声，把实验资源集中在最能支撑 Phase 3 的位置。

---

## 11. LiteMLA Plugin 边界：从整体模块细化到候选子路径

**原问题**：最初容易把候选表述为“LiteMLA 整体 Plugin”。

**纠正**：Plan D 显示 `aggregation` 和 `relu_linear_att` 中间存在 `cat`，Plugin 边界应按子路径分层比较。

**最终候选**：

1. `aggregation-only` 或 `relu_linear_att-only`：边界小，适合作为 MVP。
2. `aggregation + cat + relu_linear_att`：更像性能优化主线，能减少中间 tensor 写回和 launch。
3. 整体 LiteMLA Plugin：复杂度最高，可作为 fallback / 上限方案。

**影响**：Phase 3 不再停留在“优化 LiteMLA”这种粗粒度口号，而有了可执行的 Plugin 边界候选。

---

## 12. Phase 1 精度口径：不把 mIoU 作为阶段完成条件

**原问题**：不同文档曾对 Phase 1 是否测精度表述不一致。

**纠正**：Phase 1 是 baseline / profiling / bottleneck attribution 阶段，不是精度评测阶段。

**最终口径**：

- Phase 1 可做样图可视化 sanity check。
- Phase 1 不以 mIoU 为完成条件。
- PyTorch vs TensorRT / Plugin 的数值一致性与 mIoU 回归放到 Phase 2/3。

**影响**：避免阶段目标失焦，也避免没有完整 Cityscapes 评测集时被精度流程拖住。

---

## 13. Nsight 权限边界：普通权限不能完全排除 CPU/WDDM 因素

**原问题**：曾倾向把普通权限 Nsight timeline 解释得过满。

**纠正**：Windows 下 CPU sampling / context switch / WDDM tracing 通常需要管理员权限。

**最终口径**：

- 当前结果可以说“没有明显证据表明 CPU enqueue / WDDM 是主导瓶颈”。
- 不能说“CPU/OS 因素已被完全排除”。
- 若后续出现明显 GPU 空洞或 tail latency 异常，需要管理员权限重跑更完整 trace。

**影响**：让报告结论保持证据边界，不越权解释。

---

## 14. Nsight 截图口径：截图辅助说明，SQLite 表负责定量结论

**原问题**：曾对 `Threads -> NVTX`、`CUDA HW -> NVTX`、`CUDA HW -> Kernels` 的用途解释不够清楚。

**纠正**：

- `Threads -> NVTX`：看 Python/NVTX 逻辑顺序与 range 边界。
- `CUDA HW -> Kernels`：看实际 GPU kernel 执行形态。
- `CUDA HW -> NVTX`：看 GPU 侧投影趋势，可辅助观察但不是最终定量依据。
- 定量结论以 Nsight SQLite attribution 表为准。

**影响**：避免从截图直觉直接推出错误的组件耗时排序。

---

## 总结

Phase 1 的关键价值不只是“跑出了 baseline”，而是通过多轮纠偏把 profiling 方法论稳定下来：

- 先区分计时、标注、归因三件事；
- 再按 Plan A/B/C/D 建立分层证据链；
- 最后把 Phase 3 候选从“最大耗时模块”修正为“最适合展示 TensorRT Plugin 价值的非标准可融合路径”。

这些修正是后续 Phase 2/3 继续推进时必须遵守的边界。
