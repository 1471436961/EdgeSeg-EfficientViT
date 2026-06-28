# Phase 3 Decision Corrections

> 本文记录 Phase 3 推进过程中由人工 review 修正的关键设计决策。它不是 CUDA 概念笔记，也不是 P1a/P1b 每轮实验流水账；学习概念写入 [`../../LEARNING_LOG.md`](../../LEARNING_LOG.md)，kernel 实验历史写入 [`plugin_kernel_optimization_history.md`](plugin_kernel_optimization_history.md)。本文只记录会改变路线判断、验收口径或项目叙事的纠偏点。

---

## 1. P1b-7 不能直接和 P1a stage2+stage3 公平比较

### 早期不充分判断

P1b-7 的 `aggregation + cat + relu_linear_att` 中段 proxy 明显快于 Phase 2 TensorRT baseline，也减少了 launch 数。这个结果容易被误读成“P1b 已经优于 P1a stage2+stage3，可以替代主线”。

### 修正口径

P1b-7 只覆盖 `stage2/context` 两个 LiteMLA block，是 **stage2-only** 扩大边界实验；P1a stage2+stage3 覆盖四个 LiteMLA context block，覆盖范围更大。二者不能直接做一对一公平比较。

真正的全范围对照是：

```text
P1mix:
  stage2 = P1b-7
  stage3 = P1a-3b
```

P1mix 技术链路和 correctness 通过，但未稳定优于 P1a-3b stage2+stage3。因此当前主交付线仍是 P1a-3b stage2+stage3 `relu_linear_att-only` 两阶段 FP32 Plugin，P1b/P1mix 保留为消融证据和后续候选。

### 影响

- `phase3/README.md`、`integration_validation_report.md` 和 `PROJECT_STRATEGY.md` 均改为该口径。
- P1b-7 的价值被保留为“stage2-only 扩大边界有效”，而不是被误写成“全范围替代 P1a”。

---

## 2. 扩大 Plugin 边界不等于自然更快

### 早期不充分判断

Phase 1/2 指向 `aggregation + cat + relu_linear_att` 是更大的可融合边界，直觉上容易认为更大边界会减少中间 tensor 和 launch，因此一定更快。

### 修正口径

扩大边界确实可能减少 launch 和中间 tensor 读写，但也可能把 TensorRT/cuDNN 擅长的标准 Conv 路径移入自写 Plugin。P1b naive 版本已经证明：graph surgery、Plugin build、correctness 都成立时，性能仍可能因为自写 aggregation kernel 不如 TensorRT/cuDNN 标准实现而退化。

### 影响

- P1b 后续不再只追求“边界更大”，而是围绕 `fusedAggregationCatKernel` 做小步 A/B。
- P1b-1 到 P1b-7 证明扩大边界仍有价值；P1b-8 到 P1b-15 记录了继续微调的边际收益与反例。
- 最终主线采纳需要看全范围端到端、Nsight attribution 和 mIoU gate，而不是只看 launch 数。

---

## 3. P1a 两阶段 kernel 不应为了少一次 launch 强行合并

### 早期不充分判断

P1a `relu_linear_att-only` 使用 `computeVk` 和 `computeOutput` 两个 kernel。为了减少 launch 数和 workspace global write/read，曾考虑把两阶段合成单 kernel。

### 修正口径

`relu_linear_att` 先要对完整 spatial 维度完成全局归约：

```text
VK = sum_n V[n] * relu(K[n])
Z  = sum_n relu(K[n])
```

随后每个 spatial position 才能使用完整 `VK/Z` 计算输出。普通 CUDA kernel 只有 block 内同步，没有普通全 grid 同步；强行单 kernel 合并会导致重复归约、atomic/同步复杂度或破坏 `VK` 复用。Phase 3 的 single-kernel prototype 已作为反证记录，当前不采纳。

### 影响

- P1a 主线保持两阶段 kernel。
- 后续优化集中在 `computeVk` 归约策略、dim=16 专用路径和 shared-memory `VK` cache，而不是继续强推单 kernel 合并。

---

## 4. MX250 上 1ms 级性能差异必须冷机/同口径复测

### 早期不充分判断

部分实验曾在热机或后台负载不稳定时直接读 benchmark 数字，容易把温度、功耗墙、Windows 调度和频率漂移误判为 kernel 真实差异。

### 修正口径

MX250 是低功耗 Pascal GPU，SM 数少、无 Tensor Core、频率容易受温度影响。Phase 3 多次出现热机样本与冷机重测不同的情况，因此 1ms 级收益必须同时满足：

- 同一 warmup/measure 协议；
- 尽量冷机或温度稳定；
- 单任务运行，避免后台 GPU/CPU 干扰；
- CUDA Events latency 与 Nsight attribution 互相印证；
- 对低于噪声阈值的差异保持保守判断。

### 影响

- P1a stage2+stage3 的最终结论采用冷机复测和 Nsight attribution。
- P1b 多个 probe 的“不采纳”结论不只看单次 p50，也结合差异量级和硬件指标。

---

## 5. mIoU gate 是最终主线采纳条件，不是可选装饰

### 早期不充分判断

Plugin correctness 初期主要看单图 allclose、cosine similarity 和 argmax agreement。这足以做调试 gate，但不足以支撑最终语义安全结论。

### 修正口径

Phase 3 最终主线必须通过 Cityscapes val mIoU gate。mIoU 不用于证明 Plugin 更快，而用于证明 Plugin 替换后没有引入语义级回归。P1a-3b stage2+stage3 最终通过了 Cityscapes val 500 张图 mIoU gate，因此可以作为主交付线。

### 影响

- `integration_validation_report.md` 将 mIoU gate 列为正式验收项。
- 失败分支如 P1b/P1mix 不强制跑完整 mIoU，除非它们要竞争最终主线。

---

## 6. P1b/P1mix 保留为消融证据，而不是删除失败路线

### 早期不充分判断

当 P1b/P1mix 没成为当前主线时，容易倾向于删掉相关代码或弱化文档。

### 修正口径

P1b/P1mix 虽不作为当前主交付线，但它们证明了几个重要事实：

- 扩大边界能显著减少 stage2 中段 launch 数和 kernel time；
- 自写 aggregation 必须非常谨慎，否则会反噬 TensorRT/cuDNN 优化；
- P1mix 是判断 P1b 是否能替代 P1a stage2+stage3 的必要全范围对照；
- P1b 的反例和 probe 能展示真实 CUDA 优化中的取舍，而不是只展示成功路径。

### 影响

- P1a 和 P1b 代码都保留。
- P1b probe 中间结果归档到 `results/metrics/archive/p1b_probes/`。
- `plugin_kernel_optimization_history.md` 成为 Phase 3 优化总账，避免把失败尝试散落在对话中。

