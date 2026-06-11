# LiteMLA Plugin Fusion Design

> **关联阶段**：[`phase3/README.md`](../README.md)
>
> **状态**：v0.1，Phase 3 启动版设计文档。本文先定义 Plugin 候选、证据来源、融合边界和验证口径；尚未开始 CUDA / TensorRT Plugin 代码实现。

---

## 1. 设计目标

Phase 3 的核心目标不是“随便写一个 CUDA kernel”，而是基于 Phase 1/2 的证据链，选择一个有数据支撑、能展示 TensorRT Plugin 工程能力的 LiteMLA 子路径。

当前默认目标：

1. 先做 **P1a MVP**：`relu_linear_att-only` 或 `aggregation-only`。
2. MVP 成功后评估 **P1b 主性能边界**：`aggregation + cat + relu_linear_att`。
3. 只有在必要时再考虑 **P1c fallback**：整体 LiteMLA Plugin。

---

## 2. 证据链摘要

### 2.1 Phase 1 证据

Phase 1 Plan D 只细拆 `stage2/context` LiteMLA，结论是：

- `aggregation` 与 `relu_linear_att` 是 LiteMLA 内部两大主要耗时段。
- 二者之间存在 `cat`，因此中段组合 `aggregation + cat + relu_linear_att` 有减少中间 tensor 写回 / 读取 / launch 的潜在价值。
- `stage0/head` 虽然端到端耗时高，但主要是标准 MBConv / Conv 路径，不能直接抢占 LiteMLA Plugin 主线。

### 2.2 Phase 2 证据

Phase 2 TensorRT 后：

- TensorRT FP32 p50 从 PyTorch `85.70 ms` 降到 `54.44 ms`，证明默认 TensorRT 优化有效。
- EngineInspector 显示 ONNX `393` nodes 压缩为 TensorRT `155` layers，但 `stage2/context` LiteMLA 没有自动合成一个单一 fused operator。
- TensorRT Nsight attribution 中 `stage2/context` 仍有 residual runtime，`aggregation + attention_core` proxy 约 `5.443 ms / iter`、`38 launches / iter`。

因此 Phase 3 不是重复 TensorRT 已做好的标准融合，而是针对 TensorRT 未能整体融合的 LiteMLA 子路径做 Plugin 验证。

---

## 3. 候选边界

### 3.1 P1a：`relu_linear_att-only`

**目标**：作为第一版 MVP，验证 TensorRT Plugin 注册、engine build、runtime 执行和数值对齐链路。

**优势**：

- 源码语义边界清晰。
- 与 LiteMLA 的非标准线性注意力核心直接相关。
- 展示价值高，适合面试讲解。

**风险**：

- 单段绝对收益有限。
- 需要谨慎处理 FP32 / FP16 / FP32 accumulate。
- TensorRT layer name 中的 `attention_core` 只是 residual proxy，不能直接等同于源码边界。

### 3.2 P1a：`aggregation-only`

**目标**：作为 P1a 的对照 MVP，验证 aggregation 分支是否比 `relu_linear_att` 更适合作为第一版 Plugin。

**优势**：

- Phase 1 耗时与 `relu_linear_att` 接近。
- Phase 2 TensorRT 后仍保留一定 runtime。

**风险**：

- aggregation 更接近卷积/聚合路径，可能已经被 TensorRT/cuDNN 部分优化。
- 若单独实现，展示“非标准注意力”的区分度低于 `relu_linear_att`。

### 3.3 P1b：`aggregation + cat + relu_linear_att`

**目标**：作为主性能评估边界，尝试减少中间 tensor 落地、拼接和重复读取。

**优势**：

- Phase 1 Plan D 和 Phase 2 residual proxy 都支持该组合有较高 runtime 覆盖。
- 比单段 Plugin 更可能产生可观端到端收益。
- 能展示更完整的 fusion 思路。

**风险**：

- TensorRT graph 替换更复杂。
- 输入输出 tensor contract 更难定义。
- 共享内存 / 寄存器容量可能限制组合融合收益。

### 3.4 P1c：整体 LiteMLA Plugin

**目标**：作为 fallback / 上限方案。

**优势**：

- 融合空间最大。
- 可最大化减少中间 tensor 和 launch。

**风险**：

- 实现复杂度最高。
- 数值对齐难度最高。
- 调试成本高，不适合作为 Phase 3 第一版。

---

## 4. 第一版建议路线

第一版不直接写整体 LiteMLA Plugin，而是按下面顺序推进：

1. **确认实际 tensor contract**
   - 从 ONNX graph / TensorRT EngineInspector / PyTorch LiteMLA 源码中确认 `stage2/context` 输入输出 shape、layout、dtype。
   - 明确 batch size 固定为 1，输入 shape 固定为 Cityscapes `1024x2048` 导出的 TensorRT engine。

2. **选择 P1a MVP**
   - 若 `relu_linear_att` 的输入输出边界能在 TensorRT graph 中清晰替换，优先选 `relu_linear_att-only`。
   - 若 `relu_linear_att` 边界难以插入，但 aggregation layer 边界更清晰，则先选 `aggregation-only`。

3. **实现 Plugin skeleton**
   - 先不优化 kernel，只跑通 TensorRT Plugin Creator、serialization、engine build、runtime enqueue。
   - 输出可先做 pass-through / reference-like 实现，用于验证接入链路。

4. **实现 CUDA kernel**
   - 在 skeleton 通过后再写真实计算。
   - 每次只扩大一个功能边界，避免同时调 Plugin API 和 CUDA 数值。

5. **benchmark + Nsight**
   - 复用 Phase 2 `benchmark_trt_engine.py` 的 CUDA Events 口径。
   - 复用 TensorRT Nsight attribution 口径。
   - 对比 TensorRT FP32 baseline vs Plugin engine。

---

## 5. 数值策略

LiteMLA 原始 PyTorch 实现中 `relu_linear_att` 存在 FP32 保护倾向。Phase 3 不应默认强行 FP16。

第一版建议：

- 主 Plugin baseline 使用 FP32。
- 若测试 FP16，必须单独记录为风险实验。
- attention / normalization / division 路径优先考虑 FP32 accumulate。
- 输出对齐至少记录：
  - `max_abs_diff`
  - `mean_abs_diff`
  - relaxed allclose
  - cosine similarity
  - argmax pixel agreement

完整 mIoU 不作为第一版 Plugin MVP 的必要条件，但最终集成验证应规划更大样本或 mIoU 回归。

---

## 6. TensorRT 集成策略待确认

后续需要在 Step 2 具体确认以下问题：

1. 使用 ONNX graph surgery 插入 Plugin，还是使用 TensorRT Network API 手动替换子图。
2. Plugin 接口采用 `IPluginV2DynamicExt` 还是 TensorRT 8.6.1 更推荐的兼容接口。
3. 是否需要为 Windows / TensorRT 8.6.1 单独处理 DLL 导出和 Plugin 注册。
4. 是否复用 Phase 2 C++ demo 作为 Plugin engine runtime smoke。

这些问题确认前，不应开始写正式 CUDA kernel。

---

## 7. 验证计划

| 层级 | 验证内容 | 通过标准 |
|---|---|---|
| Build | Plugin 可编译、可被 TensorRT 发现 | CMake build 通过，engine build 不报 plugin missing |
| Runtime | engine 可执行 | C++ / Python runtime smoke 通过 |
| Correctness | 输出对齐 | relaxed allclose / argmax agreement 达到设计阈值 |
| Performance | latency 对比 | 与 TensorRT FP32 baseline 同口径比较 |
| Attribution | Nsight 复核 | Plugin 覆盖范围在 trace 中可解释，residual hotspot 变化可说明 |

---

## 8. 当前不做的事

- 不直接上整体 LiteMLA Plugin。
- 不先做 INT8。
- 不为 stage0/head 写第一版 Plugin。
- 不把 TensorRT `attention_core` proxy 当作源码级候选命名。
- 不用单次 Nsight screenshot 代替 SQLite attribution。

