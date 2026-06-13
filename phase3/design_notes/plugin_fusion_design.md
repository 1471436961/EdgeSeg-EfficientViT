# LiteMLA Plugin Fusion Design

> **关联阶段**：[`phase3/README.md`](../README.md)
>
> **状态**：v0.4，已吸收 Phase 3 Step 2 的 `stage2/context` tensor contract、Step 4 Plugin skeleton / toy engine build 结果，以及 Step 5 单层 CUDA 数学验证结果。后续流程继续保持拆分：Step 6 完整 EfficientViT graph 集成，Step 7/8 性能与 Nsight 验证。

---

## 1. 设计目标

Phase 3 的核心目标不是“随便写一个 CUDA kernel”，而是基于 Phase 1/2 的证据链，选择一个有数据支撑、能展示 TensorRT Plugin 工程能力的 LiteMLA 子路径。

当前默认目标：

1. 先做 **P1a MVP**：`relu_linear_att-only`，`aggregation-only` 作为 fallback / 对照实验。
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

### 2.3 Step 2 tensor contract

Step 2 已落盘 [`stage2_context_tensor_contract.md`](stage2_context_tensor_contract.md)，关键结论如下：

- 目标只覆盖 ONNX / TensorRT 中的 `backbone.stages.2.op_list.{1,2}.context_module/main` 两个 LiteMLA 实例。
- ONNX `stages.2` 是 0-indexed，对应 backbone forward 输出字典里的语义 `stage3` 特征层。
- 固定输入 `1x3x1024x2048` 下，两个目标 context 的输入特征都是 `[1,64,64,128]`。
- B0 配置中 LiteMLA `dim=16`；qkv Conv 输出 `[1,192,64,128]`；aggregation 输出 `[1,192,64,128]`；cat 后输入 attention 的 tensor 是 `[1,384,64,128]`。
- `relu_linear_att-only` 的真实 contract 是 `[1,384,64,128] -> [1,128,64,128]`。
- `aggregation + cat + relu_linear_att` 的真实 contract 是 `[1,192,64,128] -> [1,128,64,128]`。

### 2.4 Step 3 Plugin API / CMake design

Step 3 已落盘 [`plugin_api_cmake_design.md`](plugin_api_cmake_design.md)，关键结论如下：

- 第一版 Plugin 使用 TensorRT 8.6.1 支持的 `IPluginV2DynamicExt`。
- Plugin 名称为 `EdgesegReluLinearAttention_TRT`，namespace 为 `edgeseg`。
- 第一版仅支持 FP32、NCHW、fixed shape `[1,384,64,128] -> [1,128,64,128]`。
- 构建产物是 Windows DLL：`edgeseg_relu_linear_attention_plugin.dll`。
- Step 4 先实现 Plugin skeleton 和 toy network build，不直接跳到完整 EfficientViT graph surgery。
- Step 5 已在 toy/plugin 单层层面实现真实 `relu_linear_att` CUDA 数学并完成 PyTorch reference 对齐。

### 2.5 Step 4 Plugin skeleton

Step 4 已完成 P1a Plugin skeleton：

- CMake + MSVC + CUDA 编译通过，生成 `phase3/plugin/build/edgeseg_relu_linear_attention_plugin.dll`。
- Python toy builder 成功加载 DLL，并在 TensorRT Plugin Registry 中找到 `EdgesegReluLinearAttention_TRT` Creator。
- 最小 toy network 输入 `[1,384,64,128]`、输出 `[1,128,64,128]`，已成功 build serialized engine。
- 实测元数据见 [`../results/metrics/relu_linear_attention_toy_build.json`](../results/metrics/relu_linear_attention_toy_build.json)。
- Step 4 的初始 skeleton 曾只做 zero-fill；Step 5 已将 enqueue 替换为真实 `relu_linear_att` CUDA 实现。

### 2.6 Step 5 单层 CUDA 数学验证

Step 5 已完成 P1a `relu_linear_att-only` 的真实 CUDA kernel 与单层对齐：

- CUDA 实现采用两阶段 FP32 kernel：先计算每个 head 的 `vk = v_pad @ relu(k)^T` 小 workspace，再按 `(head, pixel)` 计算 16 维归一化输出。
- MX250 约束下不使用 FP16 / Tensor Core 路径；workspace 只保存 `8 x 17 x 16` 个 FP32 数值，约 2.1KB。
- 验证脚本为 [`../scripts/validate_relu_linear_attention_plugin.py`](../scripts/validate_relu_linear_attention_plugin.py)。
- 实测结果见 [`../results/metrics/relu_linear_attention_plugin_validation.json`](../results/metrics/relu_linear_attention_plugin_validation.json)：`max_abs_diff=1.4156e-07`、`mean_abs_diff=8.4468e-09`、`cosine_similarity≈1.0`、`allclose_pass=true`、`argmax_pixel_agreement=1.0`。
- 本结果只证明单层 Plugin 数学正确，不代表完整 EfficientViT graph 已完成替换。

---

## 3. 候选边界

### 3.1 P1a：`relu_linear_att-only`

**目标**：作为第一版 MVP，验证 TensorRT Plugin 注册、engine build、runtime 执行和数值对齐链路。

**Step 2 contract**：替换 `Concat_output_0 -> Cast_1_output_0`，输入 `[1,384,64,128]`，输出 `[1,128,64,128]`，FP32，不需要 Plugin 权重。

**优势**：

- 源码语义边界清晰。
- 与 LiteMLA 的非标准线性注意力核心直接相关。
- 展示价值高，适合面试讲解。

**风险**：

- 单段绝对收益有限。
- 需要谨慎处理 FP32 / FP16 / FP32 accumulate。
- TensorRT layer name 中的 `attention_core` 只是 residual proxy，不能直接等同于源码边界。

### 3.2 P1a：`aggregation-only`

**目标**：作为 P1a 的对照 / fallback，验证 aggregation 分支是否值得在 `relu_linear_att-only` 之外单独实现。

**Step 2 contract**：替换 `qkv/conv/Conv_output_0 -> aggreg.0/aggreg.0.1/Conv_output_0`，输入 `[1,192,64,128]`，输出 `[1,192,64,128]`，FP32，需要 aggregation depthwise 5x5 与 grouped 1x1 权重。

**优势**：

- Phase 1 耗时与 `relu_linear_att` 接近。
- Phase 2 TensorRT 后仍保留一定 runtime。

**风险**：

- aggregation 更接近卷积/聚合路径，可能已经被 TensorRT/cuDNN 部分优化。
- 若单独实现，展示“非标准注意力”的区分度低于 `relu_linear_att`。

### 3.3 P1b：`aggregation + cat + relu_linear_att`

**目标**：作为主性能评估边界，尝试减少中间 tensor 落地、拼接和重复读取。

**Step 2 contract**：替换 `qkv/conv/Conv_output_0 -> Cast_1_output_0`，输入 `[1,192,64,128]`，输出 `[1,128,64,128]`，FP32，需要 aggregation 权重，后续输出继续喂给现有 `proj/conv/Conv`。

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
   - 已完成，见 [`stage2_context_tensor_contract.md`](stage2_context_tensor_contract.md)。

2. **选择 P1a MVP**
   - Step 2 结论：第一版优先选 `relu_linear_att-only`，因为它边界清晰、不需要权重，最适合先验证 Plugin 接入闭环。
   - 若 `relu_linear_att-only` 的 graph replacement 在 TensorRT 8.6.1 / Windows 路径下不可行，再退到 `aggregation-only`。

3. **实现 Plugin skeleton**
   - 先不优化 kernel，只跑通 TensorRT Plugin Creator、serialization、engine build、runtime enqueue。
   - 输出可先做 pass-through / reference-like 实现，用于验证接入链路。

4. **实现 CUDA kernel 与单层对齐**
   - 在 skeleton 通过后再写真实计算。
   - 在 toy/plugin 单层层面与 PyTorch reference 对齐，先证明 Plugin 自己算得对。
   - 本步不做完整 EfficientViT graph surgery，避免同时调 Plugin API、CUDA 数值和图替换。

5. **集成完整 EfficientViT TensorRT graph**
   - 优先用 ONNX graph surgery 把 `Concat_output_0 -> Cast_1_output_0` 子图替换为 custom op。
   - 若 ONNX parser custom op 路线不稳定，再评估 TensorRT Network API 手动重建局部网络。
   - 该步骤通过后，才进入完整 Plugin engine 的端到端 benchmark。

6. **benchmark + Nsight**
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

## 6. TensorRT 集成策略状态

Step 3 已确认第一版 Plugin API 与 CMake 构建口径，详见 [`plugin_api_cmake_design.md`](plugin_api_cmake_design.md)。当前状态如下：

| 问题 | 当前结论 |
|---|---|
| Plugin 接口 | 使用 TensorRT 8.6.1 支持的 `IPluginV2DynamicExt` |
| 构建产物 | Windows DLL：`edgeseg_relu_linear_attention_plugin.dll` |
| DLL 加载 | Python 侧用 `ctypes.CDLL`，C++ 侧用 `LoadLibraryA`，均需早于 engine build / deserialize |
| C++ runtime smoke | 复用 Phase 2 C++ demo 的 TensorRT Runtime 链路 |
| 真实 EfficientViT graph 替换 | Step 6 再做；优先 ONNX graph surgery，若不稳定再评估 TensorRT Network API 手动替换 |
| Plugin skeleton | Step 4 已完成 toy engine build；Step 5 已实现真实 `relu_linear_att` 数学并完成单层验证 |

在 Step 4 中只实现 Plugin skeleton 与 toy network build；Step 5 专注真实 CUDA kernel 与单层数值对齐；Step 6 再做真实 EfficientViT graph surgery。三者不应混在同一次改动里。

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
