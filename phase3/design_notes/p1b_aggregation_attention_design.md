# P1b Aggregation + Attention Plugin Design

> **关联阶段**：[`../README.md`](../README.md)
>
> **状态**：v0.9。P1b skeleton / parser toy、真实 EfficientViT ONNX surgery build smoke、PyTorch reference 捕获、第一版 CUDA 数学 block-level correctness、真实 P1b engine cold-run correctness / latency、P1b Nsight attribution、P1b-1 fused aggregation+cat 优化均已完成。当前结论是：naive P1b 数学正确但性能退化；P1b-1 fused aggregation+cat 修复了主要退化点，使中段边界从 `6.189ms/iter` 降到 `4.848ms/iter`，对比 Phase 2 baseline `aggregation + attention_core` `5.443ms/iter` 达到 `1.123x` kernel-time speedup。P1b fused 已证明扩大边界有价值，但仍未达到 P1a `aggregation + plugin` proxy 约 `3.062ms/iter` 的水平。

---

## 1. 设计目标

P1b 的目标是在 P1a `relu_linear_att-only` 已跑通的基础上，把替换边界向前扩展到 LiteMLA 的中段组合：

```text
qkv output
  -> aggregation depthwise 5x5
  -> aggregation grouped 1x1
  -> cat(original qkv, aggregation output)
  -> relu_linear_att
  -> attention output
```

它不是整体 LiteMLA Plugin：`qkv Conv` 仍在 TensorRT/cuDNN 路径中，`proj Conv` 与 residual add 也保留在 Plugin 外部。

---

## 2. 为什么进入 P1b

Phase 2 TensorRT baseline 与 Phase 3 P1a 结果共同说明：

- Phase 2 中 `stage2/context` 的 `aggregation + attention_core` proxy 约为 `5.443 ms / iter`、`38 launches / iter`。
- P1a-3b 后，`relu_linear_att` Plugin layer 约为 `1.310 ms / iter`、`4 launches / iter`。
- P1a-3b 后，`aggregation + plugin` proxy 约为 `3.062 ms / iter`、`30 launches / iter`。
- 其中 `aggregation_preserved` 仍约为 `1.752 ms / iter`、`26 launches / iter`。

因此 P1a 已经证明 `relu_linear_att-only` 可以减少目标边界的 kernel time 和 launch 数，但剩余 runtime 仍有很大一部分留在 aggregation 与中间 tensor 流转上。P1b 的价值是验证：把 aggregation 与 attention 放进同一个 Plugin 边界后，是否能进一步减少中间写回、读取、concat 相关开销和 launch 数。

冷机端到端结果表明：naive P1b 数学正确，但性能慢于 baseline。随后的 P1b-1 fused aggregation+cat 实验证明，问题不在 P1b 边界本身，而在第一版 naive aggregation 中间 workspace 设计；融合 depthwise / grouped pointwise / cat 后，P1b 中段边界已能打赢 Phase 2 TensorRT baseline 的 `aggregation + attention_core` proxy。

---

## 3. 目标范围

P1b 只覆盖两个 `stage2/context` 实例：

| 模块 | 语义 |
|---|---|
| `backbone.stages.2.op_list.1.context_module.main` | stage2 第一个 LiteMLA context block |
| `backbone.stages.2.op_list.2.context_module.main` | stage2 第二个 LiteMLA context block |

固定 Cityscapes 输入 `1x3x1024x2048` 下，P1b 的 tensor contract 为：

| 项 | shape | 说明 |
|---|---:|---|
| runtime input | `[1, 192, 64, 128]` | `qkv/conv/Conv_output_0` |
| output | `[1, 128, 64, 128]` | 去 `Cast_1_output_0`，继续喂给现有 `proj/conv/Conv` |
| dtype | `float32` | 第一版不走 FP16 / Tensor Core |
| layout | `NCHW` | 固定 shape、batch=1 |

每个 block 有独立权重，不能假设共享：

| 权重 | shape | 语义 |
|---|---:|---|
| `aggreg.0.0.weight` | `[192, 1, 5, 5]` | depthwise 5x5，`groups=192`，`padding=2` |
| `aggreg.0.1.weight` | `[192, 16, 1, 1]` | grouped 1x1，`groups=12`，每组 16 输入通道到 16 输出通道 |

当前 ONNX 中 aggregation 无 bias。

---

## 4. 关键决策

### D1：新增 Plugin 类型，不扩展 P1a 类型

P1b 使用新的 op type：

```text
EdgesegAggregationReluLinearAttention_TRT
```

不把现有 `EdgesegReluLinearAttention_TRT` 扩展成多模式 Plugin。原因是 P1a 已有完整验证结果和可复现实验链，P1b 的权重、输入数量、替换边界、序列化字段都会改变。拆成新类型可以保留 P1a 的 ABI 和历史结果，降低回归风险。

### D2：同一 DLL / CMake target，但 CUDA 源文件分开

P1b 复用当前 `edgeseg_relu_linear_attention_plugin.dll` 的构建、加载和注册路径，但新增独立 Creator 与独立 C++ 类。这样减少 Windows DLL、TensorRT registry、Python loader 的变量，同时保持 Plugin 类型隔离。

CUDA 实现层面不直接写进 P1a 的 `relu_linear_attention_kernel.cu`。P1a 已经有独立验证结果、性能历史和 Nsight attribution；P1b 的 aggregation workspace、depthwise/grouped pointwise kernel 和后续优化路线都会分叉，因此 P1b 使用独立 CUDA 文件：

```text
phase3/plugin/src/aggregation_relu_linear_attention_kernel.cu
```

该文件负责 P1b 的 aggregation、cat workspace 和 P1a attention launcher 调用；P1a 文件继续只维护 `relu_linear_att-only`。

### D3：先 parser/build，再 block correctness，再端到端

P1b 的风险分三层：

1. TensorRT ONNX parser 是否能稳定创建一个带权重输入的 Plugin node。
2. P1b CUDA 数学是否与 PyTorch block-local reference 对齐。
3. 真实 EfficientViT P1b engine 是否在端到端 correctness / latency / Nsight attribution 上有收益。

当前已完成第 1 层、第 2 层、端到端 correctness / latency、Nsight attribution 和 P1b-1 fused aggregation+cat 优化。性能结论是不采纳 naive P1b；P1b fused 可作为继续优化候选，但还不能替代 P1a 主线。

### D4：权重 initializer 作为 Plugin 输入

P1b Plugin node 有 3 个输入：

```text
input0: qkv runtime tensor [1,192,64,128]
input1: aggregation depthwise weight initializer [192,1,5,5]
input2: aggregation grouped pointwise weight initializer [192,16,1,1]
```

这样权重仍保留在 ONNX initializer 中，来源、shape、hash 更容易追溯，也避免把几千个 float 手写进 PluginField attribute。

### D5：Graph surgery 边界

P1b surgery 的替换边界是：

```text
qkv/conv/Conv_output_0 -> Cast_1_output_0
```

被移除的子图包括：

- aggregation depthwise Conv；
- aggregation grouped 1x1 Conv；
- Concat；
- P1a 已替换过的 `relu_linear_att` 子图。

保留在 Plugin 外部的节点包括：

- qkv Conv；
- proj Conv；
- residual Add；
- 其它 stage0/stage1/stage3/head 标准算子。

---

## 5. 验证顺序

P1b 按下面顺序推进：

1. **设计与 contract 落盘**：本文即本步产物。
2. **P1b parser toy / skeleton**：新增 Plugin 类型，先用 zero-fill 验证 parser/build，不宣称数值正确。
3. **真实 ONNX surgery build smoke**：替换两个 stage2 context 的 P1b 边界，确认 TensorRT engine 可构建。
4. **单 block 数值验证**：用 PyTorch LiteMLA 子模块输出作为 reference，验证 P1b Plugin 的 aggregation + attention 数学。
5. **端到端 correctness / latency**：复用 Phase 2/3 benchmark 口径，与 TensorRT FP32 baseline 和 P1a engine 对比。
6. **Nsight attribution**：若继续投入 P1b，再比较 `aggregation + attention` proxy 在 P1b 后的 kernel time / launch 数变化。

---

## 6. 已知风险

| 风险 | 说明 | 应对 |
|---|---|---|
| TensorRT parser 不接受 initializer 作为 Plugin 输入 | 3-input Plugin 是最干净的权重输入方式，但 TensorRT 8.6.1 parser 行为必须实测 | 已通过 toy parser/build 和真实 graph build smoke；若后续版本变化，fallback 到 PluginField / serialized weights |
| aggregation 权重布局弄错 | depthwise 5x5 与 grouped 1x1 的 group 语义不同，不能按普通 dense Conv 处理 | block-level reference 验证必须覆盖 aggregation 输出与最终 attention 输出 |
| P1b 可能比 P1a 慢 | aggregation 是标准 Conv 路径，TensorRT/cuDNN 已有优化；naive 自写 aggregation 可能抵消 attention 收益 | naive 版已确认退化；fused aggregation+cat 后已打赢 Phase 2 中段 baseline，但仍未达到 P1a proxy 水平 |
| 两个 block 权重不同 | block1/block2 shape 相同但数值不同 | surgery 与 toy engine 必须逐 block 绑定权重 |
| P1b 不等于 P1c | qkv/proj/residual 仍在 Plugin 外 | 文档和报告中继续区分 P1b、整体 LiteMLA、整网端到端收益 |

---

## 7. 当前文件

P1b 相关产物：

- `phase3/plugin/include/edgeseg_aggregation_relu_linear_attention_plugin.h`
- `phase3/plugin/src/edgeseg_aggregation_relu_linear_attention_plugin.cpp`
- `phase3/plugin/src/aggregation_relu_linear_attention_kernel.cu`
- `phase3/scripts/build_p1b_plugin_toy_engine.py`
- `phase3/scripts/integrate_p1b_aggregation_attention_plugin_onnx.py`
- `phase3/scripts/build_p1b_plugin_engine.py`
- `phase3/scripts/capture_p1b_stage2_reference.py`
- `phase3/scripts/validate_p1b_aggregation_attention_plugin.py`
- `phase3/scripts/benchmark_plugin_engine.py`
- `phase3/results/metrics/p1b_aggregation_attention_toy_build.json`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_engine_build.json`
- `phase3/results/metrics/p1b_stage2_reference_capture.json`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_validation.json`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_engine_benchmark.json`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_nsys.json`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.md`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.json`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_fused_engine_benchmark_nsys.json`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.md`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.json`

当前 P1b 验收口径已经从 parser/build feasibility 推进到 block-level Plugin correctness、真实 engine 端到端 cold-run benchmark、Nsight attribution 和 fused aggregation+cat 优化。结论是：P1b 边界本身有价值；naive aggregation 版本不采纳，P1b-1 fused 版本可继续作为候选，但需要进一步优化才能挑战 P1a。

---

## 8. Parser Toy 验证结果

P1b skeleton / parser toy 已通过：

| 项 | 结果 |
|---|---|
| Plugin op type | `EdgesegAggregationReluLinearAttention_TRT` |
| Plugin namespace / version | `edgeseg` / `1` |
| Plugin DLL | `phase3/plugin/build/edgeseg_relu_linear_attention_plugin.dll` |
| Toy ONNX | `phase3/results/onnx/p1b_aggregation_attention_toy.onnx` |
| Toy engine | `phase3/results/engines/p1b_aggregation_attention_toy_fp32.engine` |
| Metadata | [`../results/metrics/p1b_aggregation_attention_toy_build.json`](../results/metrics/p1b_aggregation_attention_toy_build.json) |
| TensorRT parser | 通过，`parser_errors=[]` |
| TensorRT network IO | input `qkv [1,192,64,128]`，output `attention_out [1,128,64,128]` |
| Network layer count | `3` |

解释：

- ONNX checker 对自定义 op 报 “No Op registered” 是预期 warning；本步以 TensorRT parser/build 为权威检查。
- TensorRT 日志显示 `Successfully created plugin: EdgesegAggregationReluLinearAttention_TRT`。
- TensorRT build 日志显示网络只有 `1 inputs and 1 output network tensors`，说明两个 aggregation weight initializer 没有变成外部 runtime binding。
- Parser toy 阶段的 skeleton `enqueue()` 只 zero-fill 输出，用于验证 Plugin 创建、序列化、shape 和 parser/build 路径；后续第一版 CUDA 数学正确性见本文 §11。

---

## 9. 真实 ONNX Surgery / Engine Build Smoke

真实 EfficientViT P1b graph surgery 与 TensorRT build smoke 已通过：

| 项 | 结果 |
|---|---|
| ONNX surgery script | [`../scripts/integrate_p1b_aggregation_attention_plugin_onnx.py`](../scripts/integrate_p1b_aggregation_attention_plugin_onnx.py) |
| Engine build script | [`../scripts/build_p1b_plugin_engine.py`](../scripts/build_p1b_plugin_engine.py) |
| Integration metadata | [`../results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json`](../results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json) |
| Engine build metadata | [`../results/metrics/p1b_aggregation_attention_plugin_engine_build.json`](../results/metrics/p1b_aggregation_attention_plugin_engine_build.json) |
| Patched graph | `393 -> 256` nodes |
| Plugin node count | `2` |
| Removed subgraph | block1: `58` nodes，block2: `81` nodes |
| Plugin inputs | `qkv` runtime tensor + depthwise weight initializer + grouped pointwise weight initializer |
| TensorRT parser | 通过，`parser_errors=[]` |
| TensorRT network IO | input `input [1,3,1024,2048]`，output `segout [1,19,128,256]` |
| TensorRT layer count | `239` |
| Engine size | `3,544,244` bytes |

解释：

- 真实 ONNX surgery 删除了每个目标 block 中从 aggregation depthwise Conv 到 `Cast_1_output_0` 的子图，并用 P1b Plugin node 替换。
- 两个 aggregation 权重仍作为 initializer 输入进入 Plugin node：`aggreg.0.0.weight [192,1,5,5]` 与 `aggreg.0.1.weight [192,16,1,1]`。
- TensorRT 日志显示两个 P1b Plugin node 都被成功创建。
- 当前 engine 已用第一版 P1b CUDA 数学路径重建；端到端结果见本文 §12。

---

## 10. 单 Block Reference 捕获结果

P1b 单 block reference 捕获已通过：

| 项 | 结果 |
|---|---|
| Capture script | [`../scripts/capture_p1b_stage2_reference.py`](../scripts/capture_p1b_stage2_reference.py) |
| Metadata | [`../results/metrics/p1b_stage2_reference_capture.json`](../results/metrics/p1b_stage2_reference_capture.json) |
| Tensor bundle | `phase3/results/tensors/p1b_stage2_reference_capture.npz`（本地大文件，不入 git） |
| 输入 | `phase1/data/city_asset_cityscapes_like.png`，固定 resize 到 `1024x2048` |
| 权重 | `efficientvit_seg_b0_cityscapes.pt`，sha256=`923d6fdd5e93640cc0c2f3f213764f34e80b477cd98a6b294d870ea6df5acc50` |
| 目标模块 | `backbone.stages.2.op_list.1.context_module.main`、`backbone.stages.2.op_list.2.context_module.main` |
| P1b runtime input | `qkv [1,192,64,128]` |
| P1b output reference | `attention_out [1,128,64,128]` |
| depthwise weight | `[192,1,5,5]`，`groups=192` |
| pointwise weight | `[192,16,1,1]`，`groups=12` |
| projection sanity check | 两个 block 的 `module.proj(attention_out)` 均与原模块输出 `allclose(atol=1e-5, rtol=1e-5)` |

解释：

- 该结果证明 P1b 替换边界、aggregation 权重布局和 PyTorch block-level reference 可复现。
- 该结果本身只证明 reference 可用；后续第一版 P1b CUDA 数学验证结果见本文 §11。

---

## 11. P1b 第一版 CUDA 数学验证结果

P1b 第一版 CUDA 数学路径已独立落盘在：

```text
phase3/plugin/src/aggregation_relu_linear_attention_kernel.cu
```

它没有把 P1b 逻辑写回 P1a 的 `relu_linear_attention_kernel.cu`，而是采用分文件维护：

- P1a 文件继续只维护 `relu_linear_att-only`。
- P1b 文件负责 `depthwise 5x5 -> grouped pointwise 1x1 -> cat workspace -> P1a attention launcher`。
- 两者共用同一个 DLL / CMake target，降低 TensorRT registry 与 Windows DLL 加载变量。

验证脚本：

```text
phase3/scripts/validate_p1b_aggregation_attention_plugin.py
```

验证结果：

| 项 | block1 | block2 |
|---|---:|---:|
| `max_abs_diff` | `1.311302e-06` | `2.384186e-06` |
| `mean_abs_diff` | `9.504385e-08` | `7.170572e-08` |
| `cosine_similarity` | `0.9999999999996935` | `0.9999999999997492` |
| `allclose(atol=1e-3, rtol=1e-3)` | `true` | `true` |
| `argmax_channel_agreement` | `1.0` | `1.0` |

Metadata：

```text
phase3/results/metrics/p1b_aggregation_attention_plugin_validation.json
```

解释：

- 这一步证明两个真实 `stage2/context` block 的 P1b Plugin 输出与 PyTorch `attention_out` reference 对齐。
- 这一步仍是 block-local toy/plugin correctness，不代表真实 EfficientViT P1b engine 的端到端 correctness。
- 第一版实现以正确性优先，使用 TensorRT workspace 暂存 depthwise 输出、cat 后 attention input 与 P1a VK workspace；它不是最终性能优化版。

---

## 12. 真实 P1b Engine 冷机 Benchmark 结果

使用第一版 P1b CUDA 数学路径重建真实 EfficientViT P1b engine 后，完成冷机端到端 benchmark：

| 项 | 结果 |
|---|---|
| Engine build metadata | [`../results/metrics/p1b_aggregation_attention_plugin_engine_build.json`](../results/metrics/p1b_aggregation_attention_plugin_engine_build.json) |
| Benchmark metadata | [`../results/metrics/p1b_aggregation_attention_plugin_engine_benchmark.json`](../results/metrics/p1b_aggregation_attention_plugin_engine_benchmark.json) |
| Benchmark summary | [`../results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md) |
| Protocol | `warmup=20`、`measure=100`、CUDA Events execute-only |
| Baseline TRT p50 | `54.4532 ms` |
| P1b Plugin TRT p50 | `56.3395 ms` |
| p50 delta | `+1.8862 ms` |
| p50 speedup | `0.9665x` |
| Baseline TRT mean | `54.4771 ms` |
| P1b Plugin TRT mean | `56.7579 ms` |
| mean speedup | `0.9598x` |

Correctness：

| Comparison | 结果 |
|---|---|
| Plugin TRT vs Baseline TRT | `allclose=True`、`max_abs_diff=4.4346e-05`、argmax agreement `1.0` |
| Plugin TRT vs PyTorch | strict `1e-4` allclose 未通过，relaxed `1e-3` allclose 通过，argmax agreement `1.0` |

解释：

- 冷机结果排除了上一轮 hot run 中 300ms 级 outlier 的主要干扰。
- 第一版 P1b 在数学上正确，并且真实 engine 端到端输出保持可接受对齐。
- 但第一版 P1b 端到端性能慢于 Phase 2 TensorRT FP32 baseline，说明“把 aggregation 也放入 Plugin”在 naive 实现下会破坏 TensorRT/cuDNN 对标准 Conv 路径的优化收益。
- 因此 P1b 当前结论是 `correctness passed, performance rejected for first implementation`。Nsight attribution 已经进一步确认退化来源，见本文 §13；后续不能继续沿用当前 naive aggregation 实现扩大 fusion 边界。

---

## 13. P1b Nsight Attribution 结果

P1b Plugin engine 已完成正式 Nsight Systems 采集和 SQLite correlationId 归因：

| 项 | 结果 |
|---|---|
| Benchmark metadata | [`../results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_nsys.json`](../results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_nsys.json) |
| Attribution summary | [`../results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.md) |
| Attribution JSON | [`../results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.json`](../results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.json) |
| Protocol | `warmup=20`、`measure=100`、CUDA/NVTX trace |
| CUDA Events latency mean / p50 | `56.180 ms` / `56.124 ms` |
| `trt/execute` kernel avg | `54.902 ms / iter` |
| `trt/execute` launches | `151.0 / iter` |

Stage2 context attribution：

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---:|---:|---:|
| `p1b_aggregation_attention_plugin` | `6.189` | `11.27%` | `10.0` |
| `qkv` | `0.548` | `1.00%` | `2.0` |
| `proj_add` | `0.399` | `0.73%` | `2.0` |

Plugin 内部 kernel attribution：

| Kernel | Avg ms / iter | Plugin 内占比 | Count |
|---|---:|---:|---:|
| `depthwise5x5Kernel` | `2.462` | `39.77%` | `200` |
| `groupedPointwise1x1Kernel` | `2.427` | `39.22%` | `200` |
| `computeVkKernelDim16WarpD4` | `0.949` | `15.33%` | `200` |
| `computeOutputKernelDim16` | `0.351` | `5.67%` | `200` |

与 Phase 2 TensorRT baseline 对比：

| Boundary | Before | After | Speedup | 解释 |
|---|---:|---:|---:|---|
| `attention_core -> p1b_plugin_only` | `3.689 ms / 12 launches` | `6.189 ms / 10 launches` | `0.596x` | P1b plugin 吸收了 aggregation，因此不能只按 attention-only 解释 |
| `aggregation + attention_core -> p1b_plugin_boundary` | `5.443 ms / 38 launches` | `6.189 ms / 10 launches` | `0.879x` | launch 数显著减少，但 kernel time 变差 |
| `stage2_context_total` | `6.383 ms / 42 launches` | `7.136 ms / 14 launches` | `0.894x` | P1b 让 stage2 context 总耗时退化 |

解释：

- P1b 证明了“扩大边界可以减少 launch 数”：`aggregation + attention_core` proxy 从 `38 launches/iter` 降到 `10 launches/iter`。
- 但 P1b 没有证明“扩大边界可以提速”：kernel time 从 `5.443ms/iter` 增加到 `6.189ms/iter`。
- 退化主因不是 P1a attention 数学。`computeVk + computeOutput` 合计约 `1.300ms/iter`，与 P1a 路径量级一致。
- 退化主因是自写 aggregation：`depthwise5x5Kernel + groupedPointwise1x1Kernel` 合计约 `4.889ms/iter`，占 P1b Plugin kernel time 约 `79%`。这验证了本文风险表中的判断：aggregation 是标准 Conv 路径，TensorRT/cuDNN 已经有更成熟的实现，naive CUDA 复刻会抵消甚至反噬 fusion 收益。

采集纪律：

- Codex 沙盒内运行 `nsys profile` 曾出现 75s 超时；同一命令在提权执行后约 22s 完成，并成功生成 `.nsys-rep`、`.sqlite` 和 attribution summary。因此这次异常判断为沙盒/权限环境问题，不是 benchmark 脚本或 Plugin engine 问题。
- Windows 普通权限下 CPU sampling / context switch trace 会被禁用；本次结论只依赖 CUDA/NVTX trace 与 SQLite correlationId 归因，不依赖 CPU sampling。

后续判断（naive 版）：

- 当前 naive P1b 第一版到此应停止作为性能主线；它的价值是证明 graph surgery、权重输入、端到端 correctness 和退化诊断闭环。
- 若继续 P1b，必须先重写 aggregation kernel 或改变边界设计。P1b-1 fused aggregation+cat 正是这一判断的后续实现，结果见本文 §14。

---

## 14. P1b-1 Fused Aggregation + Cat 结果

根据 §13 的退化定位，P1b-1 将原来的三段路径：

```text
cudaMemcpyAsync(qkv -> attentionInput[0:192])
depthwise5x5Kernel(qkv -> depthwiseWorkspace)
groupedPointwise1x1Kernel(depthwiseWorkspace -> attentionInput[192:384])
```

改为单个 fused kernel：

```text
fusedAggregationCatKernel(qkv, depthwiseWeight, pointwiseWeight -> attentionInput[0:384])
```

设计目的：

- 去掉 `depthwiseWorkspace` 的 global write/read。
- 去掉 `depthwise5x5Kernel` 与 `groupedPointwise1x1Kernel` 之间的一次 launch。
- 把原始 `qkv` 的 cat 前半部分写入合并进同一个 kernel，去掉独立 D2D copy。
- 保留后续 P1a attention launcher 不变，因此输出 contract、Plugin ABI、op type 和真实 engine tensor contract 都不变。

实现文件：

```text
phase3/plugin/src/aggregation_relu_linear_attention_kernel.cu
```

Workspace 变化：

| 版本 | Workspace 内容 |
|---|---|
| naive P1b | `depthwiseWorkspace + attentionInput + vkWorkspace` |
| P1b-1 fused | `attentionInput + vkWorkspace` |

Block-level correctness：

| 项 | 结果 |
|---|---|
| Validation metadata | [`../results/metrics/p1b_aggregation_attention_plugin_validation.json`](../results/metrics/p1b_aggregation_attention_plugin_validation.json) |
| Overall pass | `true` |
| 验证对象 | 两个真实 `stage2/context` block |
| 口径 | Plugin output vs PyTorch `attention_out` reference |

端到端 benchmark：

| 项 | 结果 |
|---|---:|
| Benchmark summary | [`../results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md) |
| Engine build metadata | [`../results/metrics/p1b_aggregation_attention_plugin_engine_build.json`](../results/metrics/p1b_aggregation_attention_plugin_engine_build.json) |
| Baseline TRT p50 | `54.3314 ms` |
| P1b fused Plugin TRT p50 | `53.6100 ms` |
| p50 delta | `-0.7214 ms` |
| p50 speedup | `1.0135x` |
| Baseline TRT mean | `54.3386 ms` |
| P1b fused Plugin TRT mean | `53.6366 ms` |
| mean speedup | `1.0131x` |
| Plugin TRT vs baseline TRT | `allclose=True`、argmax agreement `1.0` |

Nsight attribution：

| 项 | 结果 |
|---|---:|
| Attribution summary | [`../results/metrics/p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.md) |
| P1b fused Plugin layer | `4.848 ms / iter` |
| P1b fused Plugin launches | `6 launches / iter` |
| `fusedAggregationCatKernel` | `3.536 ms / iter` |
| `computeVkKernelDim16WarpD4` | `0.960 ms / iter` |
| `computeOutputKernelDim16` | `0.352 ms / iter` |
| stage2/context total | `5.792 ms / iter`、`10 launches / iter` |

与 Phase 2 TensorRT baseline 对比：

| Boundary | Before | After | Speedup |
|---|---:|---:|---:|
| `aggregation + attention_core -> p1b_fused_plugin_boundary` | `5.443 ms / 38 launches` | `4.848 ms / 6 launches` | `1.123x` |
| `stage2_context_total` | `6.383 ms / 42 launches` | `5.792 ms / 10 launches` | `1.102x` |

解释：

- P1b-1 修正了 naive P1b 的主要问题：不再把 depthwise 中间结果写入 global workspace 后再读回。
- P1b-1 已经证明“扩大边界”本身有性能价值：它打赢了 Phase 2 TensorRT baseline 的 `aggregation + attention_core` 中段 proxy。
- 但 P1b-1 仍未达到 P1a 路径的 `aggregation + plugin` proxy 约 `3.062ms/iter`。当前 `fusedAggregationCatKernel` 仍有 `3.536ms/iter`，是后续 P1b 优化的主要对象。
- 本轮已用真实 rebuild 后的 P1b fused engine 复测确认，engine sha256 为 `dcba4c1d10e692f4922c9b1332cadcadfc0055371e4e945a1216e567a1d2e945`。真实 engine rebuild 耗时约 `342s`；此前 `180s` timeout 不足导致误判为卡住。

---

## 15. P1b-2：Grouped Pointwise 权重 Shared Cache

P1b-2 保留 P1b-1 的 fused aggregation+cat 边界，只修改 `fusedAggregationCatKernel` 内部的 grouped pointwise 权重读取方式。

### 15.1 改动

P1b-1 中，每个空间线程在计算当前 group 的 16 个 grouped pointwise 输出时，会反复从 global memory 读取同一组 `16x16` pointwise 权重。P1b-2 改为：

```text
每个 CTA 对应一个 aggregation group 和一段 spatial tile
  -> CTA 内线程协作加载当前 group 的 16x16 pointwise weight 到 shared memory
  -> __syncthreads()
  -> 每个空间线程复用 shared-memory weight 计算 grouped pointwise 输出
```

该改动不改变 tensor contract、Plugin op type、workspace contract 或数学公式。

### 15.2 验证结果

| 项 | 结果 |
|---|---:|
| Block-level validation | [`../results/metrics/p1b_aggregation_attention_plugin_validation.json`](../results/metrics/p1b_aggregation_attention_plugin_validation.json) |
| Overall pass | `true` |
| Benchmark summary | [`../results/metrics/p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_summary.md) |
| Nsight attribution | [`../results/metrics/p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.md) |
| Baseline TRT p50 | `54.306 ms` |
| P1b-2 Plugin TRT p50 | `53.530 ms` |
| p50 speedup | `1.0145x` |
| Plugin TRT vs baseline TRT | `allclose=True`、argmax agreement `1.0` |

Nsight attribution：

| 项 | P1b-1 | P1b-2 | 变化 |
|---|---:|---:|---:|
| P1b Plugin layer | `4.848 ms/iter` | `4.347 ms/iter` | `-0.501 ms` |
| `fusedAggregationCatKernel` | `3.536 ms/iter` | `3.038 ms/iter` | `-0.498 ms` |
| `computeVkKernelDim16WarpD4` | `0.960 ms/iter` | `0.958 ms/iter` | 基本不变 |
| `computeOutputKernelDim16` | `0.352 ms/iter` | `0.352 ms/iter` | 基本不变 |
| stage2/context total | `5.792 ms/iter` | `5.302 ms/iter` | `-0.490 ms` |

与 Phase 2 TensorRT baseline 对比：

| Boundary | Phase 2 baseline | P1b-2 | Speedup |
|---|---:|---:|---:|
| `aggregation + attention_core` / `p1b_weight_shared_plugin_boundary` | `5.443 ms / 38 launches` | `4.347 ms / 6 launches` | `1.252x` |
| `stage2_context_total` | `6.383 ms / 42 launches` | `5.302 ms / 10 launches` | `1.204x` |

### 15.3 判断

P1b-2 是有效的小步优化：它证明当前 `fusedAggregationCatKernel` 内部确实存在可消除的 pointwise weight 重复读取。与此同时，它没有改变 P1b 的大结论：

- P1b 中段边界已经可以打赢 Phase 2 TensorRT baseline 的 `aggregation + attention_core` proxy。
- P1b 仍未明显优于 P1a `aggregation + plugin` proxy，因此不能简单替代 P1a 主线。
- 如果继续 P1b，下一步仍应聚焦 `fusedAggregationCatKernel`，例如 depthwise 5x5 tile/halo、interior/border 拆分或更细线程映射。

测量纪律：P1b-2 第一次热机 benchmark 曾显示 baseline p50 `54.585 ms`、Plugin p50 `59.144 ms`，冷机重测后转为正收益。因此该热机样本只作为温度/频率敏感性的证据，不作为性能结论。

完整 P1a/P1b 优化演进统一记录在 [`plugin_kernel_optimization_history.md`](plugin_kernel_optimization_history.md)。

---

## 16. P1b-3 Probe：Interior Fast Path（不采纳）

P1b-3 probe 尝试在同一个 `fusedAggregationCatKernel` 内给非边界像素增加 depthwise 5x5 fast path：

```text
if pixel is interior:
  depthwise 5x5 without ih/iw boundary checks
else:
  original safe boundary path
```

该方案不增加 kernel launch，不改变 ABI / workspace / tensor contract，风险较低。但冷机 benchmark 显示它没有收益：

| 项 | 结果 |
|---|---:|
| Benchmark summary | [`../results/metrics/p1b_aggregation_attention_plugin_interior_fastpath_engine_benchmark_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_interior_fastpath_engine_benchmark_summary.md) |
| Baseline TRT p50 | `54.312 ms` |
| P1b-3 probe p50 | `54.710 ms` |
| p50 speedup | `0.9927x` |
| Plugin TRT vs baseline TRT | `allclose=True`、argmax agreement `1.0` |

判断：

- 当前 P1b aggregation kernel 的主要成本不在 depthwise 边界判断。
- interior fast path 可能引入额外代码体积、分支分流或寄存器/指令压力，抵消了去掉越界判断的收益。
- 因此 P1b-3 probe 记录为 `evaluated, not adopted`；主线 CUDA 代码已恢复到 P1b-2 shared weight cache 版本。
