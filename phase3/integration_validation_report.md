# Phase 3 Integration Validation Report

> **报告目的**：验证 Phase 3 TensorRT Plugin 是否已经完成真实 EfficientViT-Seg-B0 图集成、数值对齐、端到端性能验证、Nsight runtime attribution 和 Cityscapes mIoU accuracy gate，并给出 Phase 3 最终采用边界。
>
> **最终结论**：Phase 3 当前主交付线采用 **P1a `relu_linear_att-only` Plugin，覆盖 stage2+stage3 四个 LiteMLA context block**。该方案相对 Phase 2 TensorRT FP32 baseline 在 execute-only latency 上取得约 `1.07x` 端到端 p50 speedup，并通过 Cityscapes val mIoU gate。P1b 与 P1mix 保留为重要消融与后续候选，但不作为当前主线。

---

## 1. 验收摘要

| 验收项 | 结果 | 证据 |
|---|---:|---|
| ONNX graph surgery | 通过 | [`relu_linear_attention_plugin_stage2_stage3_onnx_integration.json`](results/metrics/relu_linear_attention_plugin_stage2_stage3_onnx_integration.json) |
| TensorRT Plugin engine build | 通过 | [`relu_linear_attention_plugin_stage2_stage3_engine_build.json`](results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_build.json) |
| Baseline TRT vs Plugin TRT correctness | 通过 | [`relu_linear_attention_plugin_stage2_stage3_engine_benchmark_summary.md`](results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_benchmark_summary.md) |
| Execute-only latency | 通过 | p50 `54.3995 ms -> 50.8380 ms`，speedup `1.0701x` |
| Nsight runtime attribution | 通过 | [`relu_linear_attention_plugin_stage2_stage3_nsys_attribution_summary.md`](results/metrics/relu_linear_attention_plugin_stage2_stage3_nsys_attribution_summary.md) |
| `relu_linear_att-only` 子路径加速 | 通过 | stage2 proxy `3.689 ms -> 1.309 ms`，speedup `2.819x` |
| Cityscapes val mIoU gate | 通过 | [`cityscapes_miou_p1a_stage2_stage3_summary.md`](results/metrics/cityscapes_miou_p1a_stage2_stage3_summary.md) |
| P1b / P1mix 分支决策 | 完成 | P1b/P1mix 均未作为当前主线采纳 |

---

## 2. 验证范围

本报告验证的主线 Plugin 是：

```text
EdgesegReluLinearAttention_TRT
op type: EdgesegReluLinearAttention_TRT
plugin namespace: edgeseg
plugin version: 1
precision: FP32
target scope: stage2-stage3
```

替换范围为四个 LiteMLA context block：

| Block | Plugin input shape contract |
|---|---|
| `/backbone/stages.2/op_list.1/context_module/main` | `input_c=384, height=64, width=128, dim=16` |
| `/backbone/stages.2/op_list.2/context_module/main` | `input_c=384, height=64, width=128, dim=16` |
| `/backbone/stages.3/op_list.1/context_module/main` | `input_c=768, height=32, width=64, dim=16` |
| `/backbone/stages.3/op_list.2/context_module/main` | `input_c=768, height=32, width=64, dim=16` |

边界说明：

- Plugin 只替换 `relu_linear_att-only` 子路径。
- `qkv` Conv、`aggregation`、`proj` Conv、residual add 仍保留在 TensorRT 标准路径中。
- 该边界与 [`stage2_context_tensor_contract.md`](design_notes/stage2_context_tensor_contract.md) 中的 P1a contract 一致。

---

## 3. 主要产物索引

| 类别 | 文件 |
|---|---|
| Plugin CUDA 主实现 | [`plugin/src/relu_linear_attention_kernel.cu`](plugin/src/relu_linear_attention_kernel.cu) |
| Plugin C++ 接口 | [`plugin/include/edgeseg_relu_linear_attention_plugin.h`](plugin/include/edgeseg_relu_linear_attention_plugin.h), [`plugin/src/edgeseg_relu_linear_attention_plugin.cpp`](plugin/src/edgeseg_relu_linear_attention_plugin.cpp) |
| ONNX 集成脚本 | [`scripts/integrate_relu_linear_attention_plugin_onnx.py`](scripts/integrate_relu_linear_attention_plugin_onnx.py) |
| Engine 构建脚本 | [`scripts/build_plugin_engine.py`](scripts/build_plugin_engine.py) |
| Engine benchmark 脚本 | [`scripts/benchmark_plugin_engine.py`](scripts/benchmark_plugin_engine.py) |
| Nsight attribution 脚本 | [`scripts/analyze_plugin_nsys_attribution.py`](scripts/analyze_plugin_nsys_attribution.py) |
| Cityscapes mIoU 脚本 | [`scripts/evaluate_cityscapes_miou.py`](scripts/evaluate_cityscapes_miou.py) |
| Kernel 优化历史 | [`design_notes/plugin_kernel_optimization_history.md`](design_notes/plugin_kernel_optimization_history.md) |
| Phase 3 设计文档索引 | [`design_notes/README.md`](design_notes/README.md) |
| Phase 3 脚本索引 | [`scripts/README.md`](scripts/README.md) |

---

## 4. 集成路径验证

P1a stage2+stage3 的 ONNX surgery 结果显示，四个 LiteMLA context block 均被替换为 `EdgesegReluLinearAttention_TRT` Plugin node。对应 JSON 记录在 [`relu_linear_attention_plugin_stage2_stage3_onnx_integration.json`](results/metrics/relu_linear_attention_plugin_stage2_stage3_onnx_integration.json)。

集成后的 TensorRT engine 构建结果记录在 [`relu_linear_attention_plugin_stage2_stage3_engine_build.json`](results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_build.json)。该结果证明：

- TensorRT parser 能识别并保留自定义 Plugin node。
- Plugin shared library 可被加载。
- FP32 engine 可成功 build。
- Plugin engine 与 Phase 2 baseline engine 可进入同一 benchmark 对照流程。

---

## 5. Correctness Validation

端到端 benchmark 同时比较了 Plugin TRT 与 Phase 2 TensorRT FP32 baseline / PyTorch reference：

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | `True` | `4.48227e-05` | `5.64205e-06` | `1.000000` |
| Plugin TRT vs PyTorch | `False` | `2.78473e-04` | `2.51554e-05` | `1.000000` |

解释：

- **正式 correctness gate 以 Plugin TRT vs Baseline TRT 为主**，因为 Phase 3 是在 Phase 2 TensorRT FP32 baseline 上替换 Plugin。
- Plugin TRT vs PyTorch 的 relaxed allclose 未完全通过，但 argmax agreement 为 `1.0`，且该差异与 Phase 2 ONNX/TensorRT 导出链路中的数值差异口径一致，不构成语义回归证据。
- 最终语义层面由 Cityscapes val mIoU gate 再次确认。

---

## 6. Latency Validation

Phase 3 延续 Phase 2 的 execute-only latency 口径：

- CUDA Events 计时。
- `warmup=20 / measure=100`。
- 不包含 preprocessing、H2D/D2H、postprocess。
- 不用 NVTX range duration 作为组件耗时。

P1a stage2+stage3 benchmark：

| Item | Baseline TRT | Plugin TRT | Delta | Speedup |
|---|---:|---:|---:|---:|
| p50 | `54.3995 ms` | `50.8380 ms` | `-3.5615 ms` | `1.0701x` |
| mean | `54.4019 ms` | `50.9334 ms` | `-3.4684 ms` | `1.0681x` |

结论：

- P1a stage2+stage3 在当前 MX250 / TensorRT 8.6.1 / FP32 口径下取得稳定正收益。
- 该收益是端到端 execute-only 的整体收益，已经包含了未替换模块对总时间的稀释。

---

## 7. Nsight Runtime Attribution

Nsight attribution 使用：

```text
TensorRT/NVTX layer range
-> CUDA runtime launch inside range
-> CUDA kernel with same correlationId
```

也就是说，组件耗时来自 CUDA runtime/kernel `correlationId` 归因后的实际 kernel duration，而不是 NVTX range 的 `end-start`。

P1a stage2+stage3 Plugin engine 的 Nsight 摘要：

| Metric | Value |
|---|---:|
| CUDA Events latency mean / p50 | `51.448 ms / 51.416 ms` |
| `trt/execute` kernel avg | `50.680 ms / iter` |
| `trt/execute` launches | `163.0 / iter` |
| Layer attribution / execute kernel time | `100.00%` |

Plugin context detail：

| Component | Avg kernel ms / iter | Launches / iter |
|---|---:|---:|
| `aggregation` | `2.719` | `76.0` |
| `relu_linear_att_plugin` | `1.950` | `8.0` |
| `qkv` | `1.055` | `4.0` |
| `proj_add` | `0.712` | `4.0` |

Plugin kernel 内部：

| Kernel | Avg ms / iter | Share |
|---|---:|---:|
| `computeVkKernelDim16WarpD4` | `1.418` | `72.71%` |
| `computeOutputKernelDim16` | `0.532` | `27.29%` |

这说明 P1a 最终瓶颈仍主要在 `computeVk` 跨 `N` 归约阶段；`computeOutput` 阶段经过 shared-memory VK cache 与 `dim=16` 专用路径后已经不是主瓶颈。

---

## 8. `relu_linear_att-only` 子路径加速比

需要特别区分两种加速比：

1. **端到端 speedup**：整网 TensorRT engine 的 execute-only latency，从 `54.3995 ms` 到 `50.8380 ms`，为 `1.0701x`。
2. **`relu_linear_att-only` 子路径 speedup**：只比较原始 TensorRT 中对应 `relu_linear_att` 语义的 residual runtime proxy 与 P1a Plugin runtime。

原始 TensorRT 里没有一个单独叫 `relu_linear_att` 的 layer。Phase 2 attribution 中使用的等价 proxy 是：

```text
attention_core = relu_qk + pad + matmul + norm_add_div
```

P1a 替换后的对应边界是：

```text
relu_linear_att_plugin = EdgesegReluLinearAttention_TRT
```

已落盘的 stage2 口径为：

| Boundary | Before | After | Delta | Speedup | Launches |
|---|---:|---:|---:|---:|---:|
| stage2 `attention_core -> relu_linear_att_plugin` | `3.689 ms` | `1.309 ms` | `-2.380 ms` | `2.819x` | `12 -> 4` |

如果将 stage3 按同一 layer-mapping 规则纳入，stage2+stage3 的近似子路径口径为：

| Boundary | Before | After | Speedup | Launches |
|---|---:|---:|---:|---:|
| stage2+stage3 `attention_core -> relu_linear_att_plugin` | `~5.508 ms` | `1.950 ms` | `~2.82x` | `24 -> 8` |

报告采用保守表述：

- **正式可引用值**：stage2 `relu_linear_att` proxy speedup = **`2.819x`**。
- **扩展估算值**：stage2+stage3 同规则 proxy speedup 约 **`2.82x`**。
- 这些都是 Nsight SQLite attribution 的 kernel-time proxy，不是端到端 latency speedup。

---

## 9. Cityscapes mIoU Validation

P1a stage2+stage3 通过 Cityscapes val 500 张图的 mIoU gate：

| Engine | mIoU |
|---|---:|
| Baseline TRT FP32 | `75.6463126%` |
| Plugin TRT FP32 | `~75.6463248%` |
| Delta | `+0.0000123` percentage point |
| Argmax agreement on valid labels | `0.999999918` |
| Argmax mismatch pixels | `75 / 917018489` |

结论：

- Plugin 没有引入可观察的语义回归。
- mIoU delta 量级远小于模型/数据评估常见波动，可视为通过 accuracy gate。
- 该 mIoU 结果不是 latency benchmark；它只用于验证替换 Plugin 后语义输出没有偏移。

---

## 10. P1b / P1mix 分支决策

### P1b

P1b 的目标边界是：

```text
aggregation + cat + relu_linear_att
```

P1b-7 证明扩大边界有价值：

| Metric | Baseline TRT | P1b-7 Plugin | Speedup |
|---|---:|---:|---:|
| End-to-end p50 | `54.3795 ms` | `52.3110 ms` | `1.0395x` |
| `aggregation + attention_core` proxy | `5.443 ms` | `3.043 ms` | `1.789x` |
| stage2 context total | `6.383 ms` | `4.002 ms` | `1.595x` |

但 P1b-7 只覆盖 stage2 两个 context block，端到端 p50 仍未稳定优于 P1a stage2+stage3 的 `50.8380 ms`。因此 P1b 结论是：

- 作为融合路线和后续优化候选保留。
- 不作为当前 Phase 3 主交付线。
- 中间 probe 结果已归档到 [`results/metrics/archive/p1b_probes/`](results/metrics/archive/p1b_probes/)。

### P1mix

P1mix 尝试：

```text
stage2 = P1b-7
stage3 = P1a-3b
```

benchmark 结果：

| Metric | Baseline TRT | P1mix Plugin | Speedup |
|---|---:|---:|---:|
| p50 | `55.2637 ms` | `57.2959 ms` | `0.9645x` |
| mean | `55.4176 ms` | `59.3766 ms` | `0.9333x` |

P1mix 技术链路和 correctness 通过，但未带来正向端到端收益，因此不采纳。

---

## 11. 风险与限制

| 风险 / 限制 | 当前处理 |
|---|---|
| MX250 温度、频率和 Windows 调度会影响 1ms 级差异 | 使用冷机/重复 benchmark 与 Nsight attribution 共同判断，不用单次热机结果定结论 |
| 原始 TensorRT 中没有单一 `relu_linear_att` layer | 使用 `attention_core` proxy，并在报告中显式说明映射 |
| P1a 仍是两阶段 kernel | 已记录单 kernel 可行性反证，见 [`p1a_single_kernel_feasibility.md`](design_notes/p1a_single_kernel_feasibility.md) |
| P1b 有中段收益但端到端未胜出 | 保留为后续候选，不作为当前主线 |
| FP16 / INT8 未纳入当前主交付 | Phase 3 当前只声明 FP32 Plugin 结果 |
| Plugin engine 依赖 TensorRT 8.6.1、MX250 `sm_61` 与本机 DLL 路径 | 元数据与 design notes 记录环境约束 |

---

## 12. 最终结论

Phase 3 的有效结论是：

1. **P1a `relu_linear_att-only` Plugin 已真实集成进 EfficientViT-Seg-B0 TensorRT engine**，覆盖 stage2+stage3 四个 LiteMLA context block。
2. **P1a 通过 correctness、latency、Nsight attribution 和 Cityscapes mIoU gate**。
3. **端到端 execute-only p50 speedup 为 `1.0701x`**，即 `54.3995 ms -> 50.8380 ms`。
4. **`relu_linear_att-only` 子路径相对原始 TensorRT 的 stage2 attention-core proxy speedup 为 `2.819x`**；stage2+stage3 同规则估算约 `2.82x`。
5. **P1b 证明扩大边界有潜力，但当前不优于 P1a stage2+stage3 主线**。
6. **P1mix 不采纳**，因为端到端结果退化。

因此，Phase 3 当前可收敛为：

```text
Accepted MVP:
  P1a relu_linear_att-only Plugin, stage2 + stage3

Archived / future candidates:
  P1b aggregation + cat + relu_linear_att
  P1mix stage2=P1b + stage3=P1a
  P1a single-kernel prototype
```
