# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/relu_linear_attention_plugin_stage2_stage3_fullres.sqlite`
- Metrics: `phase3/results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 51.448 ms / 51.416 ms
- `trt/execute` kernel avg: 50.680 ms / iter
- `trt/execute` launches: 163.0 / iter
- Layer-attributed kernel avg: 50.680 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.651 | 28.91% | 28.48% | 10.0 | 18 |
| `stage2` | 9.888 | 19.51% | 19.22% | 49.0 | 25 |
| `stage1` | 7.430 | 14.66% | 14.44% | 10.0 | 12 |
| `head` | 6.574 | 12.97% | 12.78% | 14.0 | 14 |
| `stage3` | 6.094 | 12.02% | 11.85% | 73.0 | 35 |
| `stem` | 6.043 | 11.92% | 11.75% | 5.0 | 5 |

## Plugin Context Detail (stage2 + stage3)

- Total selected context kernel avg: 6.436 ms / iter
- Total selected context launches: 92.0 / iter
- Share of execute kernel time: 12.70%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `aggregation` | 2.719 | 5.36% | 76.0 | 8 |
| `relu_linear_att_plugin` | 1.950 | 3.85% | 8.0 | 4 |
| `qkv` | 1.055 | 2.08% | 4.0 | 4 |
| `proj_add` | 0.712 | 1.41% | 4.0 | 4 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `relu_linear_att_plugin` + `proj_add` | 6.436 | 12.70% | 92.0 |
| `aggregation_plus_plugin_stage2_stage3_proxy` | `aggregation` + `relu_linear_att_plugin` | 4.669 | 9.21% | 84.0 |
| `aggregation_only` | `aggregation` | 2.719 | 5.36% | 76.0 |
| `relu_linear_att_plugin_stage2_stage3` | `relu_linear_att_plugin` | 1.950 | 3.85% | 8.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 1.767 | 3.49% | 8.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `stage2/op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 0.655 | 1.29% | 2.0 |
| `stage2/op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 0.654 | 1.29% | 2.0 |
| `stage3/op_list.2` | `/backbone/stages.3/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 0.321 | 0.63% | 2.0 |
| `stage3/op_list.1` | `/backbone/stages.3/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 0.320 | 0.63% | 2.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::computeVkKernelDim16WarpD4(const float *, float *, int)` | 1.418 | 72.71% | 400 |
| 2 | `<unnamed>::computeOutputKernelDim16(const float *, const float *, float *, int, float)` | 0.532 | 27.29% | 400 |

## Baseline TensorRT Comparison (Stage2 Only)

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `attention_proxy: baseline attention_core -> relu_linear_att_plugin_stage2_stage3` | 3.689 | 1.309 | -2.380 | 2.819x | 12.0 | 4.0 |
| `middle_boundary: baseline aggregation_plus_attention_core -> aggregation_plus_plugin_stage2_stage3_proxy` | 5.443 | 3.058 | -2.384 | 1.780x | 38.0 | 30.0 |
| `aggregation_preserved` | 1.754 | 1.750 | -0.004 | 1.002x | 26.0 | 26.0 |
| `stage2_context_total` | 6.383 | 4.033 | -2.350 | 1.583x | 42.0 | 34.0 |

## Top 30 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.718 | 5.36% | 1.0 |
| 2 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.303 | 4.54% | 1.0 |
| 3 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.202 | 4.35% | 1.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.840 | 3.63% | 1.0 |
| 5 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.754 | 3.46% | 1.0 |
| 6 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.552 | 3.06% | 1.0 |
| 7 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.360 | 2.68% | 1.0 |
| 8 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.68% | 1.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.68% | 1.0 |
| 10 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.138 | 2.25% | 1.0 |
| 11 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.089 | 2.15% | 1.0 |
| 12 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.964 | 1.90% | 1.0 |
| 13 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.963 | 1.90% | 1.0 |
| 14 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.928 | 1.83% | 1.0 |
| 15 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.911 | 1.80% | 1.0 |
| 16 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.911 | 1.80% | 1.0 |
| 17 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_22)` | 0.681 | 1.34% | 1.0 |
| 18 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.681 | 1.34% | 1.0 |
| 19 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.680 | 1.34% | 1.0 |
| 20 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.680 | 1.34% | 1.0 |
| 21 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_24)` | 0.680 | 1.34% | 1.0 |
| 22 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish), /backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish_1)` | 0.680 | 1.34% | 1.0 |
| 23 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.680 | 1.34% | 1.0 |
| 24 | `head` | `PWN(PWN(/head/middle/op_list.0/main/depth_conv/act/HardSwish), /head/middle/op_list.0/main/depth_conv/act/HardSwish_23)` | 0.680 | 1.34% | 1.0 |
| 25 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/depth_conv/act/HardSwish_3)` | 0.678 | 1.34% | 1.0 |
| 26 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 0.655 | 1.29% | 2.0 |
| 27 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 0.654 | 1.29% | 2.0 |
| 28 | `stage2` | `/backbone/stages.2/op_list.0/main/depth_conv/conv/Conv` | 0.598 | 1.18% | 1.0 |
| 29 | `stage0` | `/backbone/stages.0/op_list.0/main/point_conv/conv/Conv` | 0.587 | 1.16% | 1.0 |
| 30 | `stage1` | `/backbone/stages.1/op_list.1/main/inverted_conv/conv/Conv` | 0.550 | 1.08% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `relu_linear_att_plugin_stage2_stage3` is the runtime cost of the selected custom Plugin layers after TensorRT graph replacement.
- `aggregation_plus_plugin_stage2_stage3_proxy` is the Phase 3 proxy for the previous middle-boundary candidate; `cat` may no longer be a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline stage2 attribution as the before state; for `--context-stages 2 3`, stage3 is reported in the Plugin Context Detail but is not mixed into the stage2 baseline comparison.
