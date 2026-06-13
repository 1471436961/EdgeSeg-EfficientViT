# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/relu_linear_attention_plugin_engine_fullres.sqlite`
- Metrics: `phase3/results/metrics/relu_linear_attention_plugin_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 54.523 ms / 54.497 ms
- `trt/execute` kernel avg: 53.736 ms / iter
- `trt/execute` launches: 178.0 / iter
- Layer-attributed kernel avg: 53.736 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.636 | 27.24% | 26.84% | 10.0 | 18 |
| `stage2` | 11.783 | 21.93% | 21.61% | 50.0 | 29 |
| `stage1` | 7.430 | 13.83% | 13.63% | 10.0 | 14 |
| `stage3` | 7.187 | 13.37% | 13.18% | 85.0 | 55 |
| `head` | 6.666 | 12.41% | 12.23% | 16.0 | 16 |
| `stem` | 6.034 | 11.23% | 11.07% | 5.0 | 9 |
| `constant/unnamed` | 0.000 | 0.00% | 0.00% | 0.0 | 2 |

## Stage2 Context Plugin Detail

- Total stage2 context kernel avg: 5.928 ms / iter
- Total stage2 context launches: 35.0 / iter
- Share of execute kernel time: 11.03%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `relu_linear_att_plugin` | 3.147 | 5.86% | 4.0 | 2 |
| `aggregation` | 1.747 | 3.25% | 26.0 | 4 |
| `qkv` | 0.549 | 1.02% | 2.0 | 2 |
| `proj_add` | 0.485 | 0.90% | 3.0 | 3 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `relu_linear_att_plugin` + `proj_add` | 5.928 | 11.03% | 35.0 |
| `aggregation_plus_plugin_proxy` | `aggregation` + `relu_linear_att_plugin` | 4.894 | 9.11% | 30.0 |
| `relu_linear_att_plugin_only` | `relu_linear_att_plugin` | 3.147 | 5.86% | 4.0 |
| `aggregation_only` | `aggregation` | 1.747 | 3.25% | 26.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 1.034 | 1.92% | 5.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 1.574 | 2.93% | 2.0 |
| `op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 1.573 | 2.93% | 2.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::computeVkKernel(const float *, float *, int, int, int)` | 1.782 | 56.62% | 200 |
| 2 | `<unnamed>::computeOutputKernel(const float *, const float *, float *, int, int, int, float)` | 1.365 | 43.38% | 200 |

## Baseline TensorRT Comparison

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `relu_linear_att_proxy: baseline attention_core -> plugin layer` | 3.689 | 3.147 | -0.542 | 1.172x | 12.0 | 4.0 |
| `p1b_proxy: baseline aggregation_plus_attention_core -> aggregation_plus_plugin` | 5.443 | 4.894 | -0.549 | 1.112x | 38.0 | 30.0 |
| `aggregation_preserved` | 1.754 | 1.747 | -0.007 | 1.004x | 26.0 | 26.0 |
| `stage2_context_total` | 6.383 | 5.928 | -0.455 | 1.077x | 42.0 | 35.0 |

## Top 25 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.719 | 5.06% | 1.0 |
| 2 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.296 | 4.27% | 1.0 |
| 3 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.192 | 4.08% | 1.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.839 | 3.42% | 1.0 |
| 5 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.749 | 3.25% | 1.0 |
| 6 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 1.574 | 2.93% | 2.0 |
| 7 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 1.573 | 2.93% | 2.0 |
| 8 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.543 | 2.87% | 1.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.360 | 2.53% | 1.0 |
| 10 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.53% | 1.0 |
| 11 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.53% | 1.0 |
| 12 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.140 | 2.12% | 1.0 |
| 13 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.088 | 2.03% | 1.0 |
| 14 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.964 | 1.79% | 1.0 |
| 15 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.964 | 1.79% | 1.0 |
| 16 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.939 | 1.75% | 1.0 |
| 17 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.910 | 1.69% | 1.0 |
| 18 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.910 | 1.69% | 1.0 |
| 19 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_109)` | 0.681 | 1.27% | 1.0 |
| 20 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.681 | 1.27% | 1.0 |
| 21 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.680 | 1.27% | 1.0 |
| 22 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_111)` | 0.680 | 1.27% | 1.0 |
| 23 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.680 | 1.27% | 1.0 |
| 24 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.680 | 1.26% | 1.0 |
| 25 | `head` | `PWN(PWN(/head/middle/op_list.0/main/depth_conv/act/HardSwish), /head/middle/op_list.0/main/depth_conv/act/HardSwish_110)` | 0.680 | 1.26% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `relu_linear_att_plugin_only` is the runtime cost of the two custom Plugin layers after TensorRT graph replacement.
- `aggregation_plus_plugin_proxy` is the Phase 3 proxy for the previous `aggregation + cat + relu_linear_att` middle-boundary candidate; `cat` is no longer a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline attribution as the before state and this Plugin engine attribution as the after state.
