# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/p1b_aggregation_attention_plugin_engine_fullres.sqlite`
- Metrics: `phase3/results/metrics/archive/p1b_probes/p1b_aggregation_attention_plugin_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 56.180 ms / 56.124 ms
- `trt/execute` kernel avg: 54.902 ms / iter
- `trt/execute` launches: 151.0 / iter
- Layer-attributed kernel avg: 54.902 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.684 | 26.75% | 26.14% | 10.0 | 12 |
| `stage2` | 12.963 | 23.61% | 23.07% | 29.0 | 23 |
| `stage1` | 7.436 | 13.54% | 13.24% | 10.0 | 16 |
| `stage3` | 7.202 | 13.12% | 12.82% | 81.0 | 45 |
| `head` | 6.573 | 11.97% | 11.70% | 14.0 | 20 |
| `stem` | 6.043 | 11.01% | 10.76% | 5.0 | 5 |
| `constant/unnamed` | 0.000 | 0.00% | 0.00% | 0.0 | 2 |

## Stage2 Context Plugin Detail

- Total stage2 context kernel avg: 7.136 ms / iter
- Total stage2 context launches: 14.0 / iter
- Share of execute kernel time: 13.00%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `p1b_aggregation_attention_plugin` | 6.189 | 11.27% | 10.0 | 2 |
| `qkv` | 0.548 | 1.00% | 2.0 | 2 |
| `proj_add` | 0.399 | 0.73% | 2.0 | 2 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `p1b_aggregation_attention_plugin` + `proj_add` | 7.136 | 13.00% | 14.0 |
| `p1b_aggregation_attention_plugin_only` | `p1b_aggregation_attention_plugin` | 6.189 | 11.27% | 10.0 |
| `p1b_aggregation_attention_plugin_boundary` | `aggregation` + `p1b_aggregation_attention_plugin` | 6.189 | 11.27% | 10.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 0.947 | 1.73% | 4.0 |
| `aggregation_only` | `aggregation` | 0.000 | 0.00% | 0.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 3.095 | 5.64% | 5.0 |
| `op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 3.094 | 5.64% | 5.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::depthwise5x5Kernel(const float *, const float *, float *, int, int, int)` | 2.462 | 39.77% | 200 |
| 2 | `<unnamed>::groupedPointwise1x1Kernel(const float *, const float *, float *, int, int, int, int)` | 2.427 | 39.22% | 200 |
| 3 | `<unnamed>::computeVkKernelDim16WarpD4(const float *, float *, int)` | 0.949 | 15.33% | 200 |
| 4 | `<unnamed>::computeOutputKernelDim16(const float *, const float *, float *, int, float)` | 0.351 | 5.67% | 200 |

## Baseline TensorRT Comparison

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `attention_proxy: baseline attention_core -> p1b_aggregation_attention_plugin_only` | 3.689 | 6.189 | 2.500 | 0.596x | 12.0 | 10.0 |
| `middle_boundary: baseline aggregation_plus_attention_core -> p1b_aggregation_attention_plugin_boundary` | 5.443 | 6.189 | 0.746 | 0.879x | 38.0 | 10.0 |
| `aggregation_preserved` | missing | missing | missing | missing | missing | missing |
| `stage2_context_total` | 6.383 | 7.136 | 0.754 | 0.894x | 42.0 | 14.0 |

## Top 25 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 3.095 | 5.64% | 5.0 |
| 2 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 3.094 | 5.64% | 5.0 |
| 3 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.719 | 4.95% | 1.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.297 | 4.18% | 1.0 |
| 5 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.202 | 4.01% | 1.0 |
| 6 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.838 | 3.35% | 1.0 |
| 7 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.751 | 3.19% | 1.0 |
| 8 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.551 | 2.82% | 1.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.360 | 2.48% | 1.0 |
| 10 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.48% | 1.0 |
| 11 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.48% | 1.0 |
| 12 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.141 | 2.08% | 1.0 |
| 13 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.111 | 2.02% | 1.0 |
| 14 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.965 | 1.76% | 1.0 |
| 15 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.964 | 1.76% | 1.0 |
| 16 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.930 | 1.69% | 1.0 |
| 17 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.912 | 1.66% | 1.0 |
| 18 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.911 | 1.66% | 1.0 |
| 19 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_109)` | 0.681 | 1.24% | 1.0 |
| 20 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.681 | 1.24% | 1.0 |
| 21 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.680 | 1.24% | 1.0 |
| 22 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_111)` | 0.680 | 1.24% | 1.0 |
| 23 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.680 | 1.24% | 1.0 |
| 24 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish), /backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish_1)` | 0.680 | 1.24% | 1.0 |
| 25 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.680 | 1.24% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `p1b_aggregation_attention_plugin_only` is the runtime cost of the two custom Plugin layers after TensorRT graph replacement.
- `p1b_aggregation_attention_plugin_boundary` is the Phase 3 proxy for the previous middle-boundary candidate; `cat` may no longer be a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline attribution as the before state and this Plugin engine attribution as the after state.
