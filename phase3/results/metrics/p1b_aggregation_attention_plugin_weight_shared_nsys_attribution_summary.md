# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/p1b_aggregation_attention_plugin_weight_shared_fullres.sqlite`
- Metrics: `phase3/results/metrics/p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 54.170 ms / 54.159 ms
- `trt/execute` kernel avg: 53.470 ms / iter
- `trt/execute` launches: 155.0 / iter
- Layer-attributed kernel avg: 53.470 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.648 | 27.39% | 27.04% | 10.0 | 12 |
| `stage2` | 11.143 | 20.84% | 20.57% | 25.0 | 25 |
| `stage3` | 7.583 | 14.18% | 14.00% | 88.0 | 61 |
| `stage1` | 7.442 | 13.92% | 13.74% | 10.0 | 14 |
| `head` | 6.613 | 12.37% | 12.21% | 15.0 | 21 |
| `stem` | 6.041 | 11.30% | 11.15% | 5.0 | 5 |
| `constant/unnamed` | 0.000 | 0.00% | 0.00% | 0.0 | 2 |

## Stage2 Context Plugin Detail

- Total stage2 context kernel avg: 5.302 ms / iter
- Total stage2 context launches: 10.0 / iter
- Share of execute kernel time: 9.92%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `p1b_weight_shared_aggregation_attention_plugin` | 4.347 | 8.13% | 6.0 | 2 |
| `qkv` | 0.556 | 1.04% | 2.0 | 2 |
| `proj_add` | 0.399 | 0.75% | 2.0 | 2 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `p1b_weight_shared_aggregation_attention_plugin` + `proj_add` | 5.302 | 9.92% | 10.0 |
| `p1b_weight_shared_plugin_only` | `p1b_weight_shared_aggregation_attention_plugin` | 4.347 | 8.13% | 6.0 |
| `p1b_weight_shared_plugin_boundary` | `aggregation` + `p1b_weight_shared_aggregation_attention_plugin` | 4.347 | 8.13% | 6.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 0.955 | 1.79% | 4.0 |
| `aggregation_only` | `aggregation` | 0.000 | 0.00% | 0.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 2.175 | 4.07% | 3.0 |
| `op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 2.172 | 4.06% | 3.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::fusedAggregationCatKernel(const float *, const float *, const float *, float *, int, int)` | 3.038 | 69.88% | 200 |
| 2 | `<unnamed>::computeVkKernelDim16WarpD4(const float *, float *, int)` | 0.958 | 22.03% | 200 |
| 3 | `<unnamed>::computeOutputKernelDim16(const float *, const float *, float *, int, float)` | 0.352 | 8.09% | 200 |

## Baseline TensorRT Comparison

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `attention_proxy: baseline attention_core -> p1b_weight_shared_plugin_only` | 3.689 | 4.347 | 0.658 | 0.849x | 12.0 | 6.0 |
| `middle_boundary: baseline aggregation_plus_attention_core -> p1b_weight_shared_plugin_boundary` | 5.443 | 4.347 | -1.096 | 1.252x | 38.0 | 6.0 |
| `aggregation_preserved` | missing | missing | missing | missing | missing | missing |
| `stage2_context_total` | 6.383 | 5.302 | -1.081 | 1.204x | 42.0 | 10.0 |

## Top 25 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.718 | 5.08% | 1.0 |
| 2 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.300 | 4.30% | 1.0 |
| 3 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.200 | 4.11% | 1.0 |
| 4 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 2.175 | 4.07% | 3.0 |
| 5 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 2.172 | 4.06% | 3.0 |
| 6 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.839 | 3.44% | 1.0 |
| 7 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.752 | 3.28% | 1.0 |
| 8 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.551 | 2.90% | 1.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.360 | 2.54% | 1.0 |
| 10 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.54% | 1.0 |
| 11 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.54% | 1.0 |
| 12 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.141 | 2.13% | 1.0 |
| 13 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.089 | 2.04% | 1.0 |
| 14 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.966 | 1.81% | 1.0 |
| 15 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.964 | 1.80% | 1.0 |
| 16 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.929 | 1.74% | 1.0 |
| 17 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.911 | 1.70% | 1.0 |
| 18 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.910 | 1.70% | 1.0 |
| 19 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_109)` | 0.681 | 1.27% | 1.0 |
| 20 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.681 | 1.27% | 1.0 |
| 21 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_111)` | 0.681 | 1.27% | 1.0 |
| 22 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.681 | 1.27% | 1.0 |
| 23 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.680 | 1.27% | 1.0 |
| 24 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish), /backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish_1)` | 0.680 | 1.27% | 1.0 |
| 25 | `head` | `PWN(PWN(/head/middle/op_list.0/main/depth_conv/act/HardSwish), /head/middle/op_list.0/main/depth_conv/act/HardSwish_110)` | 0.680 | 1.27% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `p1b_weight_shared_plugin_only` is the runtime cost of the two custom Plugin layers after TensorRT graph replacement.
- `p1b_weight_shared_plugin_boundary` is the Phase 3 proxy for the previous middle-boundary candidate; `cat` may no longer be a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline attribution as the before state and this Plugin engine attribution as the after state.
