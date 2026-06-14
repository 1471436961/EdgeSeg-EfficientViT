# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/relu_linear_attention_plugin_engine_fullres.sqlite`
- Metrics: `phase3/results/metrics/relu_linear_attention_plugin_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 53.771 ms / 53.215 ms
- `trt/execute` kernel avg: 52.965 ms / iter
- `trt/execute` launches: 178.0 / iter
- Layer-attributed kernel avg: 52.965 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.777 | 27.90% | 27.48% | 10.0 | 18 |
| `stage2` | 10.358 | 19.56% | 19.26% | 50.0 | 29 |
| `stage1` | 7.500 | 14.16% | 13.95% | 10.0 | 14 |
| `stage3` | 7.371 | 13.92% | 13.71% | 85.0 | 55 |
| `head` | 6.767 | 12.78% | 12.58% | 16.0 | 16 |
| `stem` | 6.193 | 11.69% | 11.52% | 5.0 | 9 |
| `constant/unnamed` | 0.000 | 0.00% | 0.00% | 0.0 | 2 |

## Stage2 Context Plugin Detail

- Total stage2 context kernel avg: 4.404 ms / iter
- Total stage2 context launches: 35.0 / iter
- Share of execute kernel time: 8.31%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `aggregation` | 1.793 | 3.39% | 26.0 | 4 |
| `relu_linear_att_plugin` | 1.550 | 2.93% | 4.0 | 2 |
| `qkv` | 0.571 | 1.08% | 2.0 | 2 |
| `proj_add` | 0.490 | 0.93% | 3.0 | 3 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `relu_linear_att_plugin` + `proj_add` | 4.404 | 8.31% | 35.0 |
| `aggregation_plus_plugin_proxy` | `aggregation` + `relu_linear_att_plugin` | 3.343 | 6.31% | 30.0 |
| `aggregation_only` | `aggregation` | 1.793 | 3.39% | 26.0 |
| `relu_linear_att_plugin_only` | `relu_linear_att_plugin` | 1.550 | 2.93% | 4.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 1.061 | 2.00% | 5.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 0.775 | 1.46% | 2.0 |
| `op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 0.775 | 1.46% | 2.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::computeVkKernelDim16Warp4(const float *, float *, int)` | 1.198 | 77.28% | 200 |
| 2 | `<unnamed>::computeOutputKernelDim16(const float *, const float *, float *, int, float)` | 0.352 | 22.72% | 200 |

## Baseline TensorRT Comparison

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `relu_linear_att_proxy: baseline attention_core -> plugin layer` | 3.689 | 1.550 | -2.139 | 2.380x | 12.0 | 4.0 |
| `p1b_proxy: baseline aggregation_plus_attention_core -> aggregation_plus_plugin` | 5.443 | 3.343 | -2.100 | 1.628x | 38.0 | 30.0 |
| `aggregation_preserved` | 1.754 | 1.793 | 0.039 | 0.978x | 26.0 | 26.0 |
| `stage2_context_total` | 6.383 | 4.404 | -1.979 | 1.449x | 42.0 | 35.0 |

## Top 25 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.719 | 5.13% | 1.0 |
| 2 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.342 | 4.42% | 1.0 |
| 3 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.283 | 4.31% | 1.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.856 | 3.50% | 1.0 |
| 5 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.780 | 3.36% | 1.0 |
| 6 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.596 | 3.01% | 1.0 |
| 7 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.361 | 2.57% | 1.0 |
| 8 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.57% | 1.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.57% | 1.0 |
| 10 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.160 | 2.19% | 1.0 |
| 11 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.098 | 2.07% | 1.0 |
| 12 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.986 | 1.86% | 1.0 |
| 13 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.983 | 1.86% | 1.0 |
| 14 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.954 | 1.80% | 1.0 |
| 15 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.921 | 1.74% | 1.0 |
| 16 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.921 | 1.74% | 1.0 |
| 17 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 0.775 | 1.46% | 2.0 |
| 18 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 0.775 | 1.46% | 2.0 |
| 19 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_109)` | 0.681 | 1.29% | 1.0 |
| 20 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.681 | 1.29% | 1.0 |
| 21 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.681 | 1.29% | 1.0 |
| 22 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_111)` | 0.681 | 1.28% | 1.0 |
| 23 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.680 | 1.28% | 1.0 |
| 24 | `head` | `PWN(PWN(/head/middle/op_list.0/main/depth_conv/act/HardSwish), /head/middle/op_list.0/main/depth_conv/act/HardSwish_110)` | 0.680 | 1.28% | 1.0 |
| 25 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.680 | 1.28% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `relu_linear_att_plugin_only` is the runtime cost of the two custom Plugin layers after TensorRT graph replacement.
- `aggregation_plus_plugin_proxy` is the Phase 3 proxy for the previous `aggregation + cat + relu_linear_att` middle-boundary candidate; `cat` is no longer a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline attribution as the before state and this Plugin engine attribution as the after state.
