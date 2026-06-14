# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/relu_linear_attention_plugin_engine_fullres.sqlite`
- Metrics: `phase3/results/metrics/relu_linear_attention_plugin_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 53.866 ms / 53.825 ms
- `trt/execute` kernel avg: 53.023 ms / iter
- `trt/execute` launches: 178.0 / iter
- Layer-attributed kernel avg: 53.023 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.633 | 27.60% | 27.17% | 10.0 | 18 |
| `stage2` | 11.082 | 20.90% | 20.57% | 50.0 | 29 |
| `stage1` | 7.423 | 14.00% | 13.78% | 10.0 | 14 |
| `stage3` | 7.184 | 13.55% | 13.34% | 85.0 | 55 |
| `head` | 6.669 | 12.58% | 12.38% | 16.0 | 16 |
| `stem` | 6.033 | 11.38% | 11.20% | 5.0 | 9 |
| `constant/unnamed` | 0.000 | 0.00% | 0.00% | 0.0 | 2 |

## Stage2 Context Plugin Detail

- Total stage2 context kernel avg: 5.226 ms / iter
- Total stage2 context launches: 35.0 / iter
- Share of execute kernel time: 9.86%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `relu_linear_att_plugin` | 2.447 | 4.62% | 4.0 | 2 |
| `aggregation` | 1.745 | 3.29% | 26.0 | 4 |
| `qkv` | 0.549 | 1.04% | 2.0 | 2 |
| `proj_add` | 0.485 | 0.91% | 3.0 | 3 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `relu_linear_att_plugin` + `proj_add` | 5.226 | 9.86% | 35.0 |
| `aggregation_plus_plugin_proxy` | `aggregation` + `relu_linear_att_plugin` | 4.192 | 7.91% | 30.0 |
| `relu_linear_att_plugin_only` | `relu_linear_att_plugin` | 2.447 | 4.62% | 4.0 |
| `aggregation_only` | `aggregation` | 1.745 | 3.29% | 26.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 1.034 | 1.95% | 5.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 1.224 | 2.31% | 2.0 |
| `op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 1.223 | 2.31% | 2.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::computeVkKernel(const float *, float *, int, int, int)` | 1.776 | 72.56% | 200 |
| 2 | `<unnamed>::computeOutputKernel(const float *, const float *, float *, int, int, int, float)` | 0.672 | 27.44% | 200 |

## Baseline TensorRT Comparison

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `relu_linear_att_proxy: baseline attention_core -> plugin layer` | 3.689 | 2.447 | -1.242 | 1.507x | 12.0 | 4.0 |
| `p1b_proxy: baseline aggregation_plus_attention_core -> aggregation_plus_plugin` | 5.443 | 4.192 | -1.251 | 1.298x | 38.0 | 30.0 |
| `aggregation_preserved` | 1.754 | 1.745 | -0.009 | 1.005x | 26.0 | 26.0 |
| `stage2_context_total` | 6.383 | 5.226 | -1.157 | 1.221x | 42.0 | 35.0 |

## Top 25 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.718 | 5.13% | 1.0 |
| 2 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.292 | 4.32% | 1.0 |
| 3 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.192 | 4.13% | 1.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.838 | 3.47% | 1.0 |
| 5 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.749 | 3.30% | 1.0 |
| 6 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.543 | 2.91% | 1.0 |
| 7 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.360 | 2.57% | 1.0 |
| 8 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.56% | 1.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.56% | 1.0 |
| 10 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 1.224 | 2.31% | 2.0 |
| 11 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 1.223 | 2.31% | 2.0 |
| 12 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.139 | 2.15% | 1.0 |
| 13 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.088 | 2.05% | 1.0 |
| 14 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.965 | 1.82% | 1.0 |
| 15 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.959 | 1.81% | 1.0 |
| 16 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.938 | 1.77% | 1.0 |
| 17 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.909 | 1.72% | 1.0 |
| 18 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.909 | 1.71% | 1.0 |
| 19 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_109)` | 0.681 | 1.28% | 1.0 |
| 20 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.681 | 1.28% | 1.0 |
| 21 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.680 | 1.28% | 1.0 |
| 22 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.680 | 1.28% | 1.0 |
| 23 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_111)` | 0.680 | 1.28% | 1.0 |
| 24 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.680 | 1.28% | 1.0 |
| 25 | `head` | `PWN(PWN(/head/middle/op_list.0/main/depth_conv/act/HardSwish), /head/middle/op_list.0/main/depth_conv/act/HardSwish_110)` | 0.680 | 1.28% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `relu_linear_att_plugin_only` is the runtime cost of the two custom Plugin layers after TensorRT graph replacement.
- `aggregation_plus_plugin_proxy` is the Phase 3 proxy for the previous `aggregation + cat + relu_linear_att` middle-boundary candidate; `cat` is no longer a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline attribution as the before state and this Plugin engine attribution as the after state.
