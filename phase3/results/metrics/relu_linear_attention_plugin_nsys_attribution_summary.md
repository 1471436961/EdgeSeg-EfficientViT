# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/relu_linear_attention_plugin_engine_fullres.sqlite`
- Metrics: `phase3/results/metrics/relu_linear_attention_plugin_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 53.705 ms / 53.660 ms
- `trt/execute` kernel avg: 52.905 ms / iter
- `trt/execute` launches: 178.0 / iter
- Layer-attributed kernel avg: 52.905 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.659 | 27.71% | 27.30% | 10.0 | 18 |
| `stage2` | 10.849 | 20.51% | 20.20% | 50.0 | 29 |
| `stage1` | 7.440 | 14.06% | 13.85% | 10.0 | 14 |
| `stage3` | 7.214 | 13.63% | 13.43% | 85.0 | 55 |
| `head` | 6.684 | 12.63% | 12.45% | 16.0 | 16 |
| `stem` | 6.059 | 11.45% | 11.28% | 5.0 | 9 |
| `constant/unnamed` | 0.000 | 0.00% | 0.00% | 0.0 | 2 |

## Stage2 Context Plugin Detail

- Total stage2 context kernel avg: 4.978 ms / iter
- Total stage2 context launches: 35.0 / iter
- Share of execute kernel time: 9.41%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `relu_linear_att_plugin` | 2.186 | 4.13% | 4.0 | 2 |
| `aggregation` | 1.754 | 3.32% | 26.0 | 4 |
| `qkv` | 0.552 | 1.04% | 2.0 | 2 |
| `proj_add` | 0.486 | 0.92% | 3.0 | 3 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `relu_linear_att_plugin` + `proj_add` | 4.978 | 9.41% | 35.0 |
| `aggregation_plus_plugin_proxy` | `aggregation` + `relu_linear_att_plugin` | 3.940 | 7.45% | 30.0 |
| `relu_linear_att_plugin_only` | `relu_linear_att_plugin` | 2.186 | 4.13% | 4.0 |
| `aggregation_only` | `aggregation` | 1.754 | 3.32% | 26.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 1.038 | 1.96% | 5.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 1.093 | 2.07% | 2.0 |
| `op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 1.093 | 2.07% | 2.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::computeVkKernel(const float *, float *, int, int, int)` | 1.512 | 69.18% | 200 |
| 2 | `<unnamed>::computeOutputKernel(const float *, const float *, float *, int, int, int, float)` | 0.674 | 30.82% | 200 |

## Baseline TensorRT Comparison

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `relu_linear_att_proxy: baseline attention_core -> plugin layer` | 3.689 | 2.186 | -1.503 | 1.688x | 12.0 | 4.0 |
| `p1b_proxy: baseline aggregation_plus_attention_core -> aggregation_plus_plugin` | 5.443 | 3.940 | -1.503 | 1.381x | 38.0 | 30.0 |
| `aggregation_preserved` | 1.754 | 1.754 | 0.000 | 1.000x | 26.0 | 26.0 |
| `stage2_context_total` | 6.383 | 4.978 | -1.405 | 1.282x | 42.0 | 35.0 |

## Top 25 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.718 | 5.14% | 1.0 |
| 2 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.302 | 4.35% | 1.0 |
| 3 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.206 | 4.17% | 1.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.842 | 3.48% | 1.0 |
| 5 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.754 | 3.32% | 1.0 |
| 6 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.552 | 2.93% | 1.0 |
| 7 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.360 | 2.57% | 1.0 |
| 8 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.57% | 1.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.57% | 1.0 |
| 10 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.141 | 2.16% | 1.0 |
| 11 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 1.093 | 2.07% | 2.0 |
| 12 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 1.093 | 2.07% | 2.0 |
| 13 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.090 | 2.06% | 1.0 |
| 14 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.968 | 1.83% | 1.0 |
| 15 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.968 | 1.83% | 1.0 |
| 16 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.941 | 1.78% | 1.0 |
| 17 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.912 | 1.72% | 1.0 |
| 18 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.912 | 1.72% | 1.0 |
| 19 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_109)` | 0.681 | 1.29% | 1.0 |
| 20 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.681 | 1.29% | 1.0 |
| 21 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.680 | 1.29% | 1.0 |
| 22 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.680 | 1.29% | 1.0 |
| 23 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_111)` | 0.680 | 1.29% | 1.0 |
| 24 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.680 | 1.28% | 1.0 |
| 25 | `head` | `PWN(PWN(/head/middle/op_list.0/main/depth_conv/act/HardSwish), /head/middle/op_list.0/main/depth_conv/act/HardSwish_110)` | 0.680 | 1.28% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `relu_linear_att_plugin_only` is the runtime cost of the two custom Plugin layers after TensorRT graph replacement.
- `aggregation_plus_plugin_proxy` is the Phase 3 proxy for the previous `aggregation + cat + relu_linear_att` middle-boundary candidate; `cat` is no longer a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline attribution as the before state and this Plugin engine attribution as the after state.
