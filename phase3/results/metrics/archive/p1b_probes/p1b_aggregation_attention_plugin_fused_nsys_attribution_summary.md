# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/p1b_aggregation_attention_plugin_fused_fullres.sqlite`
- Metrics: `phase3/results/metrics/archive/p1b_probes/p1b_aggregation_attention_plugin_fused_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 54.366 ms / 54.264 ms
- `trt/execute` kernel avg: 53.588 ms / iter
- `trt/execute` launches: 149.0 / iter
- Layer-attributed kernel avg: 53.588 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.675 | 27.39% | 26.99% | 10.0 | 14 |
| `stage2` | 11.590 | 21.63% | 21.32% | 25.0 | 23 |
| `stage1` | 7.451 | 13.90% | 13.70% | 10.0 | 18 |
| `stage3` | 7.208 | 13.45% | 13.26% | 82.0 | 52 |
| `head` | 6.614 | 12.34% | 12.16% | 15.0 | 22 |
| `stem` | 6.051 | 11.29% | 11.13% | 5.0 | 5 |
| `constant/unnamed` | 0.000 | 0.00% | 0.00% | 0.0 | 2 |

## Stage2 Context Plugin Detail

- Total stage2 context kernel avg: 5.792 ms / iter
- Total stage2 context launches: 10.0 / iter
- Share of execute kernel time: 10.81%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `p1b_fused_aggregation_attention_plugin` | 4.848 | 9.05% | 6.0 | 2 |
| `qkv` | 0.545 | 1.02% | 2.0 | 2 |
| `proj_add` | 0.399 | 0.74% | 2.0 | 2 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `p1b_fused_aggregation_attention_plugin` + `proj_add` | 5.792 | 10.81% | 10.0 |
| `p1b_fused_plugin_only` | `p1b_fused_aggregation_attention_plugin` | 4.848 | 9.05% | 6.0 |
| `p1b_fused_plugin_boundary` | `aggregation` + `p1b_fused_aggregation_attention_plugin` | 4.848 | 9.05% | 6.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 0.944 | 1.76% | 4.0 |
| `aggregation_only` | `aggregation` | 0.000 | 0.00% | 0.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 2.424 | 4.52% | 3.0 |
| `op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 2.423 | 4.52% | 3.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::fusedAggregationCatKernel(const float *, const float *, const float *, float *, int, int)` | 3.536 | 72.94% | 200 |
| 2 | `<unnamed>::computeVkKernelDim16WarpD4(const float *, float *, int)` | 0.960 | 19.79% | 200 |
| 3 | `<unnamed>::computeOutputKernelDim16(const float *, const float *, float *, int, float)` | 0.352 | 7.27% | 200 |

## Baseline TensorRT Comparison

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `attention_proxy: baseline attention_core -> p1b_fused_plugin_only` | 3.689 | 4.848 | 1.159 | 0.761x | 12.0 | 6.0 |
| `middle_boundary: baseline aggregation_plus_attention_core -> p1b_fused_plugin_boundary` | 5.443 | 4.848 | -0.595 | 1.123x | 38.0 | 6.0 |
| `aggregation_preserved` | missing | missing | missing | missing | missing | missing |
| `stage2_context_total` | 6.383 | 5.792 | -0.591 | 1.102x | 42.0 | 10.0 |

## Top 25 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.719 | 5.07% | 1.0 |
| 2 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 2.424 | 4.52% | 3.0 |
| 3 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 2.423 | 4.52% | 3.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.304 | 4.30% | 1.0 |
| 5 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.207 | 4.12% | 1.0 |
| 6 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.840 | 3.43% | 1.0 |
| 7 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.753 | 3.27% | 1.0 |
| 8 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.553 | 2.90% | 1.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.361 | 2.54% | 1.0 |
| 10 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.54% | 1.0 |
| 11 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.54% | 1.0 |
| 12 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.142 | 2.13% | 1.0 |
| 13 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.100 | 2.05% | 1.0 |
| 14 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.975 | 1.82% | 1.0 |
| 15 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.975 | 1.82% | 1.0 |
| 16 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.931 | 1.74% | 1.0 |
| 17 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.912 | 1.70% | 1.0 |
| 18 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.912 | 1.70% | 1.0 |
| 19 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_109)` | 0.681 | 1.27% | 1.0 |
| 20 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.681 | 1.27% | 1.0 |
| 21 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.681 | 1.27% | 1.0 |
| 22 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.680 | 1.27% | 1.0 |
| 23 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_111)` | 0.680 | 1.27% | 1.0 |
| 24 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish), /backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish_1)` | 0.680 | 1.27% | 1.0 |
| 25 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.680 | 1.27% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `p1b_fused_plugin_only` is the runtime cost of the two custom Plugin layers after TensorRT graph replacement.
- `p1b_fused_plugin_boundary` is the Phase 3 proxy for the previous middle-boundary candidate; `cat` may no longer be a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline attribution as the before state and this Plugin engine attribution as the after state.
