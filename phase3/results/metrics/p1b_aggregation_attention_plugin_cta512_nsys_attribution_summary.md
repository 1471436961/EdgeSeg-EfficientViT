# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/p1b_aggregation_attention_plugin_cta512_fullres.sqlite`
- Metrics: `phase3/results/metrics/p1b_aggregation_attention_plugin_cta512_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 52.991 ms / 52.898 ms
- `trt/execute` kernel avg: 52.287 ms / iter
- `trt/execute` launches: 155.0 / iter
- Layer-attributed kernel avg: 52.287 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.669 | 28.05% | 27.68% | 10.0 | 12 |
| `stage2` | 9.860 | 18.86% | 18.61% | 25.0 | 25 |
| `stage3` | 7.613 | 14.56% | 14.37% | 88.0 | 61 |
| `stage1` | 7.454 | 14.26% | 14.07% | 10.0 | 14 |
| `head` | 6.626 | 12.67% | 12.50% | 15.0 | 21 |
| `stem` | 6.065 | 11.60% | 11.44% | 5.0 | 5 |
| `constant/unnamed` | 0.000 | 0.00% | 0.00% | 0.0 | 2 |

## Stage2 Context Plugin Detail

- Total stage2 context kernel avg: 4.002 ms / iter
- Total stage2 context launches: 10.0 / iter
- Share of execute kernel time: 7.65%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `p1b_cta512_aggregation_attention_plugin` | 3.043 | 5.82% | 6.0 | 2 |
| `qkv` | 0.560 | 1.07% | 2.0 | 2 |
| `proj_add` | 0.400 | 0.76% | 2.0 | 2 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `p1b_cta512_aggregation_attention_plugin` + `proj_add` | 4.002 | 7.65% | 10.0 |
| `p1b_cta512_plugin_only` | `p1b_cta512_aggregation_attention_plugin` | 3.043 | 5.82% | 6.0 |
| `p1b_cta512_plugin_boundary` | `aggregation` + `p1b_cta512_aggregation_attention_plugin` | 3.043 | 5.82% | 6.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 0.960 | 1.84% | 4.0 |
| `aggregation_only` | `aggregation` | 0.000 | 0.00% | 0.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 1.522 | 2.91% | 3.0 |
| `op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 1.521 | 2.91% | 3.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::fusedAggregationCatKernel(const float *, const float *, const float *, float *, int, int)` | 1.730 | 56.86% | 200 |
| 2 | `<unnamed>::computeVkKernelDim16WarpD4(const float *, float *, int)` | 0.961 | 31.60% | 200 |
| 3 | `<unnamed>::computeOutputKernelDim16(const float *, const float *, float *, int, float)` | 0.351 | 11.55% | 200 |

## Baseline TensorRT Comparison

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `attention_proxy: baseline attention_core -> p1b_cta512_plugin_only` | 3.689 | 3.043 | -0.646 | 1.212x | 12.0 | 6.0 |
| `middle_boundary: baseline aggregation_plus_attention_core -> p1b_cta512_plugin_boundary` | 5.443 | 3.043 | -2.400 | 1.789x | 38.0 | 6.0 |
| `aggregation_preserved` | missing | missing | missing | missing | missing | missing |
| `stage2_context_total` | 6.383 | 4.002 | -2.380 | 1.595x | 42.0 | 10.0 |

## Top 25 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.719 | 5.20% | 1.0 |
| 2 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.305 | 4.41% | 1.0 |
| 3 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.214 | 4.23% | 1.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.844 | 3.53% | 1.0 |
| 5 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.757 | 3.36% | 1.0 |
| 6 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.560 | 2.98% | 1.0 |
| 7 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 1.522 | 2.91% | 3.0 |
| 8 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 1.521 | 2.91% | 3.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.360 | 2.60% | 1.0 |
| 10 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.60% | 1.0 |
| 11 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.60% | 1.0 |
| 12 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.142 | 2.18% | 1.0 |
| 13 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.091 | 2.09% | 1.0 |
| 14 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.969 | 1.85% | 1.0 |
| 15 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.968 | 1.85% | 1.0 |
| 16 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.930 | 1.78% | 1.0 |
| 17 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.913 | 1.75% | 1.0 |
| 18 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.912 | 1.74% | 1.0 |
| 19 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.681 | 1.30% | 1.0 |
| 20 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_109)` | 0.681 | 1.30% | 1.0 |
| 21 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_111)` | 0.681 | 1.30% | 1.0 |
| 22 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.681 | 1.30% | 1.0 |
| 23 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.681 | 1.30% | 1.0 |
| 24 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish), /backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish_1)` | 0.680 | 1.30% | 1.0 |
| 25 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.680 | 1.30% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `p1b_cta512_plugin_only` is the runtime cost of the two custom Plugin layers after TensorRT graph replacement.
- `p1b_cta512_plugin_boundary` is the Phase 3 proxy for the previous middle-boundary candidate; `cat` may no longer be a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline attribution as the before state and this Plugin engine attribution as the after state.
