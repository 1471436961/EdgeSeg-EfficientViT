# Phase 3 Plugin Engine Nsight Attribution Summary

- SQLite: `phase3/results/nsight/p1mix_stage2_p1b_stage3_p1a_fullres.sqlite`
- Metrics: `phase3/results/metrics/p1mix_stage2_p1b_stage3_p1a_engine_benchmark_nsys.json`
- Precision: `fp32`
- Benchmark target: `plugin`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 51.534 ms / 51.458 ms
- `trt/execute` kernel avg: 50.784 ms / iter
- `trt/execute` launches: 146.0 / iter
- Layer-attributed kernel avg: 50.784 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.653 | 28.85% | 28.43% | 10.0 | 12 |
| `stage2` | 10.004 | 19.70% | 19.41% | 26.0 | 24 |
| `stage1` | 7.439 | 14.65% | 14.44% | 10.0 | 10 |
| `head` | 6.627 | 13.05% | 12.86% | 16.0 | 16 |
| `stem` | 6.041 | 11.90% | 11.72% | 5.0 | 5 |
| `stage3` | 6.019 | 11.85% | 11.68% | 77.0 | 35 |

## Plugin Context Detail (stage2 + stage3)

- Total selected context kernel avg: 6.624 ms / iter
- Total selected context launches: 71.0 / iter
- Share of execute kernel time: 13.04%

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter | Layer count |
|---|---:|---:|---:|---:|
| `aggregation_attention_p1b_plugin` | 3.097 | 6.10% | 6.0 | 2 |
| `qkv` | 1.031 | 2.03% | 4.0 | 4 |
| `aggregation` | 0.969 | 1.91% | 50.0 | 4 |
| `proj_add` | 0.886 | 1.74% | 7.0 | 7 |
| `relu_linear_att_plugin` | 0.640 | 1.26% | 4.0 | 2 |

### Candidate Boundaries

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context_plugin_path` | `qkv` + `aggregation` + `relu_linear_att_plugin` + `aggregation_attention_p1b_plugin` + `proj_add` | 6.624 | 13.04% | 71.0 |
| `p1mix_plugin_plus_remaining_aggregation_proxy` | `aggregation` + `relu_linear_att_plugin` + `aggregation_attention_p1b_plugin` | 4.707 | 9.27% | 60.0 |
| `p1mix_plugin_total` | `relu_linear_att_plugin` + `aggregation_attention_p1b_plugin` | 3.737 | 7.36% | 10.0 |
| `qkv_proj_overhead` | `qkv` + `proj_add` | 1.917 | 3.77% | 11.0 |
| `aggregation_only` | `aggregation` | 0.969 | 1.91% | 50.0 |

### Plugin Layer Rows

| Block | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `stage2/op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 1.549 | 3.05% | 3.0 |
| `stage2/op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 1.548 | 3.05% | 3.0 |
| `stage3/op_list.2` | `/backbone/stages.3/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT` | 0.320 | 0.63% | 2.0 |
| `stage3/op_list.1` | `/backbone/stages.3/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT` | 0.320 | 0.63% | 2.0 |

### Plugin Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `<unnamed>::fusedAggregationCatKernel(const float *, const float *, const float *, float *, int, int)` | 1.787 | 47.81% | 200 |
| 2 | `<unnamed>::computeVkKernelDim16WarpD4(const float *, float *, int)` | 1.418 | 37.94% | 400 |
| 3 | `<unnamed>::computeOutputKernelDim16(const float *, const float *, float *, int, float)` | 0.533 | 14.25% | 400 |

## Baseline TensorRT Comparison (Stage2 Only)

| Boundary | Before ms | After ms | Delta ms | Speedup | Before launches | After launches |
|---|---:|---:|---:|---:|---:|---:|
| `attention_proxy: baseline attention_core -> p1mix_plugin_total` | 3.689 | 3.097 | -0.592 | 1.191x | 12.0 | 6.0 |
| `middle_boundary: baseline aggregation_plus_attention_core -> p1mix_plugin_plus_remaining_aggregation_proxy` | 5.443 | 3.097 | -2.346 | 1.757x | 38.0 | 6.0 |
| `aggregation_preserved` | missing | missing | missing | missing | missing | missing |
| `stage2_context_total` | 6.383 | 4.138 | -2.245 | 1.542x | 42.0 | 11.0 |

## Top 30 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.718 | 5.35% | 1.0 |
| 2 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.299 | 4.53% | 1.0 |
| 3 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.199 | 4.33% | 1.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.838 | 3.62% | 1.0 |
| 5 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.751 | 3.45% | 1.0 |
| 6 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.553 | 3.06% | 1.0 |
| 7 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 1.549 | 3.05% | 3.0 |
| 8 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/EdgesegAggregationReluLinearAttention_TRT` | 1.548 | 3.05% | 3.0 |
| 9 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.68% | 1.0 |
| 10 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.360 | 2.68% | 1.0 |
| 11 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.359 | 2.68% | 1.0 |
| 12 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.143 | 2.25% | 1.0 |
| 13 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.097 | 2.16% | 1.0 |
| 14 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.967 | 1.90% | 1.0 |
| 15 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.966 | 1.90% | 1.0 |
| 16 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.929 | 1.83% | 1.0 |
| 17 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.910 | 1.79% | 1.0 |
| 18 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.910 | 1.79% | 1.0 |
| 19 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_22)` | 0.681 | 1.34% | 1.0 |
| 20 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.681 | 1.34% | 1.0 |
| 21 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.680 | 1.34% | 1.0 |
| 22 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_24)` | 0.680 | 1.34% | 1.0 |
| 23 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.680 | 1.34% | 1.0 |
| 24 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish), /backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish_1)` | 0.680 | 1.34% | 1.0 |
| 25 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.680 | 1.34% | 1.0 |
| 26 | `head` | `PWN(PWN(/head/middle/op_list.0/main/depth_conv/act/HardSwish), /head/middle/op_list.0/main/depth_conv/act/HardSwish_23)` | 0.680 | 1.34% | 1.0 |
| 27 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/depth_conv/act/HardSwish_3)` | 0.677 | 1.33% | 1.0 |
| 28 | `stage2` | `/backbone/stages.2/op_list.0/main/depth_conv/conv/Conv` | 0.597 | 1.18% | 1.0 |
| 29 | `stage0` | `/backbone/stages.0/op_list.0/main/point_conv/conv/Conv` | 0.586 | 1.15% | 1.0 |
| 30 | `stage1` | `/backbone/stages.1/op_list.1/main/inverted_conv/conv/Conv` | 0.550 | 1.08% | 1.0 |

## Interpretation Notes

- This Plugin engine trace executes only the Phase 3 Plugin engine, not the Phase 2 baseline engine.
- `p1mix_plugin_total` is the runtime cost of the selected custom Plugin layers after TensorRT graph replacement.
- `p1mix_plugin_plus_remaining_aggregation_proxy` is the Phase 3 proxy for the previous middle-boundary candidate; `cat` may no longer be a separate TensorRT layer at this boundary.
- The comparison table uses Phase 2 TensorRT baseline stage2 attribution as the before state; for `--context-stages 2 3`, stage3 is reported in the Plugin Context Detail but is not mixed into the stage2 baseline comparison.
