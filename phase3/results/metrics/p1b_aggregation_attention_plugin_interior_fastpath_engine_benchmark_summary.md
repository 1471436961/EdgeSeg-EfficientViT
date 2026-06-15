# EdgesegAggregationReluLinearAttention_TRT Engine Benchmark Summary

- Benchmark target: `both`
- Scope: `p1b_aggregation_attention_plugin_interior_fastpath_probe`

| Item | Value |
|---|---:|
| Baseline TRT p50 | 54.3119 ms |
| Plugin TRT p50 | 54.7103 ms |
| p50 delta (plugin - baseline) | 0.3983 ms |
| p50 speedup (baseline / plugin) | 0.9927x |
| Baseline TRT mean | 54.3102 ms |
| Plugin TRT mean | 54.7572 ms |
| mean delta (plugin - baseline) | 0.4470 ms |
| mean speedup (baseline / plugin) | 0.9918x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 4.43459e-05 | 5.70779e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000281811 | 2.51751e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
