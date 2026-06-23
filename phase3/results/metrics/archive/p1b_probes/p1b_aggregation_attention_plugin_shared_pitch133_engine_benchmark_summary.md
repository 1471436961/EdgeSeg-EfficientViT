# EdgesegAggregationReluLinearAttention_TRT Engine Benchmark Summary

- Benchmark target: `both`
- Scope: `p1b_aggregation_attention_plugin_shared_pitch133_probe`

| Item | Value |
|---|---:|
| Baseline TRT p50 | 54.2981 ms |
| Plugin TRT p50 | 52.2680 ms |
| p50 delta (plugin - baseline) | -2.0301 ms |
| p50 speedup (baseline / plugin) | 1.0388x |
| Baseline TRT mean | 54.3185 ms |
| Plugin TRT mean | 52.5222 ms |
| mean delta (plugin - baseline) | -1.7964 ms |
| mean speedup (baseline / plugin) | 1.0342x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 4.43459e-05 | 5.70779e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000281811 | 2.51751e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
