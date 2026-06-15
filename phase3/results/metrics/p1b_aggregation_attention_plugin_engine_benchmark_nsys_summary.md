# EdgesegAggregationReluLinearAttention_TRT Engine Benchmark Summary

- Benchmark target: `plugin`
- Scope: `nsys_plugin_only_stage2_p1b_aggregation_attention_plugin`

| Item | Value |
|---|---:|
| Baseline TRT p50 | skipped |
| Plugin TRT p50 | 56.1236 ms |
| p50 delta (plugin - baseline) | skipped |
| p50 speedup (baseline / plugin) | skipped |
| Baseline TRT mean | skipped |
| Plugin TRT mean | 56.1802 ms |
| mean delta (plugin - baseline) | skipped |
| mean speedup (baseline / plugin) | skipped |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | None | nan | nan | nan |
| Plugin TRT vs PyTorch | None | nan | nan | nan |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
