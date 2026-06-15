# EdgesegAggregationReluLinearAttention_TRT Engine Benchmark Summary

- Benchmark target: `both`
- Scope: `end_to_end_efficientvit_fp32_baseline_vs_stage2_p1b_aggregation_attention_plugin`

| Item | Value |
|---|---:|
| Baseline TRT p50 | 54.4532 ms |
| Plugin TRT p50 | 56.3395 ms |
| p50 delta (plugin - baseline) | 1.8862 ms |
| p50 speedup (baseline / plugin) | 0.9665x |
| Baseline TRT mean | 54.4771 ms |
| Plugin TRT mean | 56.7579 ms |
| mean delta (plugin - baseline) | 2.2807 ms |
| mean speedup (baseline / plugin) | 0.9598x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 4.43459e-05 | 5.70779e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000281811 | 2.51751e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
