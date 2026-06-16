# EdgesegReluLinearAttention_TRT Engine Benchmark Summary

- Benchmark target: `both`
- Scope: `p1a_relu_linear_att_plugin_stage2_stage3`

| Item | Value |
|---|---:|
| Baseline TRT p50 | 54.3995 ms |
| Plugin TRT p50 | 50.8380 ms |
| p50 delta (plugin - baseline) | -3.5615 ms |
| p50 speedup (baseline / plugin) | 1.0701x |
| Baseline TRT mean | 54.4019 ms |
| Plugin TRT mean | 50.9334 ms |
| mean delta (plugin - baseline) | -3.4684 ms |
| mean speedup (baseline / plugin) | 1.0681x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 4.48227e-05 | 5.64205e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000278473 | 2.51554e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
