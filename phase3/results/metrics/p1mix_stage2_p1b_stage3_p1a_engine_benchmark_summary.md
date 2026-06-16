# EdgesegReluLinearAttention_TRT Engine Benchmark Summary

- Benchmark target: `both`
- Scope: `p1mix_stage2_p1b_stage3_p1a`

| Item | Value |
|---|---:|
| Baseline TRT p50 | 55.2637 ms |
| Plugin TRT p50 | 57.2959 ms |
| p50 delta (plugin - baseline) | 2.0321 ms |
| p50 speedup (baseline / plugin) | 0.9645x |
| Baseline TRT mean | 55.4176 ms |
| Plugin TRT mean | 59.3766 ms |
| mean delta (plugin - baseline) | 3.9590 ms |
| mean speedup (baseline / plugin) | 0.9333x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 4.48227e-05 | 5.64205e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000278473 | 2.51554e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
