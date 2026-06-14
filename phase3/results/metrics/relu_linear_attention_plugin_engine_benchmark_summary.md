# ReLU Linear Attention Plugin Engine Benchmark Summary

- Benchmark target: `both`

| Item | Value |
|---|---:|
| Baseline TRT p50 | 54.3903 ms |
| Plugin TRT p50 | 53.7754 ms |
| p50 delta (plugin - baseline) | -0.6149 ms |
| p50 speedup (baseline / plugin) | 1.0114x |
| Baseline TRT mean | 54.4060 ms |
| Plugin TRT mean | 54.2403 ms |
| mean delta (plugin - baseline) | -0.1657 ms |
| mean speedup (baseline / plugin) | 1.0031x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 7.24792e-05 | 7.02371e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000324249 | 2.54047e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
- This run uses the P1a-1d `dim=16` specialized fast path. The p50 result is positive, but the end-to-end delta is still sub-millisecond scale, so plugin-only Nsight attribution and repeat runs remain the safer basis for performance conclusions.
