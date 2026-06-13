# ReLU Linear Attention Plugin Engine Benchmark Summary

| Item | Value |
|---|---:|
| Baseline TRT p50 | 54.4036 ms |
| Plugin TRT p50 | 53.8639 ms |
| p50 delta (plugin - baseline) | -0.5396 ms |
| p50 speedup (baseline / plugin) | 1.0100x |
| Baseline TRT mean | 54.4161 ms |
| Plugin TRT mean | 54.2751 ms |
| mean delta (plugin - baseline) | -0.1410 ms |
| mean speedup (baseline / plugin) | 1.0026x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 6.48499e-05 | 7.02655e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000305176 | 2.56277e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
