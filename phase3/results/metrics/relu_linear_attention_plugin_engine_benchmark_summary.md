# ReLU Linear Attention Plugin Engine Benchmark Summary

- Benchmark target: `both`

| Item | Value |
|---|---:|
| Baseline TRT p50 | 54.3944 ms |
| Plugin TRT p50 | 52.1682 ms |
| p50 delta (plugin - baseline) | -2.2262 ms |
| p50 speedup (baseline / plugin) | 1.0427x |
| Baseline TRT mean | 54.4078 ms |
| Plugin TRT mean | 52.3320 ms |
| mean delta (plugin - baseline) | -2.0758 ms |
| mean speedup (baseline / plugin) | 1.0397x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 6.19888e-05 | 6.8738e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000305653 | 2.55789e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
- This file records the cold sequential rerun after the GPU cooled down. An earlier hot/parallel-contaminated P1a-3b run showed negative end-to-end speedup and is not used as the performance conclusion.
