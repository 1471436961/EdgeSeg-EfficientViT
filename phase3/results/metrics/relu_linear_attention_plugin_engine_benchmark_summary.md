# ReLU Linear Attention Plugin Engine Benchmark Summary

- Benchmark target: `both`

| Item | Value |
|---|---:|
| Baseline TRT p50 | 54.4988 ms |
| Plugin TRT p50 | 55.8372 ms |
| p50 delta (plugin - baseline) | 1.3384 ms |
| p50 speedup (baseline / plugin) | 0.9760x |
| Baseline TRT mean | 54.6603 ms |
| Plugin TRT mean | 56.2853 ms |
| mean delta (plugin - baseline) | 1.6250 ms |
| mean speedup (baseline / plugin) | 0.9711x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 7.24792e-05 | 7.02371e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000324249 | 2.54047e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
- This `both` run executes baseline first and Plugin second in the same process. For the current ~1 ms delta scale, order / frequency drift is large enough that standalone probes and plugin-only Nsight attribution must be checked before drawing a performance conclusion.
