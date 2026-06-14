# ReLU Linear Attention Plugin Engine Benchmark Summary

- Benchmark target: `both`

| Item | Value |
|---|---:|
| Baseline TRT p50 | 54.4061 ms |
| Plugin TRT p50 | 53.7288 ms |
| p50 delta (plugin - baseline) | -0.6774 ms |
| p50 speedup (baseline / plugin) | 1.0126x |
| Baseline TRT mean | 54.4357 ms |
| Plugin TRT mean | 53.5173 ms |
| mean delta (plugin - baseline) | -0.9184 ms |
| mean speedup (baseline / plugin) | 1.0172x |

## Correctness

| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |
|---|---:|---:|---:|---:|
| Plugin TRT vs Baseline TRT | True | 6.19888e-05 | 6.8738e-06 | 1.000000 |
| Plugin TRT vs PyTorch | False | 0.000305653 | 2.55789e-05 | 1.000000 |

## Interpretation

- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.
- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.
- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.
- This run uses the P1a-3a warp-per-output-scalar VK reduction. The same-process `both` run is positive, but the end-to-end delta is still sub-millisecond to ~1 ms, so it should be interpreted together with the plugin-only Nsight attribution summary.
