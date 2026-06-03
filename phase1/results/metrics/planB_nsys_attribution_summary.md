# Nsight Attribution Summary

- SQLite: `phase1/results/nsight/baseline_planB_fullres.sqlite`
- Metrics: `phase1/results/metrics/baseline_b0_cityscapes_1024x2048_levelB_latency_nsys.json`
- Script version: `baseline_inference.py@4c57d90`
- NVTX level: `B`
- Warmup / measure: 20 / 100
- CUDA Event forward mean: 88.196 ms
- Attribution method: CUDA runtime launch `correlationId` -> CUDA kernel duration -> NVTX range.
- Note: NVTX range duration itself is not GPU component time.

## Group Summary

| Group | Avg Kernel ms / iter | Share of attributed kernels | Share of forward mean |
|---|---:|---:|---:|
| `stage0` | 24.528 | 28.25% | 27.81% |
| `stage2` | 18.458 | 21.26% | 20.93% |
| `stage1` | 12.223 | 14.08% | 13.86% |
| `head` | 10.882 | 12.53% | 12.34% |
| `stage3` | 10.403 | 11.98% | 11.80% |
| `stem` | 10.324 | 11.89% | 11.71% |

## Range Summary

| Range | Count | Launches | Matched | Avg Kernel ms / iter | Share of attributed kernels | Share of forward mean |
|---|---:|---:|---:|---:|---:|---:|
| `stage0` | 100 | 4300 | 2000 | 24.528 | 28.25% | 27.81% |
| `stage2` | 100 | 17100 | 8100 | 18.458 | 21.26% | 20.93% |
| `stage1` | 100 | 4300 | 2000 | 12.223 | 14.08% | 13.86% |
| `head` | 100 | 6800 | 3300 | 10.882 | 12.53% | 12.34% |
| `stage3` | 100 | 29500 | 15500 | 10.403 | 11.98% | 11.80% |
| `stem` | 100 | 2100 | 1000 | 10.324 | 11.89% | 11.71% |
