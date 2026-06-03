# Nsight Attribution Summary

- SQLite: `phase1/results/nsight/baseline_planC_fullres.sqlite`
- Metrics: `phase1/results/metrics/baseline_b0_cityscapes_1024x2048_levelC_latency_nsys.json`
- Script version: `baseline_inference.py@39819e3`
- NVTX level: `C`
- Warmup / measure: 20 / 100
- CUDA Event forward mean: 90.222 ms
- Attribution method: CUDA runtime launch `correlationId` -> CUDA kernel duration -> NVTX range.
- Note: NVTX range duration itself is not GPU component time.

## Group Summary

| Group | Avg Kernel ms / iter | Share of attributed kernels | Share of forward mean |
|---|---:|---:|---:|
| `stage0` | 24.510 | 45.31% | 27.17% |
| `stage2` | 19.008 | 35.13% | 21.07% |
| `head` | 10.581 | 19.56% | 11.73% |

## Range Summary

| Range | Count | Launches | Matched | Avg Kernel ms / iter | Share of attributed kernels | Share of forward mean |
|---|---:|---:|---:|---:|---:|---:|
| `stage0/block0/main` | 100 | 2200 | 1000 | 12.294 | 22.73% | 13.63% |
| `stage0/block1/main` | 100 | 2200 | 1000 | 12.216 | 22.58% | 13.54% |
| `head/middle` | 100 | 2300 | 1100 | 6.399 | 11.83% | 7.09% |
| `stage2/block1/context` | 100 | 6100 | 2700 | 4.992 | 9.23% | 5.53% |
| `stage2/block2/context` | 100 | 6100 | 2700 | 4.989 | 9.22% | 5.53% |
| `stage2/downsample` | 100 | 1700 | 900 | 3.016 | 5.57% | 3.34% |
| `stage2/block2/local` | 100 | 1600 | 900 | 3.006 | 5.56% | 3.33% |
| `stage2/block1/local` | 100 | 1600 | 900 | 3.005 | 5.55% | 3.33% |
| `head/output_segout` | 100 | 1500 | 700 | 2.577 | 4.76% | 2.86% |
| `head/input_stage3` | 100 | 1000 | 500 | 0.654 | 1.21% | 0.73% |
| `head/input_stage4` | 100 | 800 | 400 | 0.555 | 1.03% | 0.62% |
| `head/input_stage2` | 100 | 800 | 300 | 0.396 | 0.73% | 0.44% |
