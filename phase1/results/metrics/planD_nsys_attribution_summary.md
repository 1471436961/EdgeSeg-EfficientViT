# Nsight Attribution Summary

- SQLite: `phase1/results/nsight/baseline_planD_fullres.sqlite`
- Metrics: `phase1/results/metrics/baseline_b0_cityscapes_1024x2048_levelD_latency_nsys.json`
- Script version: `baseline_inference.py@06f7cc9`
- NVTX level: `D`
- Warmup / measure: 20 / 100
- CUDA Event forward mean: 88.043 ms
- Attribution method: CUDA runtime launch `correlationId` -> CUDA kernel duration -> NVTX range.
- Note: NVTX range duration itself is not GPU component time.

## Group Summary

| Group | Avg Kernel ms / iter | Share of attributed kernels | Share of forward mean |
|---|---:|---:|---:|
| `stage2` | 9.471 | 100.00% | 10.76% |

## Range Summary

| Range | Count | Launches | Matched | Avg Kernel ms / iter | Share of attributed kernels | Share of forward mean |
|---|---:|---:|---:|---:|---:|---:|
| `stage2/block1/litemla/aggregation` | 100 | 4000 | 1300 | 1.840 | 19.43% | 2.09% |
| `stage2/block2/litemla/aggregation` | 100 | 4000 | 1300 | 1.840 | 19.42% | 2.09% |
| `stage2/block2/litemla/relu_linear_att` | 100 | 800 | 800 | 1.805 | 19.06% | 2.05% |
| `stage2/block1/litemla/relu_linear_att` | 100 | 800 | 800 | 1.802 | 19.02% | 2.05% |
| `stage2/block1/litemla/cat` | 100 | 100 | 100 | 0.528 | 5.57% | 0.60% |
| `stage2/block2/litemla/cat` | 100 | 100 | 100 | 0.528 | 5.57% | 0.60% |
| `stage2/block1/litemla/qkv` | 100 | 300 | 100 | 0.296 | 3.13% | 0.34% |
| `stage2/block2/litemla/qkv` | 100 | 300 | 100 | 0.289 | 3.05% | 0.33% |
| `stage2/block1/litemla/proj` | 100 | 800 | 300 | 0.272 | 2.87% | 0.31% |
| `stage2/block2/litemla/proj` | 100 | 800 | 300 | 0.272 | 2.87% | 0.31% |
