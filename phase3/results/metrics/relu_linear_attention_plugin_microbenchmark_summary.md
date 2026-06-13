# Relu Linear Attention Plugin Microbenchmark Summary

> 该文件汇总 Phase 3 Step 5.5 的单层 toy Plugin microbenchmark。它只覆盖 P1a `relu_linear_att-only` contract：`[1,384,64,128] -> [1,128,64,128]`，不代表完整 EfficientViT TensorRT graph 已完成替换。

## 1. 运行配置

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce MX250 / `sm_61` |
| 精度 | FP32 |
| 输入 | `[1,384,64,128]`，固定 seed 42 |
| 输出 | `[1,128,64,128]` |
| warmup / measure | `20 / 100` |
| 计时口径 | CUDA Events |
| Plugin workspace | `8 x 17 x 16` FP32，约 2.1KB |
| Plugin engine | `phase3/results/engines/relu_linear_attention_toy_fp32.engine` |

## 2. 正确性

| 指标 | 结果 |
|---|---:|
| `max_abs_diff` | `1.4156e-07` |
| `mean_abs_diff` | `8.4468e-09` |
| cosine similarity | `0.9999999999995786` |
| relaxed allclose | `true` (`atol=1e-3`, `rtol=1e-3`) |
| argmax pixel agreement | `1.0` |

结论：当前 Plugin 的 FP32 单层数学与 PyTorch reference 对齐。

## 3. Latency

| 运行 | Plugin p50 | PyTorch reference p50 | p50 speedup |
|---|---:|---:|---:|
| 普通运行 | `2.1023 ms` | `1.9876 ms` | `0.945x` |
| Nsight / NVTX 运行 | `1.7346 ms` | `1.9811 ms` | `1.142x` |

结论：当前单层 Plugin 与 PyTorch reference 处于同一量级；由于两次运行的 p50 方向不同，不能把它写成稳定加速结论。它足以支持继续进入 Step 6 做真实 graph 集成，但后续仍需要端到端 benchmark 和 Nsight attribution 判断实际收益。

## 4. Nsight Kernel Summary

Nsight trace 使用 `--trace=cuda,nvtx` 采集，原始 `.nsys-rep` / `.sqlite` 不入库；入库的是导出的 CSV 汇总：

- [`relu_linear_attention_plugin_microbenchmark_kernel_stats_cuda_gpu_kern_sum.csv`](relu_linear_attention_plugin_microbenchmark_kernel_stats_cuda_gpu_kern_sum.csv)

Top kernel 统计如下：

| Time % | Avg | Instances | Kernel |
|---:|---:|---:|---|
| `27.2%` | `0.907 ms` | `121` | `computeVkKernel` |
| `22.2%` | `0.740 ms` | `121` | `maxwell_sgemm_128x32_tn` |
| `19.6%` | `0.656 ms` | `121` | `computeOutputKernel` |
| `10.4%` | `0.174 ms` | `242` | PyTorch clamp elementwise |
| `6.9%` | `0.230 ms` | `121` | `maxwell_sgemm_128x32_nn` |

解释：

- `computeVkKernel` 和 `computeOutputKernel` 是当前 Plugin 的两个自定义 CUDA kernel。
- `maxwell_sgemm_*` 与 PyTorch elementwise kernel 来自 PyTorch reference 分支，不是 Plugin engine 内部 kernel。
- Plugin 分支已经把 `relu_linear_att` 的 PyTorch 多 kernel 序列压成两个自定义 kernel，但当前实现并没有稳定击败 PyTorch/cuBLAS reference。

## 5. 对后续步骤的影响

1. **继续 Step 6 是合理的**：当前 Plugin 已经证明数学正确、TensorRT 可执行、Nsight 可观察。
2. **不能提前宣称加速**：普通运行 p50 低于 PyTorch reference，Nsight 运行 p50 高于 PyTorch reference，说明结果对运行环境和 profiler overhead 敏感。
3. **Step 6/7 的重点**：先验证真实 EfficientViT graph replacement 是否可行，再用端到端 TensorRT FP32 baseline vs Plugin engine 做同口径对比。
4. **后续优化方向**：若端到端无收益，再回到单层 kernel 优化，重点检查访存模式、block/grid 映射、occupancy 与 launch 数量，而不是直接扩大到 P1b。
