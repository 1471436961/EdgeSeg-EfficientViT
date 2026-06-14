# ReLU Linear Attention Plugin Microbenchmark Summary

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

| 版本 / 运行 | Plugin p50 | PyTorch reference p50 | p50 speedup |
|---|---:|---:|---:|
| 初始两阶段 kernel | `2.1023 ms` | `1.9876 ms` | `0.945x` |
| P0：output kernel 缓存 VK 到 shared memory | `1.2877 ms` | `1.9261 ms` | `1.496x` |

P0 修改只改变 `computeOutputKernel`：每个 CTA 负责一个 head 的一个 spatial tile，先把该 head 的 `(dim+1) x dim` VK 小矩阵加载到 shared memory，再计算输出。它没有改变 `computeVkKernel` 的跨 `N` 维归约方式，也没有改变 Plugin API / ONNX graph replacement。

## 4. Kernel Attribution 口径

早期单层 Nsight CSV 仍保留为历史记录，但不再代表当前 P0 kernel 的最新结论。当前 kernel 级归因以 Step 8 的 Plugin engine Nsight 结果为准：

- `computeOutputKernel` 从旧版约 `1.365 ms/iter` 降到 `0.672 ms/iter`。
- `computeVkKernel` 仍约 `1.776 ms/iter`，成为当前 P1a Plugin 的主要内部瓶颈。
- Plugin layer 总耗时从旧版约 `3.147 ms/iter` 降到 `2.447 ms/iter`。

因此，P0 证明“VK 小矩阵不应在 output 阶段反复从全局显存读取”这个判断是对的；下一轮如果继续优化 P1a，应优先处理 `computeVkKernel` 的归约并行度和访存模式，而不是继续只调 output 阶段。

## 5. 对后续步骤的影响

1. **继续 Step 7/8 是必要的**：单层收益需要在真实 EfficientViT TensorRT graph 中复核。
2. **P0 后端到端收益仍有限**：Plugin engine p50 约 `53.223 ms`，Phase 2 baseline p50 约 `54.388 ms`，speedup 约 `1.022x`。
3. **P1a 仍是正确 MVP**：它已经证明 graph replacement、Plugin runtime、数值对齐和目标边界 kernel time 改善均成立。
4. **下一步优化方向**：若继续 P1a，重点是 `computeVkKernel`；若追求更大端到端收益，才进入 P1b `aggregation + cat + relu_linear_att` 边界。
