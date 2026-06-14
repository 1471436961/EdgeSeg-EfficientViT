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
| `max_abs_diff` | `1.3784e-07` |
| `mean_abs_diff` | `8.4648e-09` |
| cosine similarity | `0.9999999999995786` |
| relaxed allclose | `true` (`atol=1e-3`, `rtol=1e-3`) |
| argmax pixel agreement | `1.0` |

结论：当前 Plugin 的 FP32 单层数学与 PyTorch reference 对齐。

## 3. Latency

| 版本 / 运行 | Plugin p50 | PyTorch reference p50 | p50 speedup |
|---|---:|---:|---:|
| v0：初始两阶段 kernel | `2.1023 ms` | `1.9876 ms` | `0.945x` |
| P0：output kernel 缓存 VK 到 shared memory | `1.2877 ms` | `1.9261 ms` | `1.496x` |
| P1a-1c：computeVk warp reduction + 128 threads | `1.2175 ms` | `2.0096 ms` | `1.651x` |
| P1a-1d：`dim=16` specialized fast path | `0.9938 ms` | `1.8985 ms` | `1.910x` |
| P1a-3a：warp-per-output-scalar VK reduction | `0.8335 ms` | `1.9497 ms` | `2.339x` |

## 4. Kernel Attribution 口径

当前 kernel 级归因以 Step 8 的 Plugin engine Nsight 结果为准：

- v0 Plugin layer：`3.147 ms/iter`
- P0 Plugin layer：`2.447 ms/iter`
- P1a-1c Plugin layer：`2.186 ms/iter`
- P1a-1c 内部：`computeVkKernel = 1.512 ms/iter`，`computeOutputKernel = 0.674 ms/iter`
- P1a-1d Plugin layer：`1.865 ms/iter`
- P1a-1d 内部：`computeVkKernelDim16 = 1.513 ms/iter`，`computeOutputKernelDim16 = 0.352 ms/iter`
- P1a-3a Plugin layer：`1.550 ms/iter`
- P1a-3a 内部：`computeVkKernelDim16Warp4 = 1.198 ms/iter`，`computeOutputKernelDim16 = 0.352 ms/iter`

P1a-3a 保留 P0 的 output shared-memory VK cache 与 P1a-1d 的 `dim=16` output fast path，同时把 VK 归约改成 warp-per-output-scalar：一个 CTA 内 4 个 warp 分别计算同一 row 下 4 个 `d` 标量。它没有扩大 Plugin API，也没有修改 ONNX graph replacement；其他 `dim` 仍走通用 fallback。

## 5. 对后续步骤的影响

1. **P1a 子路径仍在改善**：单层 p50 从 v0 `2.1023 ms` 降到 P1a-3a `0.8335 ms`。
2. **端到端结论仍需谨慎**：同进程 `baseline -> plugin` 对小幅差异有顺序/频率偏置；当前整网差异只有 1ms 量级，不能只看一次 `both` 结果。
3. **下一步若继续 P1a**：`computeVkKernelDim16Warp4` 仍是内部主瓶颈；下一轮应继续做小步 A/B，而不是盲目合并 kernel。
4. **若追求更大端到端收益**：P1b `aggregation + cat + relu_linear_att` 仍是更高收益边界，但 graph surgery 和数值风险更高。
