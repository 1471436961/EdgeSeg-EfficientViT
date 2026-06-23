# P1a-4 Single-Kernel Feasibility

> **状态**：Phase 3 P1a-4 可行性评估。
>
> **结论先行**：当前不建议把 `relu_linear_att-only` 正式重构为单 kernel。P1a-4 值得作为设计反证记录，但不应抢占 P1b `aggregation + cat + relu_linear_att` 的主线优先级。
>
> **后续结果说明**：本文写于 P1b 系统消融之前，因此“下一主线转向 P1b”是当时的探索顺序。后续 P1b-7 与 P1mix 均已完成，最终 Phase 3 MVP 收敛为 P1a `relu_linear_att-only` 覆盖 stage2+stage3；P1b 保留为重要消融和后续候选。

---

## 1. 背景

当前 P1a Plugin 的 `relu_linear_att-only` 实现是两阶段 kernel：

```text
computeVkKernelDim16WarpD4
  input q/k/v
  -> reduce over spatial N
  -> write VK workspace

computeOutputKernelDim16
  read VK workspace
  -> compute normalized attention output
```

P1a-3b 后，冷机重测结果为：

| 口径 | 结果 |
|---|---:|
| 单层 Plugin p50 | `0.7485 ms` |
| Plugin layer Nsight attribution | `1.310 ms / iter` |
| `computeVkKernelDim16WarpD4` | `0.959 ms / iter` |
| `computeOutputKernelDim16` | `0.351 ms / iter` |
| 整网 `both` p50 speedup | `1.043x` |

因此继续优化 P1a 的直觉目标是：消除一次 kernel launch，以及消除 VK workspace 的 global write/read。

---

## 2. 核心依赖关系

`computeOutputKernelDim16` 不是只依赖局部输入 tile，而是依赖完整的 VK 矩阵：

```text
VK[head, row, d] = sum_over_N( V[head, row, n] * relu(K[head, d, n]) )
```

这里的 `N = H * W = 64 * 128 = 8192`。也就是说，输出阶段必须看到跨完整 spatial 维度归约后的 VK。

当前两阶段 kernel 的 kernel boundary 实际承担了一个重要职责：

```text
computeVk 完成所有 CTA 的跨 N 归约
CUDA kernel boundary 提供全局同步
computeOutput 开始读取完整 VK
```

这就是 P1a-4 的关键困难：**CUDA 普通 kernel 内没有跨 CTA 的轻量全局同步点**。

---

## 3. 候选方案评估

| 方案 | 思路 | 主要问题 | 判断 |
|---|---|---|---|
| A. 单 CTA / 少数 CTA 先算完整 VK，再算 output | 一个 kernel 内先由少量 CTA 完成 VK，再由同一 CTA 处理 output | 并行度太低，output 覆盖 `heads * spatial`，少量 CTA 无法高效覆盖整层 | 不采用 |
| B. 每个 output tile 重算 VK | 每个输出 tile 自己完成所需 VK 归约，再立刻计算 output | VK 归约被重复大量计算，计算量暴涨，抵消 launch/workspace 节省 | 不采用 |
| C. kernel 内 atomic counter / spin wait 做全局 barrier | 所有 CTA 先写 VK，再用 device-side barrier 等待，最后算 output | 容易死锁；与 TensorRT Plugin enqueue 路径和 Windows/MX250 环境不匹配 | 不采用 |
| D. cooperative groups grid sync | 使用 cooperative launch 做 grid-level sync | TensorRT Plugin 中 launch 约束更复杂，Windows + MX250 风险高，不适合作为 MVP | 暂不采用 |
| E. 保留两阶段，继续小步优化 computeVk | 不改变同步语义，只降低主瓶颈 kernel | 风险低，但收益递减 | 可作为低优先级小步实验 |
| F. 转向 P1b 中段融合 | 扩大边界到 `aggregation + cat + relu_linear_att` | graph surgery 与数值风险更高 | 推荐作为下一主线 |

---

## 4. 为什么不直接写 single kernel

P1a-4 的收益上限主要来自两点：

1. 少一次 kernel launch。
2. 少一次小型 VK workspace 的 global write/read。

但 VK workspace 本身很小：

```text
heads * (dim + 1) * dim * sizeof(float)
= 8 * 17 * 16 * 4 bytes
= 8704 bytes
```

当前主要耗时仍在跨 `N=8192` 的 K/V 读取与归约上，而不是这 8.5KB workspace 的读写本身。若为了消除 workspace 而引入重复 VK 归约、低并行度或全局同步风险，收益很容易变成负数。

因此，“两阶段合并为单 kernel”不是一个天然正确的优化方向。它只有在能证明以下条件时才值得进入正式实现：

- 不重复大量 VK 归约。
- 不显著降低 output 并行度。
- 不依赖高风险 device-side global barrier。
- 在 MX250 `sm_61` 上实际测得稳定收益。

目前这些条件尚不成立。

---

## 5. 可做但不优先的反证实验

如果后续仍想给 P1a-4 做实验证据，可以只做 micro prototype，不接入正式 engine：

1. **重复 VK 版本**：让每个 output tile 自己重算 VK，估算重复归约代价下界。
2. **低并行度版本**：每个 head 一个 CTA / 少量 CTA 完成 VK + output，估算并行度损失。
3. **只测 launch 开销版本**：保留两阶段数学，但用空 kernel 或极简 kernel 估算一次 launch 对当前 1ms 级 Plugin layer 的贡献。

这些实验的目标是证明或反证 P1a-4，不应替代 P1b 主线。

---

## 6. 对 Phase 3 路线的影响

P1a-4 的评估结论是：

1. 当前 P1a 两阶段结构虽然不完美，但同步语义清楚、数值风险低、已有正收益。
2. 继续 P1a 小步优化可以做，但收益递减明显。
3. 若追求更可见的端到端收益，下一主线应转向 P1b：`aggregation + cat + relu_linear_att`。
4. P1b 的核心价值不是“把 P1a 两个 kernel 合成一个”，而是扩大 fusion 边界，减少更多中间 tensor、更多 launch 和更多 TensorRT residual layer。

因此，P1a-4 当前记录为：

```text
evaluated, not adopted as mainline
```
