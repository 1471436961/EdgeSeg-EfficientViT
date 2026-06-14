# Plugin Kernel Optimization History

> **目的**：记录 Phase 3 P1a `relu_linear_att-only` Plugin kernel 的优化演进，避免后续只看到最新 JSON，而看不到“为什么改、改了什么、改善在哪里、下一步瓶颈变成什么”。

---

## 1. 记录口径

本文件只记录关键对比指标和设计判断，不复制每一版完整 raw JSON。

- v0 raw 结果保存在 git 历史 commit `190ca25` 中。
- P0 raw 结果保存在当前工作区的 metrics JSON 中，并会随本轮提交进入 git。
- 若后续继续 P1a kernel 优化，应在本文件继续追加 P1 / P2，而不是只覆盖现有结果。

旧版 raw 数据可用以下方式查看：

```powershell
git show 190ca25:phase3/results/metrics/relu_linear_attention_plugin_microbenchmark.json
git show 190ca25:phase3/results/metrics/relu_linear_attention_plugin_engine_benchmark.json
git show 190ca25:phase3/results/metrics/relu_linear_attention_plugin_nsys_attribution_summary.md
```

---

## 2. 版本对比

| Version | Kernel 设计 | 单层 Plugin p50 | 端到端 / standalone Plugin p50 | Plugin layer kernel time | `computeVkKernel` | `computeOutputKernel` | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| v0 | 两阶段 kernel；`computeOutputKernel` 每个输出线程直接从 global memory 读取 VK workspace | `2.1023 ms` | `53.8639 ms` | `3.147 ms/iter` | `1.782 ms/iter` | `1.365 ms/iter` | 接入链路正确，但单层普通运行不稳定，output 阶段存在重复 global load |
| P0 | 保留两阶段结构；`computeOutputKernel` 改为每个 CTA 将当前 head 的 VK 小矩阵缓存到 shared memory | `1.2877 ms` | `53.2234 ms` | `2.447 ms/iter` | `1.776 ms/iter` | `0.672 ms/iter` | output 阶段优化有效，P1a 内部主瓶颈转移到 `computeVkKernel` |
| P1a-1a | 每个 block 负责 `(head,row)`，同一 block 内同时计算所有 `d` | `1.4787 ms` | 未进入端到端正式记录 | 未采集 | 未采集 | 未采集 | 比 P0 慢；寄存器/共享内存压力与并行度下降抵消了减少 V 重复读取的收益 |
| P1a-1b | 每个 block 负责 `(head,d)`，同一 block 内同时累加所有 row | `1.9722 ms` | 未进入端到端正式记录 | 未采集 | 未采集 | 未采集 | 明显退化；33 个局部累加器和 33KB shared memory 对 MX250 不友好 |
| P1a-1c | 回到 P0 的 `(head,row,d)` 细粒度并行；`computeVkKernel` 改为 warp shuffle reduction，且 computeVk block size 从 256 降到 128 | `1.2175 ms` | standalone plugin-only `53.109 ms`；Nsight plugin-only `53.660 ms` | `2.186 ms/iter` | `1.512 ms/iter` | `0.674 ms/iter` | 有效；减少了 computeVk 归约开销，同时保留了足够细粒度并行 |

端到端对比仍使用 Phase 2 TensorRT FP32 baseline 作为参照：

| Version | Baseline TRT p50 | Plugin TRT p50 | p50 speedup | Plugin vs baseline TRT |
|---|---:|---:|---:|---|
| v0 | `54.4036 ms` | `53.8639 ms` | `1.0100x` | `allclose=True` |
| P0 | `54.3877 ms` | `53.2234 ms` | `1.0219x` | `allclose=True` |
| P1a-1c standalone probe | baseline-only `54.503 ms` | plugin-only `53.109 ms` | `~1.026x` | correctness 已由 `both` run 验证 |
| P1a-1c same-process `baseline -> plugin` | `54.499 ms` | `55.837 ms` | `0.976x` | `allclose=True` |

解释：P1a-1c 后，同进程 `baseline -> plugin` 的 `both` 结果出现顺序/频率偏置，Plugin 在第二段运行时更容易吃到温度或频率下行。由于端到端差异只有 1ms 量级，本项目后续解读 P1a kernel 改进时以 plugin-only Nsight attribution 和 standalone probe 为主，不把单次 `both` run 写成稳定端到端加速结论。

---

## 3. v0 的问题

v0 证明了三个正向事实：

1. Plugin Creator / DLL / TensorRT runtime 链路可用。
2. `relu_linear_att-only` 子图可替换进真实 EfficientViT TensorRT graph。
3. 两个自定义 kernel 可以把 PyTorch / TensorRT 中的多 kernel 注意力子路径压缩成更少的 launch。

但 v0 的 kernel 设计存在明显低效点：

1. `computeVkKernel` 把每个 `(head, row, dim)` 的跨 `N` 维归约放在一个 CTA 内完成，归约并行度有限。
2. VK workspace 很小，但 `computeOutputKernel` 的每个输出线程都会反复从 global memory 读取 VK。
3. 对 MX250 这类小 GPU，减少无意义 global memory traffic 和 launch 间接开销，比盲目扩大 block 数更重要。

---

## 4. P0 改动

P0 只修改 output 阶段：

```text
grid.x = heads
grid.y = ceil(spatialSize / threadsPerBlock)

每个 CTA：
  1. 从 global VK workspace 读取当前 head 的 (dim+1) x dim 小矩阵
  2. 写入 shared memory
  3. 每个线程处理一个 spatial position
  4. 从 shared memory 读取 VK 并计算输出
```

P0 没有修改：

- Plugin API；
- TensorRT Plugin 序列化字段；
- ONNX graph replacement；
- `computeVkKernel` 的归约策略；
- workspace 大小语义。

因此它是低风险优化：只减少 output kernel 的重复 global read，不改变数学边界。

---

## 5. P0 的结论

P0 支持以下判断：

1. **用户提出的“VK 小矩阵不应反复留在 global memory 中读”是正确的。**
2. `computeOutputKernel` 从 `1.365 ms/iter` 降到 `0.672 ms/iter`，说明 shared-memory VK cache 是有效优化。
3. `computeVkKernel` 基本没变，仍约 `1.776 ms/iter`，因此下一轮 P1a 优化不应继续只调 output 阶段。
4. 端到端 p50 speedup 从 `1.0100x` 提升到 `1.0219x`，方向正确但仍是轻微收益；整网主要 runtime 仍被 stage0 / stage2 local / stage1 / stage3 / head 等标准算子稀释。

---

## 6. P1a-1 的补充结论

P1a-1 尝试说明三件事：

1. **不能简单合并更多 `d` 或 row 到同一个 CTA**。P1a-1a / P1a-1b 都变慢，说明在 MX250 上，寄存器、shared memory 和 occupancy 压力很快会抵消数据复用收益。
2. **保留细粒度并行是重要的**。P1a-1c 仍使用 `(head,row,d)` 一个 block，保持了 v0/P0 的高 block 数，但降低了每个 block 的归约成本。
3. **128 threads 比 256 threads 更适合当前 computeVk**。对 `spatialSize=8192`，128 线程每线程约 64 个元素，仍有足够内存并行度，同时减少了 block 内归约参与者和资源压力。

---

## 7. 下一步候选

若继续优化 P1a，优先级建议如下：

| Priority | 候选 | 目标 | 风险 |
|---|---|---|---|
| P1a-2 | 专门化 `dim=16` / `heads=8` 的编译期常量路径 | 让编译器更好展开循环，降低寄存器和索引开销 | 泛化性降低，但当前 contract 本来固定 |
| P1a-3 | 评估两阶段合并为单 kernel 的可行性 | 消除 workspace global write/read 和一次 launch | 需要处理跨 CTA 全局同步问题，不能简单合并 |
| P1b | 扩大到 `aggregation + cat + relu_linear_att` | 更高端到端收益潜力 | graph surgery、数值对齐和共享内存容量风险更高 |

关键提醒：两阶段合并并不是“直接把两个 kernel 写进一个 kernel”就能正确，因为 `computeOutputKernel` 依赖完整 VK 归约结果，而完整 VK 结果通常需要跨 CTA 同步。若要合并，需要重新设计每个 CTA 的职责范围，或接受重复计算 VK 的代价。
