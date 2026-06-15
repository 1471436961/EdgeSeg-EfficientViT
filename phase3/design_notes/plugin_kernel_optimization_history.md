# Plugin Kernel Optimization History

> **目的**：记录 Phase 3 P1a `relu_linear_att-only` 与 P1b `aggregation + cat + relu_linear_att` Plugin kernel 的优化演进，避免后续只看到最新 JSON，而看不到“为什么改、改了什么、改善在哪里、下一步瓶颈变成什么”。

---

## 1. 记录口径

本文件只记录关键对比指标和设计判断，不复制每一版完整 raw JSON。

- v0 raw 结果保存在 git 历史 commit `190ca25` 中。
- P0 raw 结果保存在 git 历史 commit `7d904cd` 中。
- P1a-1c raw 结果保存在 git 历史 commit `97e45cb` 中。
- P1a-1d / P1a-3a / P1a-3b raw 结果记录在当前 `phase3/results/metrics/` 的 microbenchmark、engine benchmark 与 Nsight attribution 文件中。
- P1b naive / P1b-1 / P1b-2 raw 结果记录在当前 `phase3/results/metrics/` 的 P1b benchmark 与 Nsight attribution 文件中。
- 若后续继续 P1a 或 P1b kernel 优化，应在本文件继续追加记录，而不是只覆盖现有结果。

旧版 raw 数据可用以下方式查看：

```powershell
git show 190ca25:phase3/results/metrics/relu_linear_attention_plugin_microbenchmark.json
git show 190ca25:phase3/results/metrics/relu_linear_attention_plugin_engine_benchmark.json
git show 190ca25:phase3/results/metrics/relu_linear_attention_plugin_nsys_attribution_summary.md
git show 7d904cd:phase3/results/metrics/relu_linear_attention_plugin_microbenchmark_summary.md
git show 97e45cb:phase3/results/metrics/relu_linear_attention_plugin_nsys_attribution_summary.md
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
| P1a-1d | 保留 P1a-1c；为真实 contract 的 `dim=16` 增加编译期专用 fast path | `0.9938 ms` | `both` Plugin `53.7754 ms`；Nsight plugin-only `53.269 ms` | `1.865 ms/iter` | `1.513 ms/iter` | `0.352 ms/iter` | 有效；output 阶段继续下降，内部主瓶颈回到 computeVk 跨 N 归约 |
| P1a-3a | 保留 P1a-1d；把 `dim=16` 的 VK 归约改为一个 CTA 内 4 个 warp 分别计算同一 row 下 4 个 `d` 标量 | `0.8335 ms` | `both` Plugin `53.7288 ms`；Nsight plugin-only `53.215 ms` | `1.550 ms/iter` | `1.198 ms/iter` | `0.352 ms/iter` | 有效；computeVk 继续下降，output 不变，内部主瓶颈仍是 VK 归约 |
| P1a-3b | 保留 P1a-3a 的两阶段边界；改为单个 warp 同时累加 4 个连续 `d`，复用 `V[row,n]` load | `0.7485 ms` | `both` Plugin `52.1682 ms`；Nsight plugin-only `52.783 ms` | `1.310 ms/iter` | `0.959 ms/iter` | `0.351 ms/iter` | 冷机重测后单层、Plugin layer 与端到端均为正收益；此前热机/并行污染 run 不作为结论 |

端到端对比仍使用 Phase 2 TensorRT FP32 baseline 作为参照：

| Version | Baseline TRT p50 | Plugin TRT p50 | p50 speedup | Plugin vs baseline TRT |
|---|---:|---:|---:|---|
| v0 | `54.4036 ms` | `53.8639 ms` | `1.0100x` | `allclose=True` |
| P0 | `54.3877 ms` | `53.2234 ms` | `1.0219x` | `allclose=True` |
| P1a-1c standalone probe | baseline-only `54.503 ms` | plugin-only `53.109 ms` | `~1.026x` | correctness 已由 `both` run 验证 |
| P1a-1c same-process `baseline -> plugin` | `54.499 ms` | `55.837 ms` | `0.976x` | `allclose=True` |
| P1a-1d same-process `baseline -> plugin` | `54.390 ms` | `53.775 ms` | `1.011x` | `allclose=True` |
| P1a-3a same-process `baseline -> plugin` | `54.406 ms` | `53.729 ms` | `1.013x` | `allclose=True` |
| P1a-3b same-process `baseline -> plugin` cold rerun | `54.394 ms` | `52.168 ms` | `1.043x` | `allclose=True` |

解释：P1a-1c 后，同进程 `baseline -> plugin` 的 `both` 结果出现过顺序/频率偏置，Plugin 在第二段运行时更容易吃到温度或频率下行。P1a-3b 首次热机/并行污染 run 曾显示负收益，但在 GPU 冷却并严格顺序运行后，`both` 口径转为 baseline p50 `54.394 ms`、Plugin p50 `52.168 ms`。因此本项目后续解读 P1a kernel 改进时仍以 plugin-only Nsight attribution、单层 microbenchmark 与冷机重复 benchmark 共同判断，不把单次 `both` run 写成稳定端到端加速结论。

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
4. **`dim=16` 专用 fast path 是有效的**。P1a-1d 让 output 阶段固定小循环可被编译器展开，`computeOutputKernel` 从 P1a-1c 的约 `0.674 ms/iter` 降到 `0.352 ms/iter`；但 `computeVkKernelDim16` 仍约 `1.513 ms/iter`，成为新的内部主瓶颈。

---

## 7. P1a-3a 的补充结论

P1a-3a 是在 Nsight Compute 无法支持 MX250 后，按“小步 kernel 变体 A/B 实测”继续推进的 VK 归约优化：

1. **一个 CTA 内 4 个 warp 分别计算 4 个 `d` 标量是有效的**。它把 `dim=16` VK 归约的 CTA 数从 `heads * 17 * 16` 降为 `heads * 17 * 4`，但仍保留 warp 内跨 `N` 归约的并行度。
2. **computeVk 阶段继续下降**。Plugin-only Nsight 中 `computeVkKernelDim16Warp4` 为 `1.198 ms/iter`，低于 P1a-1d 的 `1.513 ms/iter`；`computeOutputKernelDim16` 仍为 `0.352 ms/iter`，说明 P1a-3a 的收益主要来自 VK 归约。
3. **端到端收益仍是小幅正向**。`both` 口径 p50 从 baseline `54.406 ms` 到 Plugin `53.729 ms`，约 `1.013x`；Plugin-only Nsight p50 为 `53.215 ms`。
4. **内部主瓶颈仍未消失**。当前 Plugin layer 为 `1.550 ms/iter`，其中 computeVk 仍占约 `77.28%`。如果继续 P1a，只应做更小步的 VK 归约变体，而不是直接扩大到复杂融合。

---

## 8. P1a-3b 的补充结论

P1a-3b 直接吸收 P1a-2 的 nvprof 结论：当前 VK 归约不是 occupancy-bound，也不是同步等待主导，而是 memory-dependency / load-latency 主导。因此它让单个 warp 同时累加 4 个连续 `d`，用寄存器保存 4 个 partial sum，从而把同一 `V[row,n]` load 复用于 4 个 `K[d,n]`。

实测结果：

1. **单层继续改善**：Plugin p50 从 P1a-3a 的 `0.8335 ms` 降到 P1a-3b 冷机重测的 `0.7485 ms`。
2. **Plugin layer 继续改善**：Nsight plugin-only attribution 中 Plugin layer 从 `1.550 ms/iter` 降到 `1.310 ms/iter`。
3. **computeVk 继续改善**：`computeVkKernelDim16Warp4 = 1.198 ms/iter` 下降到 `computeVkKernelDim16WarpD4 = 0.959 ms/iter`。
4. **output 阶段基本不变**：`computeOutputKernelDim16` 约 `0.351 ms/iter`，符合预期。
5. **冷机端到端 both run 为正**：当前有效 `both` 口径 baseline p50 `54.394 ms`，Plugin p50 `52.168 ms`，p50 speedup `1.043x`。此前热机/并行污染 run 显示负收益，说明整网端到端口径受温度、频率、执行顺序和系统状态影响很大，不能用单次 `both` run 作为唯一结论。

---

## 8.5 P1b 优化历程

P1b 的目标边界是 `aggregation + cat + relu_linear_att`，只替换两个 `stage2/context` block。它不是整体 LiteMLA Plugin：`qkv Conv`、`proj Conv` 和 residual add 仍保留在 TensorRT 路径中。

| Version | Kernel 设计 | 端到端 p50 | Plugin layer kernel time | 关键 kernel | 判断 |
|---|---|---:|---:|---|---|
| P1b naive | `depthwise5x5Kernel -> groupedPointwise1x1Kernel -> cat workspace -> P1a attention launcher` | `56.3395 ms` vs baseline `54.4532 ms`，`0.9665x` | `6.189 ms/iter`，`10 launches/iter` | `depthwise5x5Kernel 2.462 ms` + `groupedPointwise1x1Kernel 2.427 ms` | 数学正确但性能退化；自写 aggregation 两个 kernel 合计占 Plugin 时间约 79%，替换掉 TensorRT/cuDNN 已优化 Conv 路径后得不偿失 |
| P1b-1 | 融合 depthwise、grouped pointwise 和 cat 到 `fusedAggregationCatKernel`，消除 `depthwiseWorkspace` global write/read、D2D cat copy 和一次 launch | `53.610 ms` vs baseline `54.331 ms`，`1.0135x` | `4.848 ms/iter`，`6 launches/iter` | `fusedAggregationCatKernel 3.536 ms`、`computeVk 0.960 ms`、`computeOutput 0.352 ms` | 有效；中段边界从 Phase 2 baseline `5.443 ms / 38 launches` 降到 `4.848 ms / 6 launches`，证明 P1b 边界有价值 |
| P1b-2 | 在 P1b-1 基础上，每个 CTA 将当前 aggregation group 的 `16x16` grouped pointwise 权重缓存到 shared memory，再由空间线程复用 | `53.530 ms` vs baseline `54.306 ms`，`1.0145x` | `4.347 ms/iter`，`6 launches/iter` | `fusedAggregationCatKernel 3.038 ms`、`computeVk 0.958 ms`、`computeOutput 0.352 ms` | 有效但收益有限；减少 pointwise weight 重复读取后 aggregation kernel 下降约 `0.498 ms/iter`，中段边界相对 Phase 2 baseline 达到 `1.252x` kernel-time speedup |
| P1b-3 probe | 在 P1b-2 基础上，为非边界像素增加 depthwise 5x5 interior fast path，去掉 `ih/iw` 越界判断；边界像素保留安全路径 | `54.710 ms` vs baseline `54.312 ms`，`0.9927x` | 未采集 Nsight；冷机 benchmark 已显示退化 | 不适用 | 不采纳；去掉边界判断没有抵消代码膨胀、分支分流或寄存器/指令压力，主线代码已恢复到 P1b-2 |

P1b-2 的第一次热机 benchmark 曾显示明显负收益：baseline p50 `54.585 ms`，Plugin p50 `59.144 ms`。冷机重测后转为 baseline p50 `54.306 ms`，Plugin p50 `53.530 ms`。因此该热机样本只作为测量纪律提醒，不作为性能结论。

P1b 当前结论：

1. **扩大边界是有价值的**。P1b-2 把 `aggregation + attention_core` proxy 从 Phase 2 baseline 的 `5.443 ms / 38 launches` 降到 `4.347 ms / 6 launches`。
2. **P1b 仍未明显优于 P1a proxy**。P1a-3b 的 `aggregation + plugin` proxy 约 `3.062 ms / 30 launches`；P1b-2 虽 launch 数更少，但自写 aggregation 仍比 TensorRT/cuDNN 保留的 aggregation 路径更重。
3. **下一步若继续 P1b，应继续盯住 `fusedAggregationCatKernel`**。P1b-2 后它仍约 `3.038 ms/iter`，占 P1b Plugin layer 约 `69.88%`。
4. **P1b 结果要冷机/顺序复核**。整网 1ms 级收益对 MX250 温度、频率和 Windows 调度很敏感，热机样本不能直接写成结论。
5. **P1b-3 interior fast path 不采纳**。该变体 correctness 通过，但端到端 p50 退化到 `0.9927x`，说明在当前 kernel 中，边界判断不是主要瓶颈。

P1b-2 相关文件：

- [`../results/metrics/p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_summary.md)
- [`../results/metrics/p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.md)
- [`../results/metrics/p1b_aggregation_attention_plugin_interior_fastpath_engine_benchmark_summary.md`](../results/metrics/p1b_aggregation_attention_plugin_interior_fastpath_engine_benchmark_summary.md)

---

## 9. 下一步候选

若继续优化 P1a / P1b，优先级建议如下：

| Priority | 候选 | 目标 | 风险 |
|---|---|---|---|
| P1a-2 | VK 归约 kernel 硬件指标采集 | 判断 VK 归约 kernel 是 memory-bound、occupancy-bound 还是归约开销主导 | Nsight Compute 2024.1.1 不支持 MX250；已改用 `nvprof` 完成定性判断 |
| P1a-3c | 继续改进 computeVk 跨 N 归约策略 | 在 P1a-3b 基础上进一步降低 `computeVkKernelDim16WarpD4` 或验证更稳定端到端表现 | 收益递减明显，且端到端 1ms 级差异高度受温度/频率影响；优先级应低于 P1b 评估 |
| P1a-4 | 评估两阶段合并为单 kernel 的可行性 | 消除 workspace global write/read 和一次 launch | 已完成可行性评估，见 [`p1a_single_kernel_feasibility.md`](p1a_single_kernel_feasibility.md)；当前不作为主线采用 |
| P1b-3 | depthwise 5x5 interior fast path | 去掉非边界像素的越界判断 | 已 probe，端到端 `0.9927x`，不采纳 |
| P1b-4 | 继续优化 `fusedAggregationCatKernel` | 降低当前 P1b 内部主瓶颈 | P1b-2 后仍约 `3.038 ms/iter`；下一步不应继续只做边界判断优化，可考虑 depthwise tile/halo、通道并行重排或更细的线程映射，但复杂度会明显上升 |
| P1b-cooldown-rerun | 冷机重复 benchmark | 区分真实收益和热机/频率偏置 | MX250 整网 1ms 级差异敏感；P1b-2 已出现热机负、冷机正的反例 |

关键提醒：两阶段合并并不是“直接把两个 kernel 写进一个 kernel”就能正确，因为 `computeOutputKernel` 依赖完整 VK 归约结果，而完整 VK 结果通常需要跨 CTA 同步。若要合并，需要重新设计每个 CTA 的职责范围，或接受重复计算 VK 的代价。

### 9.1 P1a-4 评估结论

P1a-4 的完整分析见 [`p1a_single_kernel_feasibility.md`](p1a_single_kernel_feasibility.md)。简要结论是：

1. 当前两阶段 kernel boundary 不是纯开销，它同时提供了 `computeVk -> computeOutput` 之间的全局同步。
2. VK workspace 只有约 `8704 bytes`，workspace global write/read 本身不是最大开销。
3. naive single-kernel 方案要么降低 output 并行度，要么重复大量 VK 归约，要么引入高风险 device-side barrier / cooperative launch。
4. 因此 P1a-4 记录为 `evaluated, not adopted as mainline`；下一主线应优先评估 P1b `aggregation + cat + relu_linear_att`。

---

## 10. P1a-2 硬件指标采集记录

2026-06-14 尝试使用 Nsight Compute 2024.1.1 采集 `computeVkKernelDim16` 的 `basic` set：

```powershell
ncu --target-processes all `
  --kernel-name regex:computeVkKernelDim16 `
  --launch-skip 20 `
  --launch-count 4 `
  --set basic `
  ...
```

结果：

- 放开 GPU performance counters 前，失败原因是 `ERR_NVGPUCTRPERM`。
- 按 NVIDIA Control Panel 的 Developer 设置放开 performance counters 后，权限错误消失。
- 随后 Nsight Compute 报告：`Profiling is not supported on device 0.`
- `ncu --list-chips` 只列出 `gv100 / tu* / ga* / ad* / gh100` 等 Volta 及更新架构，未包含 Pascal `sm_61`。

结论：当前 MX250 (`sm_61`) 与 Nsight Compute 2024.1.1 的组合无法采集 Nsight Compute 硬件 counter。因此 Nsight Compute 路线在当前机器上 blocked。后续如果需要 Nsight Compute 级别的完整指标，有两条路：

1. 换用支持 Pascal 的旧版 profiling 工具或旧版 Nsight Compute（需要单独验证可用性）。
2. 在支持 Nsight Compute 的 Turing/Ampere/Ada GPU 上复现实验。

在当前 MX250 上继续优化时，只能依赖：

- Nsight Systems 的 kernel time / launch count / NVTX attribution；
- CUDA Event 单层 microbenchmark；
- CUDA 12.4 `nvprof` 的 Pascal metric 采集；
- 静态 launch 配置、寄存器/共享内存估算；
- 小步 kernel 变体 A/B 实测。

### 10.1 nvprof 采集结果

`nvprof` 本身仍可对 MX250 采集 Pascal 指标，但 CUDA 12.4 的 `nvprof.exe` 启动前必须把 CUPTI DLL 路径加入 `PATH`：

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\CUPTI\lib64;" + $env:PATH
```

否则 `nvprof.exe` 会因找不到 CUPTI DLL 以 `0xC0000135` 静默失败。

本轮采集对象是 P1a-3a 的 `computeVkKernelDim16Warp4`，raw CSV 与摘要记录在：

- [`../results/metrics/nvprof_p1a3a_vk_occupancy.csv`](../results/metrics/nvprof_p1a3a_vk_occupancy.csv)
- [`../results/metrics/nvprof_p1a3a_vk_memory.csv`](../results/metrics/nvprof_p1a3a_vk_memory.csv)
- [`../results/metrics/nvprof_p1a3a_vk_stall_inst.csv`](../results/metrics/nvprof_p1a3a_vk_stall_inst.csv)
- [`../results/metrics/nvprof_p1a3a_vk_transactions.csv`](../results/metrics/nvprof_p1a3a_vk_transactions.csv)
- [`../results/metrics/nvprof_p1a3a_vk_summary.md`](../results/metrics/nvprof_p1a3a_vk_summary.md)

关键指标：

| 指标 | Avg | 判断 |
|---|---:|---|
| `achieved_occupancy` | `0.969826` | occupancy 很高，不像 occupancy-bound |
| `sm_efficiency` | `99.13%` | SM 基本持续活跃 |
| `issue_slot_utilization` | `29.87%` | issue 槽利用率偏低，存在等待 |
| `dram_read_throughput` | `22.33 GB/s` | DRAM 读吞吐中等，不像带宽打满 |
| `l2_read_throughput` | `218.69 GB/s` | L2 读压力明显 |
| `gld_efficiency` | `100%` | global load 合并效率好 |
| `stall_memory_dependency` | `69.48%` | 主导 stall |
| `stall_sync` | `0%` | 不是同步等待主导 |
| `local_memory_overhead` | `0%` | 无明显 register spill |

因此当前 `computeVkKernelDim16Warp4` 更接近 **memory-dependency / load-latency dominated**，而不是 pure DRAM bandwidth-bound、occupancy-bound、sync/reduction-barrier-bound 或 compute-pipe-bound。

设计影响：下一轮如果继续 P1a，应优先考虑减少或隐藏 K/V 读取依赖，例如小步尝试“单个 warp 同时累加 4 个 `d`”来复用 `V[row,n]` load；但这会改变并行粒度，必须用 microbenchmark + Nsight Systems/nvprof 再验证。
