# P1a-3a VK Reduction nvprof Summary

> 采集对象：`computeVkKernelDim16Warp4`
>
> 采集工具：CUDA 12.4 `nvprof` + CUPTI。运行前需临时把 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\CUPTI\lib64` 加入 `PATH`，否则 `nvprof.exe` 会因缺少 CUPTI DLL 以 `0xC0000135` 静默失败。
>
> 采集目的：在 Nsight Compute 2024.1.1 不支持 MX250 (`sm_61`) 的情况下，用 `nvprof` 判断当前 VK 归约 kernel 更接近 memory-bound、occupancy-bound 还是归约/同步开销主导。

---

## 1. Raw Files

| 文件 | 指标组 |
|---|---|
| `nvprof_p1a3a_vk_occupancy.csv` | occupancy / issue / IPC |
| `nvprof_p1a3a_vk_memory.csv` | DRAM / L2 / cache / load-store efficiency |
| `nvprof_p1a3a_vk_stall_inst.csv` | stall reason / instruction mix |
| `nvprof_p1a3a_vk_transactions.csv` | global/local/shared transaction counts |

---

## 2. Key Metrics

| 维度 | 指标 | Avg | 判断 |
|---|---|---:|---|
| Occupancy | `achieved_occupancy` | `0.969826` | 很高，不像 occupancy-bound |
| SM activity | `sm_efficiency` | `99.13%` | SM 基本一直有 warp 活跃 |
| Issue | `eligible_warps_per_cycle` | `1.4575` | 可发射 warp 不多，存在等待 |
| Issue | `issue_slot_utilization` | `29.87%` | 发射槽利用率偏低 |
| Memory | `dram_read_throughput` | `22.33 GB/s` | 中等，不像 DRAM 带宽打满 |
| Memory | `dram_utilization` | `Mid (5)` | 中等利用率 |
| Cache | `l2_read_throughput` | `218.69 GB/s` | L2 读压力明显 |
| Cache | `l2_tex_read_hit_rate` | `89.80%` | L2 hit 较高 |
| Coalescing | `gld_efficiency` | `100%` | global load 合并效率好 |
| Stall | `stall_memory_dependency` | `69.48%` | 主导 stall，说明 load dependency / memory latency 很强 |
| Stall | `stall_sync` | `0%` | 不是同步等待主导 |
| Stall | `stall_pipe_busy` | `0.66%` | 不是计算管线打满 |
| Local memory | `local_memory_overhead` | `0%` | 没有寄存器溢出到 local memory |

---

## 3. Interpretation

当前 `computeVkKernelDim16Warp4` 的瓶颈更接近：

```text
memory-dependency / load-latency dominated
```

而不是：

```text
pure DRAM bandwidth-bound
occupancy-bound
sync/reduction-barrier-bound
compute-pipe-bound
```

依据：

1. `achieved_occupancy ~= 0.97`，`sm_efficiency ~= 99%`，说明 occupancy 和 SM 活跃度都不低。
2. `dram_read_throughput ~= 22.3 GB/s` 且 `dram_utilization = Mid(5)`，没有显示 DRAM 带宽已被打满。
3. `stall_memory_dependency ~= 69.5%`，远高于 execution dependency、pipe busy、sync 等 stall。
4. `local_memory_overhead = 0%`，说明当前版本没有明显 register spill。
5. `gld_efficiency = 100%`，说明不是简单的未合并 global load 问题。

---

## 4. Design Implication

下一步如果继续优化 P1a，不应优先追求更高 occupancy，也不应优先处理同步归约；更应该尝试减少或隐藏 K/V 读取带来的 memory dependency：

1. 继续小步 A/B，而不是一次性大改。
2. 优先考虑在不显著增加 register pressure 的前提下复用 `V[row, n]` 或 `K[d, n]`。
3. 一个合理候选是“单个 warp 同时累加 4 个 `d`”，用少量寄存器换取 `V` load 复用；但它会减少 CTA/warp 数，需要实测确认是否在 MX250 上仍保留足够并行度。
4. 暂不建议直接把两阶段 kernel 合成单 kernel，因为完整 VK 结果仍涉及跨 CTA 全局同步问题。

---

## 5. Caveat

`nvprof` metric 采集会重放 kernel，benchmark wall time 会被严重放大。因此这些 CSV 只用于瓶颈类型判断，不用于报告 Plugin latency；正式 latency 仍以 CUDA Events 和 Nsight Systems attribution 为准。
