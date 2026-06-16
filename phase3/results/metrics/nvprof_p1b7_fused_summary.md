# P1b-7 `fusedAggregationCatKernel` nvprof Summary

本文件记录 P1b-7 采纳版本中 `fusedAggregationCatKernel` 的 Pascal / MX250 侧硬件指标。Nsight Compute 2024.1.1 不支持 MX250 (`sm_61`)，因此这里使用 CUDA 12.4 `nvprof` 作为旧工具补充。

## 采集对象

| 项 | 值 |
|---|---|
| Kernel | `fusedAggregationCatKernel` |
| Plugin 版本 | P1b-7 CTA512 / 4-row tile |
| GPU | NVIDIA GeForce MX250 (`sm_61`) |
| 指标来源 | `nvprof --metrics` |
| raw CSV | `nvprof_p1b7_fused_occupancy.csv`, `nvprof_p1b7_fused_memory.csv`, `nvprof_p1b7_fused_stall_inst.csv`, `nvprof_p1b7_fused_transactions.csv` |

## 核心指标

| 指标 | 平均值 | 解读 |
|---|---:|---|
| achieved occupancy | `0.496` | CTA512 下不是 occupancy collapse |
| SM efficiency | `99.62%` | GPU 基本持续有工作，不是大面积空洞 |
| issue slot utilization | `47.87%` | 发射槽利用率中等，仍有调度/依赖空间 |
| DRAM read throughput | `7.53 GB/s` | 远低于纯 DRAM 带宽瓶颈形态 |
| DRAM write throughput | `13.62 GB/s` | 写带宽也不是主要上限 |
| L2 utilization | `Low (1)` | L2 压力不高 |
| gld efficiency | `86.17%` | global load 合并情况尚可 |
| gst efficiency | `100.00%` | global store 合并良好 |
| shared efficiency | `41.71%` | shared memory 访问形态存在低效信号 |
| stall exec dependency | `53.08%` | 当前最大 stall 类别 |
| stall memory dependency | `17.56%` | 有内存依赖，但不是主导 |
| local memory overhead | `0.00%` | 没有明显寄存器 spill 到 local memory |

## 结论

1. `fusedAggregationCatKernel` 不是纯 DRAM bandwidth-bound。DRAM/L2 利用率都不高，global store 合并良好。
2. 也不是 occupancy-bound。`achieved_occupancy ~= 0.5` 且 `sm_efficiency ~= 99.6%`，说明 GPU 没有因为 block 太少或 occupancy 太低而长期闲置。
3. 当前更像是 instruction / dependency scheduling 主导，`stall_exec_dependency` 明显高于 `stall_memory_dependency`。
4. `shared_efficiency ~= 41.7%` 提示 shared memory access pattern 有低效可能，但它不等于简单的“改 row pitch 就会变快”。P1b-11a 已验证 `kDepthwiseTileWidth=133` 没有实质改善 shared efficiency，也没有稳定端到端收益。

## P1b-11a 对照

P1b-11a 将 `kDepthwiseTileWidth` 从 `132` 改为 `133`，试图用 row pitch padding 改善 shared memory bank / stride 行为：

| 指标 | P1b-7 | P1b-11a |
|---|---:|---:|
| shared efficiency | `41.7087%` | `41.7221%` |
| shared load transactions | `3,244,032` | `3,244,032` |
| shared store transactions | `105,600` | `106,368` |
| global load transactions | `2,423,810` | `2,435,330` |
| local memory overhead | `0.00%` | `0.00%` |
| end-to-end plugin p50 | `52.311 ms` | `52.268 ms` |
| end-to-end plugin mean | `52.592 ms` | `52.522 ms` |

判断：P1b-11a 的 p50 改善只有 `0.043 ms`，且 shared efficiency 几乎不变、global load transactions 反而增加。因此该 probe 记录为 `evaluated, not adopted`，主线保持 P1b-7 的 `kDepthwiseTileWidth=132`。
