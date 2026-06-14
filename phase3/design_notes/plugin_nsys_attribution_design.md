# Plugin Nsight Attribution Design

> **状态**：Phase 3 Step 8 执行记录。
>
> **目标**：对 Step 7 已验证可执行的 Plugin engine 采集 Nsight Systems trace，并用 CUDA runtime/kernel `correlationId` 归因方法复核 Plugin 替换后 residual hotspot 的变化。

---

## 1. 设计目标

Step 8 回答的问题不是“Plugin engine 是否能跑”，而是：

1. Plugin engine 中两个 `EdgesegReluLinearAttention_TRT` layer 实际占多少 GPU kernel time。
2. 与 Phase 2 TensorRT baseline 的 stage2/context residual runtime 相比，Plugin 是否减少了目标边界的 kernel time 和 launch 数。
3. Plugin 后续优化应继续围绕 Plugin 内部两个 kernel，还是应转向其他 stage / 更大边界。

---

## 2. 采集口径

Nsight 采集只运行 Plugin engine：

```text
benchmark_plugin_engine.py
  --benchmark-target plugin
  --warmup 20
  --measure 100
  --nvtx
  --skip-reference
```

`--skip-reference` 只跳过 PyTorch reference 构建和输出比较，不改变 TensorRT engine 执行口径。正确性已经由 Step 7 正式 benchmark 验证。

不在同一次 Nsight trace 中同时执行 baseline 和 Plugin engine，原因是这样会让 SQLite 中的 `trt/execute`、TensorRT layer ranges 和 kernel 事件混在一起，降低归因清晰度。

---

## 3. 归因方法

沿用 Phase 1/2 方法论：

```text
TensorRT/NVTX layer range
-> range 内 CUDA runtime launch
-> 通过 correlationId 找到对应 CUDA kernel
-> 汇总 kernel duration
```

不使用 NVTX range 的 `end - start` 作为 GPU 组件耗时。

---

## 4. Plugin 映射规则

Phase 3 Plugin engine 中，两个目标 layer 在 Nsight SQLite 中表现为：

```text
/backbone/stages.2/op_list.1/context_module/main/EdgesegReluLinearAttention_TRT
/backbone/stages.2/op_list.2/context_module/main/EdgesegReluLinearAttention_TRT
```

归因脚本将它们映射为：

```text
relu_linear_att_plugin
```

并定义三个 Step8 proxy boundary：

| Boundary | 含义 |
|---|---|
| `relu_linear_att_plugin_only` | 两个自定义 Plugin layer 的实际 runtime |
| `aggregation_plus_plugin_proxy` | Phase 3 中对 P1b `aggregation + cat + relu_linear_att` 的近似 runtime proxy |
| `full_stage2_context_plugin_path` | `qkv + aggregation + plugin + proj_add` 的 stage2 context runtime |

这些 proxy boundary 是 TensorRT runtime 视角的工程映射，不等同于重新定义 Phase 1 Plan D 的候选。

---

## 5. 实测结果

| 指标 | Phase 2 TensorRT baseline | Phase 3 Plugin engine |
|---|---:|---:|
| `relu_linear_att` proxy / Plugin layer | 3.689 ms, 12 launches | 2.447 ms, 4 launches |
| `aggregation + attention` proxy | 5.443 ms, 38 launches | 4.192 ms, 30 launches |
| stage2/context total | 6.383 ms, 42 launches | 5.226 ms, 35 launches |

Plugin 内部两个 kernel：

| Kernel | Avg ms / iter | Share |
|---|---:|---:|
| `computeVkKernel` | 1.776 | 72.56% |
| `computeOutputKernel` | 0.672 | 27.44% |

---

## 6. 结论

Step 8 支持以下结论：

- Plugin 替换确实减少了目标边界的 kernel time 和 launch 数。
- 端到端收益较小的原因不是 Plugin 完全无效，而是 stage0 / stage2 / stage1 / stage3 / head 等其他标准算子热点仍占据大部分 runtime。
- P0 shared-memory VK cache 已显著降低 `computeOutputKernel`，后续若继续 P1a，应优先看 `computeVkKernel` 的跨 `N` 维归约并行度、memory access / occupancy / launch 开销。
- 若追求更明显端到端收益，P1b `aggregation + cat + relu_linear_att` 仍是下一层边界候选，但 graph surgery 和数值风险更高。
