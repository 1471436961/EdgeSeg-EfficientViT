# Phase 3 — TensorRT LiteMLA Plugin

> **阶段目标**：在 Phase 1 PyTorch/Nsight attribution 与 Phase 2 TensorRT baseline 的证据基础上，设计并实现一个面向 EfficientViT `stage2/context` LiteMLA 的 TensorRT Plugin MVP，验证自定义 C++/CUDA/TensorRT Plugin 是否能进一步优化 TensorRT 未自动整体融合的非标准线性注意力路径。
>
> **当前状态**：Phase 3 刚启动。本阶段第一步只建立目录骨架和 Plugin 融合设计文档，不直接写 CUDA / TensorRT Plugin 代码。

---

## 1. 阶段边界

Phase 3 做：

- 设计 LiteMLA Plugin 的输入输出 tensor contract。
- 先实现最小可行 Plugin MVP，优先验证 `relu_linear_att-only` 或 `aggregation-only` 的接入链路。
- 将 Plugin 集成到 TensorRT engine 构建 / runtime 路径中。
- 复用 Phase 2 benchmark 与 C++ runtime demo 验证 correctness、latency 和 Nsight attribution。
- 在成功 MVP 基础上评估 `aggregation + cat + relu_linear_att` 中段组合边界。

Phase 3 暂不做：

- 不一开始实现整体 LiteMLA Plugin。
- 不把 stage0/head MBConv 作为第一主线。
- 不做完整 ROS 2 节点。
- 不把 INT8 / 混合精度作为第一交付物。

---

## 2. 输入证据

| 证据 | 文件 | 对 Phase 3 的作用 |
|---|---|---|
| Phase 1 bottleneck report | [`../phase1/bottleneck_analysis_report.md`](../phase1/bottleneck_analysis_report.md) | 给出 PyTorch 路径下的 Plugin 候选边界 |
| Phase 1 Plan D attribution | [`../phase1/results/metrics/planD_nsys_attribution_summary.md`](../phase1/results/metrics/planD_nsys_attribution_summary.md) | 证明 `aggregation` / `relu_linear_att` / 中段组合的耗时结构 |
| Phase 2 TensorRT report | [`../phase2/tensorrt_baseline_report.md`](../phase2/tensorrt_baseline_report.md) | 复核 TensorRT 后 LiteMLA residual runtime |
| TensorRT engine inspection | [`../phase2/results/metrics/trt_engine_inspection_summary.md`](../phase2/results/metrics/trt_engine_inspection_summary.md) | 说明 TensorRT 未把 LiteMLA 自动融合成单一算子 |
| TensorRT C++ demo | [`../phase2/cpp_demo/README.md`](../phase2/cpp_demo/README.md) | 作为后续 Plugin engine runtime 验证起点 |

---

## 3. Plugin 候选排序

| 优先级 | 候选 | 目标 | 当前判断 |
|---|---|---|---|
| P1a | `relu_linear_att-only` | MVP / 接入验证 | 边界小，最适合先验证 Plugin 注册、构建、runtime 和数值对齐 |
| P1a | `aggregation-only` | MVP / 对照验证 | 耗时接近 `relu_linear_att`，但更接近卷积分支，需确认 TensorRT 已优化程度 |
| P1b | `aggregation + cat + relu_linear_att` | 主性能评估边界 | Phase 1/2 都支持的中段组合候选，潜在收益更高但 graph 集成更复杂 |
| P1c | 整体 LiteMLA | fallback / 上限方案 | 融合空间最大，但实现、调试和数值验证风险最高，不作为第一步 |
| P2 | stage0/head MBConv | 工程优化候选 | 端到端收益潜力高，但标准算子链多，展示区分度低于 LiteMLA |

---

## 4. 任务清单

- [x] Step 0：从 `master` 创建 `phase3-plugin` 分支。
- [x] Step 1：建立 Phase 3 目录骨架与 `plugin_fusion_design.md` 第一版。
- [ ] Step 2：精读 ONNX / TensorRT engine 中 `stage2/context` 的实际 tensor 边界，确定 P1a MVP 的输入输出。
- [ ] Step 3：设计最小 Plugin API 与 CMake 构建方案。
- [ ] Step 4：实现 P1a Plugin skeleton，先跑通 TensorRT Plugin 注册与 engine build。
- [ ] Step 5：实现 CUDA kernel / enqueue 路径，并做 PyTorch / TensorRT baseline 输出对齐。
- [ ] Step 6：复用 Phase 2 benchmark，比较 TensorRT baseline vs Plugin engine latency。
- [ ] Step 7：采集 Plugin engine Nsight trace，更新 attribution summary。
- [ ] Step 8：撰写 `integration_validation_report.md`。

---

## 5. 目录结构

```text
phase3/
|-- README.md
|-- design_notes/
|   `-- plugin_fusion_design.md
|-- plugin/
|   `-- .gitkeep
|-- scripts/
|   `-- .gitkeep
|-- results/
|   |-- metrics/
|   |   `-- .gitkeep
|   |-- figures/
|   |   `-- .gitkeep
|   `-- nsight/
|       `-- .gitkeep
`-- logs/
    `-- .gitkeep
```

---

## 6. 验收口径

Phase 3 的一次有效 Plugin 实验至少需要同时满足：

- TensorRT engine 可构建并能加载 Plugin。
- 输出与 TensorRT FP32 baseline / PyTorch reference 在约定阈值内对齐。
- latency 使用 Phase 2 相同的 CUDA Events execute-only 口径。
- Nsight attribution 能看到 Plugin 前后 residual hotspot 的变化。
- 报告中清楚区分：
  - 端到端 latency 改善；
  - Plugin 覆盖的 layer / kernel 范围；
  - 数值误差；
  - 工程风险与 fallback 路线。

