# Phase 3 — TensorRT LiteMLA Plugin

> **阶段目标**：在 Phase 1 PyTorch/Nsight attribution 与 Phase 2 TensorRT baseline 的证据基础上，实现并验证 EfficientViT-Seg-B0 LiteMLA 的 TensorRT Plugin，判断自定义 C++/CUDA/TensorRT Plugin 是否能进一步优化 TensorRT 未自动整体融合的非标准线性注意力路径。
>
> **当前主线**：最终采用的 Phase 3 MVP 是 **P1a `relu_linear_att-only` 覆盖 stage2+stage3 四个 LiteMLA context block**。P1b `aggregation + cat + relu_linear_att` 已完成较完整消融，证明扩大边界能降低中段 kernel time / launch 数，但端到端没有稳定优于 P1a stage2+stage3；因此 P1b 保留为实验分支与后续优化候选，不作为当前主交付线。

---

## 1. 当前结论

| 结论 | 证据 |
|---|---|
| P1a stage2+stage3 是当前最稳主线 | `relu_linear_att-only` 从 stage2 两个 block 扩到 stage2+stage3 四个 block 后，端到端 p50 从 Phase 2 baseline 约 `54.40ms` 降到 `50.84ms`，speedup 约 `1.07x` |
| P1a 数值语义通过数据集级验收 | Cityscapes val baseline mIoU `75.6463126%`，Plugin mIoU `75.6463248%`，delta 约 `+0.000012` percentage point，argmax agreement `0.999999918` |
| P1b 是重要消融但不是当前主线 | P1b-7 中段 `aggregation + attention_core` proxy 达到 `1.789x` kernel-time speedup，但端到端 p50 约 `52.31ms`，未稳定打败 P1a stage2+stage3 |
| P1mix 不采纳 | `stage2=P1b-7 + stage3=P1a-3b` 技术链路通过，但 Nsight execute avg 与 selected context total 未稳定优于 P1a stage2+stage3 |
| P1a/P1b 代码都保留 | P1a 是主线；P1b 记录了扩大边界、shared-memory 数据复用、MX250 约束下的正反例，属于有价值消融证据 |

---

## 2. 阶段边界

Phase 3 做：

- 设计 LiteMLA Plugin 的输入输出 tensor contract。
- 实现 P1a `relu_linear_att-only` Plugin，并集成到真实 TensorRT engine。
- 评估 P1a 单层、端到端、Nsight attribution、Cityscapes mIoU。
- 评估 P1b `aggregation + cat + relu_linear_att` 作为扩大边界的消融路线。
- 保留 P1mix 负向消融，避免错误选择更复杂但不稳定收益的边界。

Phase 3 暂不做：

- 不把 P1b 或整体 LiteMLA 作为当前默认主交付物。
- 不把 stage0/head MBConv 作为第一主线。
- 不做完整 ROS 2 节点。
- 不把 INT8 / 混合精度作为第一交付物。

---

## 3. 关键证据入口

| 类别 | 文件 | 作用 |
|---|---|---|
| Phase 3 最终验收报告 | [`integration_validation_report.md`](integration_validation_report.md) | 汇总 P1a stage2+stage3 Plugin 的集成、正确性、latency、Nsight attribution、mIoU 与最终分支决策 |
| Design notes index | [`design_notes/README.md`](design_notes/README.md) | 给出 Phase 3 设计文档阅读顺序，区分主线、P1b 消融和历史反证 |
| Phase 1 瓶颈分析 | [`../phase1/bottleneck_analysis_report.md`](../phase1/bottleneck_analysis_report.md) | 给出 PyTorch 路径下的热点与 Plugin 候选边界 |
| Phase 2 TensorRT baseline | [`../phase2/tensorrt_baseline_report.md`](../phase2/tensorrt_baseline_report.md) | 复核 TensorRT 后 LiteMLA residual runtime，证明 TensorRT 没有自动整体融合 LiteMLA |
| Tensor contract | [`design_notes/stage2_context_tensor_contract.md`](design_notes/stage2_context_tensor_contract.md) | 定义 P1a/P1b/P1c 的真实输入输出边界 |
| Plugin API / CMake | [`design_notes/plugin_api_cmake_design.md`](design_notes/plugin_api_cmake_design.md) | 记录 Plugin 接口、序列化字段、DLL 构建与加载策略 |
| Kernel 优化历史 | [`design_notes/plugin_kernel_optimization_history.md`](design_notes/plugin_kernel_optimization_history.md) | 集中记录 P1a/P1b kernel 演进、采纳/不采纳原因与 MX250 测量纪律 |
| P1a stage2+stage3 设计 | [`design_notes/p1a_all_context_design.md`](design_notes/p1a_all_context_design.md) | 说明为什么先把 P1a 扩展到 stage2+stage3 四个 context block |
| P1a 单 kernel 反证 | [`design_notes/p1a_single_kernel_feasibility.md`](design_notes/p1a_single_kernel_feasibility.md) | 说明为什么不把 P1a 两阶段 kernel 强行合并为单 kernel |
| P1b 设计与消融 | [`design_notes/p1b_aggregation_attention_design.md`](design_notes/p1b_aggregation_attention_design.md) | 记录 P1b parser/build、CUDA 数学、Nsight 与 P1b-1..15 消融 |
| P1mix 消融 | [`design_notes/p1mix_stage2_p1b_stage3_p1a_design.md`](design_notes/p1mix_stage2_p1b_stage3_p1a_design.md) | 记录 P1mix 为什么不采纳 |
| mIoU 设计 | [`design_notes/cityscapes_miou_evaluation_design.md`](design_notes/cityscapes_miou_evaluation_design.md) | 定义 Cityscapes mIoU 口径、数据布局、official/deployment 预处理差异 |

---

## 4. 最终主线产物

| 产物 | 文件 |
|---|---|
| Phase 3 integration validation report | [`integration_validation_report.md`](integration_validation_report.md) |
| P1a Plugin skeleton / toy build | [`results/metrics/relu_linear_attention_toy_build.json`](results/metrics/relu_linear_attention_toy_build.json) |
| P1a 单层 correctness | [`results/metrics/relu_linear_attention_plugin_validation.json`](results/metrics/relu_linear_attention_plugin_validation.json) |
| P1a 单层 microbenchmark | [`results/metrics/relu_linear_attention_plugin_microbenchmark_summary.md`](results/metrics/relu_linear_attention_plugin_microbenchmark_summary.md) |
| P1a stage2+stage3 ONNX integration | [`results/metrics/relu_linear_attention_plugin_stage2_stage3_onnx_integration.json`](results/metrics/relu_linear_attention_plugin_stage2_stage3_onnx_integration.json) |
| P1a stage2+stage3 engine build | [`results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_build.json`](results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_build.json) |
| P1a stage2+stage3 benchmark | [`results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_benchmark_summary.md`](results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_benchmark_summary.md) |
| P1a stage2+stage3 Nsight attribution | [`results/metrics/relu_linear_attention_plugin_stage2_stage3_nsys_attribution_summary.md`](results/metrics/relu_linear_attention_plugin_stage2_stage3_nsys_attribution_summary.md) |
| P1a stage2+stage3 Cityscapes mIoU | [`results/metrics/cityscapes_miou_p1a_stage2_stage3_summary.md`](results/metrics/cityscapes_miou_p1a_stage2_stage3_summary.md) |

---

## 5. 重要消融产物

| 分支 | 结论 | 文件 |
|---|---|---|
| P1b-7 | 中段 kernel-time/launch 明显改善，但端到端未打败 P1a stage2+stage3 | [`results/metrics/p1b_aggregation_attention_plugin_cta512_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_cta512_engine_benchmark_summary.md), [`results/metrics/p1b_aggregation_attention_plugin_cta512_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_cta512_nsys_attribution_summary.md) |
| P1mix | 技术链路通过，但没有稳定优于 P1a stage2+stage3 | [`results/metrics/p1mix_stage2_p1b_stage3_p1a_engine_benchmark_summary.md`](results/metrics/p1mix_stage2_p1b_stage3_p1a_engine_benchmark_summary.md), [`results/metrics/p1mix_stage2_p1b_stage3_p1a_nsys_attribution_summary.md`](results/metrics/p1mix_stage2_p1b_stage3_p1a_nsys_attribution_summary.md) |
| P1b 中间 probe | 作为优化历程和反例保留，不放在 metrics 顶层 | [`results/metrics/archive/p1b_probes/README.md`](results/metrics/archive/p1b_probes/README.md) |

---

## 6. 代码入口

脚本职责索引见 [`scripts/README.md`](scripts/README.md)。当前不移动脚本到子目录，因为脚本间存在同目录 import 和 `__file__` 路径假设；移动会影响复现实验命令。

```text
phase3/
|-- plugin/
|   |-- CMakeLists.txt
|   |-- include/
|   |   |-- edgeseg_relu_linear_attention_plugin.h
|   |   `-- edgeseg_aggregation_relu_linear_attention_plugin.h
|   `-- src/
|       |-- relu_linear_attention_kernel.cu
|       |-- aggregation_relu_linear_attention_kernel.cu
|       |-- edgeseg_relu_linear_attention_plugin.cpp
|       `-- edgeseg_aggregation_relu_linear_attention_plugin.cpp
|-- scripts/
|   |-- build_plugin_engine.py
|   |-- benchmark_plugin_engine.py
|   |-- integrate_relu_linear_attention_plugin_onnx.py
|   |-- evaluate_cityscapes_miou.py
|   |-- analyze_plugin_nsys_attribution.py
|   |-- build_p1b_plugin_engine.py
|   |-- integrate_p1b_aggregation_attention_plugin_onnx.py
|   `-- validate_p1b_aggregation_attention_plugin.py
`-- results/
    |-- metrics/
    |   |-- archive/p1b_probes/
    |   |-- cityscapes_miou_p1a_stage2_stage3_summary.md
    |   |-- relu_linear_attention_plugin_stage2_stage3_engine_benchmark_summary.md
    |   `-- relu_linear_attention_plugin_stage2_stage3_nsys_attribution_summary.md
    `-- engines/onnx/nsight/figures/tensors/
```

---

## 7. 任务清单

- [x] Step 0：从 `master` 创建 `phase3-plugin` 分支。
- [x] Step 1：建立 Phase 3 目录骨架与 `plugin_fusion_design.md`。
- [x] Step 2：确认 LiteMLA tensor contract。
- [x] Step 3：设计 Plugin API 与 CMake 构建方案。
- [x] Step 4：实现 P1a Plugin skeleton，跑通 TensorRT Plugin 注册与 toy engine build。
- [x] Step 5：实现 P1a CUDA kernel / enqueue，并完成单层 correctness。
- [x] Step 5.5：补充 P1a 单层 microbenchmark 与 Nsight/nvprof 证据。
- [x] Step 6：将 P1a 集成进真实 EfficientViT TensorRT graph。
- [x] Step 7：完成 TensorRT baseline vs P1a Plugin engine correctness / latency。
- [x] Step 8：完成 P1a Plugin engine Nsight attribution。
- [x] Step 8.5：完成 P1a VK kernel 硬件指标补充。
- [x] Step 8.6：完成 P1a 单 kernel 合并可行性反证。
- [x] Step 8.7-8.29：完成 P1b parser/build、CUDA 数学、Nsight 与 P1b-1..15 消融。
- [x] Step 8.30：完成 Cityscapes mIoU accuracy gate。
- [x] Step 9：撰写 `integration_validation_report.md`。

---

## 8. 验收口径

Phase 3 的有效 Plugin 结论至少需要同时满足：

- TensorRT engine 可构建并能加载 Plugin。
- 输出与 TensorRT FP32 baseline / PyTorch reference 在约定阈值内对齐。
- latency 使用 Phase 2 相同的 CUDA Events execute-only 口径。
- Nsight attribution 使用 CUDA runtime/kernel correlationId，不用 NVTX range duration 直接当 GPU 组件耗时。
- 最终采用的 Plugin 主线需要 Cityscapes mIoU / semantic regression 证据。
- 报告必须区分端到端收益、Plugin 覆盖范围、数值误差、工程风险与 fallback 路线。
