# Phase 3 Design Notes Index

本目录保存 Phase 3 的设计记录。文件数量较多是因为 Phase 3 不是单一路径开发，而是围绕 P1a、P1b、P1mix 做了多轮验证和消融。

阅读时不需要按文件名逐个看，建议按下面顺序。

## 1. 主线必读

| 文件 | 用途 |
|---|---|
| [`plugin_fusion_design.md`](plugin_fusion_design.md) | Phase 3 初始融合目标、候选边界与交付策略 |
| [`stage2_context_tensor_contract.md`](stage2_context_tensor_contract.md) | P1a/P1b/P1c 的真实 tensor contract；注意其中 P1b 主性能边界是早期设计假设，后续已由实验修正 |
| [`plugin_api_cmake_design.md`](plugin_api_cmake_design.md) | TensorRT Plugin API、序列化字段、CMake/DLL 构建策略 |
| [`plugin_graph_integration_design.md`](plugin_graph_integration_design.md) | P1a ONNX graph surgery 与真实 engine 集成设计 |
| [`plugin_engine_benchmark_design.md`](plugin_engine_benchmark_design.md) | TensorRT baseline vs Plugin engine 的 benchmark 口径 |
| [`plugin_nsys_attribution_design.md`](plugin_nsys_attribution_design.md) | Plugin engine Nsight attribution 口径 |
| [`cityscapes_miou_evaluation_design.md`](cityscapes_miou_evaluation_design.md) | Cityscapes mIoU accuracy gate、数据布局与预处理口径 |

## 2. 优化历史与最终判断

| 文件 | 用途 |
|---|---|
| [`plugin_kernel_optimization_history.md`](plugin_kernel_optimization_history.md) | P1a/P1b kernel 优化总账；包含采纳、不采纳、冷机重测、MX250 约束和硬件指标判断 |
| [`phase3_decision_corrections.md`](phase3_decision_corrections.md) | Phase 3 关键路线纠偏：P1b/P1mix 公平比较、两阶段 kernel、冷机复测和 mIoU gate |
| [`p1a_all_context_design.md`](p1a_all_context_design.md) | 为什么把 P1a 从 stage2 扩展到 stage2+stage3 |
| [`p1a_single_kernel_feasibility.md`](p1a_single_kernel_feasibility.md) | 为什么不把 P1a 两阶段 kernel 强行合并为单 kernel |
| [`p1mix_stage2_p1b_stage3_p1a_design.md`](p1mix_stage2_p1b_stage3_p1a_design.md) | 为什么 P1mix 技术通过但不采纳为主线 |

## 3. P1b 消融专用

| 文件 | 用途 |
|---|---|
| [`p1b_aggregation_attention_design.md`](p1b_aggregation_attention_design.md) | P1b `aggregation + cat + relu_linear_att` 的完整设计、正确性验证、Nsight 结果和 P1b-1..15 消融 |
| [`p1b_single_block_validation_design.md`](p1b_single_block_validation_design.md) | P1b block-level PyTorch reference 捕获和 correctness 验证口径 |

## 4. 当前阅读建议

若目标是理解最终交付结论：

1. 先读 [`../README.md`](../README.md)。
2. 再读 [`plugin_kernel_optimization_history.md`](plugin_kernel_optimization_history.md) 的 P1a stage2+stage3 和 P1mix 小节。
3. 最后读 [`../integration_validation_report.md`](../integration_validation_report.md)。

若目标是复盘为什么 P1b 没成为主线：

1. 读 [`p1b_aggregation_attention_design.md`](p1b_aggregation_attention_design.md)。
2. 对照 [`plugin_kernel_optimization_history.md`](plugin_kernel_optimization_history.md) 的 P1b 优化表。
3. 查看 [`../results/metrics/archive/p1b_probes/README.md`](../results/metrics/archive/p1b_probes/README.md) 中归档的中间结果。
