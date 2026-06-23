# P1b Probe Metrics Archive

本目录保存 P1b `aggregation + cat + relu_linear_att` 路线的中间 probe 结果。

这些文件不再放在 `phase3/results/metrics/` 顶层，原因是它们不是当前 Phase 3 主交付物。它们仍然是重要证据：

- 证明 naive P1b 为什么会慢于 TensorRT baseline。
- 记录 P1b-1 到 P1b-15 的采纳 / 不采纳原因。
- 支撑 `plugin_kernel_optimization_history.md` 中关于 MX250、shared memory、CTA layout、dependency stall 和测量纪律的结论。

当前顶层 metrics 只保留：

- P1a stage2+stage3 最终主线结果。
- Cityscapes mIoU accuracy gate。
- P1b-7 关键对照结果。
- P1mix 负向消融结果。

完整实验叙事见：

- [`../../../../design_notes/plugin_kernel_optimization_history.md`](../../../../design_notes/plugin_kernel_optimization_history.md)
- [`../../../../design_notes/p1b_aggregation_attention_design.md`](../../../../design_notes/p1b_aggregation_attention_design.md)
