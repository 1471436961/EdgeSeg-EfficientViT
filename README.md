# EdgeSeg-EfficientViT

EdgeSeg-EfficientViT 是一个基于 MIT Han Lab EfficientViT-Seg-B0 的边缘推理优化项目。项目目标不是简单复用上游模型，而是围绕语义分割推理链路，完成 PyTorch 基线、Nsight Systems 瓶颈归因、TensorRT 部署准备与 LiteMLA 自定义 Plugin 设计。

本仓库 fork 自 EfficientViT。原上游 README 已完整保留在 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。

## 结果速览

一句话：本项目把 EfficientViT-Seg-B0 从 PyTorch profiling 推进到 TensorRT baseline，再落到可运行的 LiteMLA TensorRT Plugin；最终证明 TensorRT 未自动整体融合的非标准线性注意力子路径可以被自定义 CUDA/TensorRT Plugin 正确替换，并在 MX250 上获得稳定正收益。

| 维度 | 结果 |
|---|---:|
| PyTorch FP32 baseline p50 | 85.70 ms |
| TensorRT FP32 baseline p50 | 54.44 ms |
| TensorRT vs PyTorch p50 speedup | 1.57x |
| P1a stage2+stage3 Plugin p50 | 50.8380 ms |
| Plugin vs TensorRT p50 speedup | 1.0701x |
| stage2 `relu_linear_att` proxy speedup | 2.819x |
| Cityscapes val baseline mIoU | 75.6463126% |
| Cityscapes val Plugin mIoU | 75.6463248% |

## 项目目标

本项目面向 AI 推理优化 / 机器人边缘部署相关岗位，重点展示：

- 可复现的 PyTorch baseline 推理与 CUDA Events 计时。
- 基于 Nsight Systems、NVTX 和 SQLite attribution 的 GPU kernel 耗时归因。
- 区分“当前 PyTorch 最大热点”和“最适合写 TensorRT Plugin 的非标准算子链”。
- 为 EfficientViT 的 LiteMLA 模块实现并验证 TensorRT / CUDA Plugin 主线。

## 当前进度

| 阶段 | 状态 | 主要产出 |
|---|---|---|
| Phase 1：PyTorch baseline + Nsight 剖析 | 已完成 | [`phase1/bottleneck_analysis_report.md`](./phase1/bottleneck_analysis_report.md) |
| Phase 2：ONNX / TensorRT baseline | 已完成 | [`phase2/tensorrt_baseline_report.md`](./phase2/tensorrt_baseline_report.md) |
| Phase 3：TensorRT Plugin | 已完成主线验证与集成报告 | [`phase3/integration_validation_report.md`](./phase3/integration_validation_report.md) |

## 推荐阅读路径

- **30 秒看结果**：先读本文件的“结果速览”和“当前进度”。
- **5 分钟看闭环**：继续读三段 Phase 摘要，再跳到 [`phase3/integration_validation_report.md`](./phase3/integration_validation_report.md)。
- **技术深挖**：按顺序读 [`phase1/bottleneck_analysis_report.md`](./phase1/bottleneck_analysis_report.md)、[`phase2/tensorrt_baseline_report.md`](./phase2/tensorrt_baseline_report.md)、[`phase3/integration_validation_report.md`](./phase3/integration_validation_report.md)。
- **工程过程与取舍**：读 [`PROJECT_DECISION_CORRECTIONS.md`](./PROJECT_DECISION_CORRECTIONS.md) 和 [`phase3/design_notes/plugin_kernel_optimization_history.md`](./phase3/design_notes/plugin_kernel_optimization_history.md)。
- **学习沉淀**：读 [`LEARNING_LOG.md`](./LEARNING_LOG.md)。

## Phase 1 摘要

Phase 1 在 NVIDIA GeForce MX250 上，以 Cityscapes 分辨率 `1024x2048` 剖析 EfficientViT-Seg-B0。

干净 PyTorch baseline：

| 指标 | 数值 |
|---|---:|
| Mean latency | 85.76 ms |
| P50 latency | 85.70 ms |
| P95 latency | 86.51 ms |
| P99 latency | 87.63 ms |
| Peak allocated memory | 1378 MB |

核心结论：

- 当前 PyTorch GPU kernel 最大热点是代码/NVTX `stage0`，主要来自早期高分辨率 MBConv / Conv 计算。
- `stage2/context` LiteMLA 不是全模型最大端到端瓶颈，但它是最高区分度的 TensorRT Plugin 主线，因为它是 TensorRT 难以自动融合的非标准线性注意力路径。
- Plan D 将 LiteMLA Plugin 候选细化为三类边界；Phase 3 实测后，当前主线收敛为 P1a-3b stage2+stage3：
  - `relu_linear_att-only`：Phase 3 最终 MVP，stage2+stage3 四个 LiteMLA context block 均已覆盖；最终采用 P1a-3b 两阶段 FP32 CUDA 实现，核心为 `computeVkKernelDim16WarpD4 + computeOutputKernelDim16`。
  - `aggregation-only`：保留为 fallback / 对照实验。
  - `aggregation + cat + relu_linear_att`：P1b 重要消融和后续候选，真实 contract 为 `[1,192,64,128] -> [1,128,64,128]`。P1b-7 是 stage2-only 扩大边界实验，应优先和 stage2-only / 中段 proxy 口径比较；最终主线取舍由 P1mix（stage2=P1b-7、stage3=P1a）对照 P1a-3b stage2+stage3 决定。
  - 整体 LiteMLA：复杂度更高的 fallback / 上限方案。

完整报告见 [`phase1/bottleneck_analysis_report.md`](./phase1/bottleneck_analysis_report.md)。

## Phase 2 摘要

Phase 2 已完成固定 `1024x2048` 输入下的 ONNX 导出、ONNXRuntime 对齐、TensorRT 8.6.1 FP32 / FP16 engine 构建与 benchmark、TensorRT Nsight 复核、EngineInspector 结构分析、C++ Runtime Demo 与阶段报告。

当前 TensorRT 结果：

| 项目 | 结果 |
|---|---:|
| PyTorch Plan A formal p50 | 85.70 ms |
| TensorRT FP32 p50 | 54.44 ms |
| TensorRT FP32 speedup | 1.57x |
| TensorRT FP16 p50 | 59.39 ms |
| FP16 结论 | 可构建且语义一致，但慢于 FP32 |

Phase 2 的关键复核结论：

- TensorRT Nsight Systems profiling / attribution 已完成，复核了 Phase 1 的 `stage0`、`stage2 LiteMLA`、`head` 候选在 TensorRT 自动优化后的 residual hotspot 排序。
- TensorRT C++ Runtime Demo 已完成，验证 FP32 engine 能被 C++ Runtime API 加载和执行，为 Phase 3 Plugin 集成铺路。
- Phase 3 已围绕 LiteMLA 完成 P1a Plugin 主线验证，并补充 P1b/P1mix 消融与 `phase3/integration_validation_report.md`。

Phase 2 不以完整 Cityscapes mIoU 为验收条件；当前精度口径是 PyTorch / ONNXRuntime / TensorRT 的转换一致性验证，包括 logits diff、relaxed allclose 和 argmax pixel agreement。

## Phase 3 摘要

Phase 3 已完成 LiteMLA TensorRT Plugin 主线验证。最终主线是 P1a-3b stage2+stage3 `relu_linear_att-only` 两阶段 FP32 Plugin，覆盖 `stage2+stage3` 四个 LiteMLA `context_module/main` block，并真实集成到 EfficientViT-Seg-B0 TensorRT engine 中。

这条 Plugin 主线的价值不只在端到端 `1.0701x` 收益，而在于完整证明了：TensorRT 未自动融合的 LiteMLA 非标准子路径可以被手写 CUDA/TensorRT Plugin 替换，且能同时通过 latency、Nsight attribution 和 Cityscapes mIoU gate。它更适合展示“非标准算子分析 + CUDA kernel 设计 + TensorRT Plugin 集成”的能力，而不是单纯追求最大端到端加速。

核心结果：

| 项目 | 结果 |
|---|---:|
| TensorRT FP32 baseline p50 | 54.3995 ms |
| P1a stage2+stage3 Plugin p50 | 50.8380 ms |
| 端到端 p50 speedup | 1.0701x |
| stage2 `relu_linear_att` proxy speedup | 2.819x |
| Cityscapes val baseline mIoU | 75.6463126% |
| Cityscapes val Plugin mIoU | 75.6463248% |

Phase 3 的关键结论：

- P1a Plugin 通过 TensorRT build / runtime correctness / Nsight attribution / Cityscapes mIoU gate。
- TensorRT 没有把 LiteMLA 自动融合成单一算子，`relu_linear_att` 仍有可优化 residual path。
- P1b-7 证明 `aggregation + cat + relu_linear_att` 在 stage2-only 中段边界上有价值，但不能直接替代 P1a stage2+stage3。
- P1mix（stage2=P1b-7，stage3=P1a）未稳定优于 P1a stage2+stage3，因此 P1b/P1mix 保留为消融和后续候选。

完整报告见 [`phase3/integration_validation_report.md`](./phase3/integration_validation_report.md)。

## 我的新增工作

项目级文档：

- [`PROJECT_STRATEGY.md`](./PROJECT_STRATEGY.md)：项目战略、阶段规划与优化候选排序。
- [`PROJECT_CONVENTIONS.md`](./PROJECT_CONVENTIONS.md)：AI 协作契约与文档规则。
- [`PROJECT_DECISION_CORRECTIONS.md`](./PROJECT_DECISION_CORRECTIONS.md)：跨阶段设计纠偏总账，记录人工 review 如何修正关键方案。
- [`LEARNING_LOG.md`](./LEARNING_LOG.md)：学习笔记与技术问答沉淀。

Phase 1 实现与分析：

- [`phase1/scripts/baseline_inference.py`](./phase1/scripts/baseline_inference.py)：可复现 PyTorch baseline 脚本，支持 CUDA Events 计时与 NVTX Plan A/B/C/D。
- [`phase1/scripts/analyze_nsys_attribution.py`](./phase1/scripts/analyze_nsys_attribution.py)：Nsight SQLite attribution 脚本，基于 CUDA runtime/kernel `correlationId` 归因 GPU kernel 耗时。
- [`phase1/architecture_analysis.md`](./phase1/architecture_analysis.md)：EfficientViT-Seg-B0 源码级架构精读，并根据 profiling 结果修订。
- [`phase1/bottleneck_analysis_report.md`](./phase1/bottleneck_analysis_report.md)：Phase 1 最终瓶颈分析与融合机会报告。
- [`phase1/design_notes/phase1_decision_corrections.md`](./phase1/design_notes/phase1_decision_corrections.md)：关键设计纠偏与人工 review 记录。

Phase 2 实现与部署：

- [`phase2/README.md`](./phase2/README.md)：Phase 2 任务清单、环境口径与当前 TensorRT 结果。
- [`phase2/scripts/export_onnx.py`](./phase2/scripts/export_onnx.py)：固定 shape ONNX 导出与 ONNXRuntime 对齐。
- [`phase2/scripts/build_trt_engine.py`](./phase2/scripts/build_trt_engine.py)：TensorRT FP32 / FP16 engine 构建。
- [`phase2/scripts/benchmark_trt_engine.py`](./phase2/scripts/benchmark_trt_engine.py)：TensorRT engine execute-only latency benchmark 与 PyTorch logits 对齐。
- [`phase2/scripts/inspect_trt_engine.py`](./phase2/scripts/inspect_trt_engine.py)：EngineInspector / ONNX node name 映射，补充 TensorRT 结构层面的 fusion 证据。
- [`phase2/design_notes/trt_nsys_attribution_design.md`](./phase2/design_notes/trt_nsys_attribution_design.md)：TensorRT 后候选复核的 Nsight attribution 设计。
- [`phase2/design_notes/phase2_decision_corrections.md`](./phase2/design_notes/phase2_decision_corrections.md)：Phase 2 关键设计纠偏记录。

Phase 3 Plugin 实现与验证：

- [`phase3/README.md`](./phase3/README.md)：Phase 3 任务清单、当前主线、P1a/P1b/P1mix 口径与验证状态。
- [`phase3/plugin/`](./phase3/plugin/)：TensorRT Plugin C++ / CUDA 实现，包含 P1a `relu_linear_att-only` 与 P1b 扩大边界消融代码。
- [`phase3/scripts/`](./phase3/scripts/)：Plugin engine 构建、benchmark、Nsight attribution、mIoU gate 与结果汇总脚本。
- [`phase3/integration_validation_report.md`](./phase3/integration_validation_report.md)：Phase 3 集成验证报告，汇总 latency、Nsight、mIoU 与候选取舍。
- [`phase3/design_notes/plugin_kernel_optimization_history.md`](./phase3/design_notes/plugin_kernel_optimization_history.md)：P1a/P1b kernel 优化历程与实测记录。
- [`phase3/design_notes/phase3_decision_corrections.md`](./phase3/design_notes/phase3_decision_corrections.md)：Phase 3 关键设计纠偏记录。

## 仓库结构

```text
.
├── README.md                         # 当前文件：项目入口，展示我的工作与当前状态
├── UPSTREAM_README.md                # MIT Han Lab EfficientViT 原始 README
├── PROJECT_STRATEGY.md               # 项目战略与阶段规划
├── PROJECT_CONVENTIONS.md            # AI 协作与文档契约
├── PROJECT_DECISION_CORRECTIONS.md   # 跨阶段设计纠偏总账
├── LEARNING_LOG.md                   # 学习笔记与技术问答沉淀
├── phase1/                           # Phase 1 profiling、报告、截图、脚本
├── phase2/                           # Phase 2 ONNX / TensorRT 部署、benchmark 与 Nsight 复核
├── phase3/                           # Phase 3 TensorRT Plugin、CUDA kernel、mIoU gate 与集成报告
├── efficientvit/                     # 上游 EfficientViT 源码
├── applications/                     # 上游应用示例与文档
├── assets/                           # 上游资源与文档资产
├── setup.py, pyproject.toml          # 上游打包元数据
└── requirements.txt                  # 上游依赖；项目环境说明见 phase 文档
```

## 上游文件如何处理

本项目保留上游代码结构，而不是删除无关目录来“伪装成从零项目”。

- 保留 `efficientvit/`、`applications/`、`assets/`、`setup.py`、`pyproject.toml`、`requirements.txt` 和 `LICENSE`，用于复现、溯源和保留上游上下文。
- 不为了让仓库变小而删除上游其它 application；这些内容说明了模型来源，也避免破坏潜在 import / 示例路径。
- 将 `phase1/`、`phase2/`、`phase3/`、`PROJECT_STRATEGY.md`、`PROJECT_CONVENTIONS.md`、`PROJECT_DECISION_CORRECTIONS.md`、`LEARNING_LOG.md` 作为个人工作主入口。
- 如果后续需要做更干净的作品集发布包，应作为单独 release / packaging 步骤处理，而不是在开发仓库中静默删除上游上下文。

一句话：默认入口展示 EdgeSeg 的个人工作，上游来源保留且可追溯。

## 快速入口

- Phase 1 计划与进度：[`phase1/README.md`](./phase1/README.md)
- Phase 1 瓶颈分析报告：[`phase1/bottleneck_analysis_report.md`](./phase1/bottleneck_analysis_report.md)
- Phase 2 计划与进度：[`phase2/README.md`](./phase2/README.md)
- Phase 2 TensorRT baseline 报告：[`phase2/tensorrt_baseline_report.md`](./phase2/tensorrt_baseline_report.md)
- Phase 3 计划与进度：[`phase3/README.md`](./phase3/README.md)
- Phase 3 集成验证报告：[`phase3/integration_validation_report.md`](./phase3/integration_validation_report.md)
- Phase 3 kernel 优化历史：[`phase3/design_notes/plugin_kernel_optimization_history.md`](./phase3/design_notes/plugin_kernel_optimization_history.md)
- 设计纠偏总账：[`PROJECT_DECISION_CORRECTIONS.md`](./PROJECT_DECISION_CORRECTIONS.md)
- baseline 脚本设计：[`phase1/design_notes/baseline_inference_design.md`](./phase1/design_notes/baseline_inference_design.md)
- Nsight attribution 汇总：[`phase1/results/metrics/`](./phase1/results/metrics/)
- 上游 EfficientViT README：[`UPSTREAM_README.md`](./UPSTREAM_README.md)

## 上游归因

本项目基于 MIT Han Lab 的 EfficientViT 代码库。上游项目说明、论文链接、模型介绍和原始使用方式见 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。本仓库新增工作的重点是边缘推理 profiling、GPU kernel attribution 方法论，以及面向 TensorRT Plugin 的优化候选设计。
