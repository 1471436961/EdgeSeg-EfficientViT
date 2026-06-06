# EdgeSeg-EfficientViT

EdgeSeg-EfficientViT 是一个基于 MIT Han Lab EfficientViT-Seg-B0 的边缘推理优化项目。项目目标不是简单复用上游模型，而是围绕语义分割推理链路，完成 PyTorch 基线、Nsight Systems 瓶颈归因、TensorRT 部署准备与 LiteMLA 自定义 Plugin 设计。

本仓库 fork 自 EfficientViT。原上游 README 已完整保留在 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。

## 项目目标

本项目面向 AI 推理优化 / 机器人边缘部署相关岗位，重点展示：

- 可复现的 PyTorch baseline 推理与 CUDA Events 计时。
- 基于 Nsight Systems、NVTX 和 SQLite attribution 的 GPU kernel 耗时归因。
- 区分“当前 PyTorch 最大热点”和“最适合写 TensorRT Plugin 的非标准算子链”。
- 为 EfficientViT 的 LiteMLA 模块设计 TensorRT / CUDA Plugin 的阶段三路线。

## 当前进度

| 阶段 | 状态 | 主要产出 |
|---|---|---|
| Phase 1：PyTorch baseline + Nsight 剖析 | 已完成 | [`phase1/bottleneck_analysis_report.md`](./phase1/bottleneck_analysis_report.md) |
| Phase 2：ONNX / TensorRT baseline | 计划中 | ONNX 导出与 TensorRT baseline 脚本 |
| Phase 3：TensorRT Plugin | 计划中 | LiteMLA Plugin MVP 与消融实验 |

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
- Plan D 将 LiteMLA Plugin 候选细化为三类边界：
  - `aggregation-only` 或 `relu_linear_att-only`：适合作为 MVP / 单段验证。
  - `aggregation + cat + relu_linear_att`：更有潜在收益的主性能边界。
  - 整体 LiteMLA：复杂度更高的 fallback / 上限方案。

完整报告见 [`phase1/bottleneck_analysis_report.md`](./phase1/bottleneck_analysis_report.md)。

## 我的新增工作

项目级文档：

- [`PROJECT_STRATEGY.md`](./PROJECT_STRATEGY.md)：项目战略、阶段规划与优化候选排序。
- [`PROJECT_CONVENTIONS.md`](./PROJECT_CONVENTIONS.md)：AI 协作契约与文档规则。
- [`LEARNING_LOG.md`](./LEARNING_LOG.md)：学习笔记与人工 review 纠偏记录。

Phase 1 实现与分析：

- [`phase1/scripts/baseline_inference.py`](./phase1/scripts/baseline_inference.py)：可复现 PyTorch baseline 脚本，支持 CUDA Events 计时与 NVTX Plan A/B/C/D。
- [`phase1/scripts/analyze_nsys_attribution.py`](./phase1/scripts/analyze_nsys_attribution.py)：Nsight SQLite attribution 脚本，基于 CUDA runtime/kernel `correlationId` 归因 GPU kernel 耗时。
- [`phase1/architecture_analysis.md`](./phase1/architecture_analysis.md)：EfficientViT-Seg-B0 源码级架构精读，并根据 profiling 结果修订。
- [`phase1/bottleneck_analysis_report.md`](./phase1/bottleneck_analysis_report.md)：Phase 1 最终瓶颈分析与融合机会报告。
- [`phase1/design_notes/phase1_decision_corrections.md`](./phase1/design_notes/phase1_decision_corrections.md)：关键设计纠偏与人工 review 记录。

## 仓库结构

```text
.
├── README.md                         # 当前文件：项目入口，展示我的工作与当前状态
├── UPSTREAM_README.md                # MIT Han Lab EfficientViT 原始 README
├── PROJECT_STRATEGY.md               # 项目战略与阶段规划
├── PROJECT_CONVENTIONS.md            # AI 协作与文档契约
├── LEARNING_LOG.md                   # 学习笔记与纠偏沉淀
├── phase1/                           # Phase 1 profiling、报告、截图、脚本
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
- 将 `phase1/`、`PROJECT_STRATEGY.md`、`PROJECT_CONVENTIONS.md`、`LEARNING_LOG.md` 作为个人工作主入口。
- 如果后续需要做更干净的作品集发布包，应作为单独 release / packaging 步骤处理，而不是在开发仓库中静默删除上游上下文。

一句话：默认入口展示 EdgeSeg 的个人工作，上游来源保留且可追溯。

## 快速入口

- Phase 1 计划与进度：[`phase1/README.md`](./phase1/README.md)
- Phase 1 瓶颈分析报告：[`phase1/bottleneck_analysis_report.md`](./phase1/bottleneck_analysis_report.md)
- baseline 脚本设计：[`phase1/design_notes/baseline_inference_design.md`](./phase1/design_notes/baseline_inference_design.md)
- Nsight attribution 汇总：[`phase1/results/metrics/`](./phase1/results/metrics/)
- 上游 EfficientViT README：[`UPSTREAM_README.md`](./UPSTREAM_README.md)

## 上游归因

本项目基于 MIT Han Lab 的 EfficientViT 代码库。上游项目说明、论文链接、模型介绍和原始使用方式见 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。本仓库新增工作的重点是边缘推理 profiling、GPU kernel attribution 方法论，以及面向 TensorRT Plugin 的优化候选设计。
