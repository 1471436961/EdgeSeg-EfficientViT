
---

# EdgeSeg-EfficientViT 机器人边缘推理优化项目 — 最终协作实施文档（V3.1）

**项目代号**: EdgeSeg-EfficientViT  
**协作对象**: Codex
**目标岗位**: AI推理优化工程师 / 机器人AI部署优化工程师 / 机器人/边缘AI系统工程师 
**核心定位**: 以顶会模型 EfficientViT（ICCV 2023）为支点，在消费级 GPU 上完成全链路推理优化。**重点通过 TensorRT 自定义算子（Plugin）开发，展示 C++/CUDA 底层优化硬实力**，辅以系统瓶颈分析和边缘部署方案设计，构建一个高竞争力、高区分度的技术作品。

> 📌 **V3.1 与 V3.0 的关系**：本版本是在 2026-05-26 完成 EfficientViT-Seg-B0 源码精读后，对 V3.0 进行的**纠错性修订**——V3.0 里所有基于"传统 Transformer 假设"提出的融合目标（如 `MatMul+Softmax+Scale`、`LayerNorm+残差`）已被证实**与本模型实际结构不符**，本版本予以更正，完整修订纪要见文末 §8。

> 📎 **AI 协作流程**：本项目所有 AI 协作行为遵循 [`PROJECT_CONVENTIONS.md`](./PROJECT_CONVENTIONS.md) 定义的契约——尤其是 §1「先解释决策 → 用户确认 → 再落盘」的三段式流程。该契约文件与本战略文档**并列**于项目根，是横切关注点，不属于本文档的子章节。

---

## 1. 项目概述

本项目基于 MIT Han Lab 提出的 **EfficientViT（ICCV 2023）**，其 **LiteMLA（轻量多尺度线性注意力）** 在多种硬件平台上均取得显著加速效果，但该模型对量化极为敏感（论文报告：标准 INT8 量化可导致精度损失高达 76.15%）。

项目的核心差异化在于：**不满足于使用 TensorRT 的默认优化，而是深入模型计算图，为 LiteMLA 这种 TensorRT 无法自动高效融合的非常规注意力模块，手写 C++/CUDA 自定义算子（Plugin），从根本上展示底层系统优化能力。**

> **Phase 1 实测修正**：Nsight SQLite attribution 显示，B0 在 1024×2048 输入下的最大 GPU kernel 热点是 `stage0` 早期高分辨率卷积区域，其次是 `stage2`；LiteMLA/context 是重要候选但不是全模型最大瓶颈。因此 Phase 3 的叙事应是：**LiteMLA Plugin 是高区分度的自定义算子主线**，而不是"最大耗时瓶颈"；`stage0` 与 `SegHead/head` 是端到端收益更强的工程优化候选，但其中大量热点属于 MBConv / Conv-BN-Act 系列，可能已被 TensorRT/cuDNN 较好处理。

**为什么 LiteMLA 是绝佳的 Plugin 目标？**
- 它是一个 **非标准注意力结构**：用 `ReLU(Q), ReLU(K)` 替代 softmax，用 `(V·K^T)·Q` 顺序乘法把复杂度从 O(N²·d) 降到 O(N·d²)，并通过"末尾补 1 行 + 除法"实现归一化——这一整套组合 TensorRT 没有现成融合规则，是真正的 Plugin 价值区。
- 它在论文中明确被强调为模型加速的核心创新，**对标论文模块写 Plugin** 比"我做了 MatMul+Softmax 融合"具体得多，简历卖点强。
- 内部包含 reshape / transpose / pad / 两次 MatMul / 逐元素除法等多种算子，是展示 CUDA kernel 设计能力（shared memory、epilogue fusion、layout 优化）的极好载体。
- 但它的定位是**能力展示与非标准算子融合**，不是当前 B0/MX250 profile 下的最大端到端加速点；报告中必须把这两件事分开说。

**为什么自定义算子是核心亮点？**
- 证明你不仅能"使用工具"，还能"改进工具"
- 直接命中 AI Infra/推理优化岗位对 C++/CUDA 的核心要求
- 在秋招候选人中建立极强的区分度

---

## 2. 技术栈与开发环境

- **基础框架**: PyTorch 2.4.1+cu124（基线推理；Pascal sm_61 最后官方支持的版本）
- **优化工具链**: ONNX、TensorRT（**重点使用 C++ API 开发自定义算子**，Python API 用于辅助验证）
- **性能分析**: **NVIDIA Nsight Systems 2026.2.1**（主要）、`torch.profiler`（辅助）
- **编程语言**: **C++/CUDA（核心产出）**、Python（流程控制与验证）
- **硬件环境**:
  - GPU: NVIDIA GeForce MX250（2GB 显存，Pascal 架构，算力 sm_61）
  - CPU: Intel Core i5-10th Gen
  - RAM: 12GB
  - CUDA: 12.6，Driver: 560.81
- **模型与数据**: EfficientViT-Seg-B0（Cityscapes 预训练权重）

---

## 3. 项目起点

- **官方仓库**: [https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
- **初始操作**: Fork 该仓库，clone 到本地，创建独立 conda 环境。按阶段创建分支（`phase1-baseline`、`phase2-tensorrt`、`phase3-plugin`）。

> **重要提醒**: 运行任何 GPU 程序前，请确保关闭其他占用显存的进程，使用 `nvidia-smi` 确认可用显存接近 2048MB。

---

## 4. 项目阶段与详细任务

### 阶段一：基线建立与性能剖析（6月上旬，已完成）

**目标**: 建立 PyTorch 原生推理基线，通过 Nsight Systems 深度剖析定位瓶颈，**为后续自定义算子开发选定融合目标**。

**任务**:
1. **环境搭建**: 创建 conda 环境，安装 PyTorch（CUDA 12.6 对应版本）、torchvision、tensorrt、pycuda、Nsight Systems 等。
2. **EfficientViT-Seg-B0 源码精读**: 在写任何剖析代码之前，先完整读懂模型结构、确认真实的算子序列，避免基于错误假设写出错误的 NVTX 标注或选错 Plugin 目标。**产出 `phase1/architecture_analysis.md`**（已完成，详见该文件 §4 候选融合目标）。
3. **跑通原生推理**: 使用官方脚本加载 EfficientViT-Seg-B0 预训练权重，以 batch size=1 进行推理，记录延迟、显存占用、吞吐量。
4. **深度性能剖析**（重点）:
   - 使用 **NVIDIA Nsight Systems** 进行系统级分析。
   - **阶段命名口径**：下文 `stage0/stage1/stage2/stage3` 默认指 Phase 1 NVTX / 代码中的 `backbone.stages` 索引；它们分别对应架构语义 `stage1/stage2/stage3/stage4`。因此 `stage2/context` 指代码/NVTX `stage2` 的 LiteMLA context，也就是架构语义 stage3。
   - **重点关注**（依据 `architecture_analysis.md` + Plan B/C/D Nsight attribution，已修正 V3.0 的错误预设）：
     - **stage0 早期高分辨率卷积/MBConv 区域**：当前最大 GPU kernel 热点，是端到端收益最高的性能候选。
     - **stage2 LiteMLA/context + local MBConv**：LiteMLA/context 是高区分度 Plugin 主线，且在 stage2 内 context 比 local 更耗时；需客观记录其不是全模型最大瓶颈。
    - **Seg Head 多输入融合**：`Conv1x1 + bicubic Upsample + add`，`head/middle` 已显示为重要组件；Phase 2 已验证当前固定 shape / TensorRT 8.6.1 下 bicubic Resize 可 parser/build/runtime，但该结论不外推到动态 shape 或其他 TensorRT 版本。
   - 采用 **四档 NVTX 策略**（详见 `phase1/README.md` 决策 2）：
     - 方案 A 无 NVTX，出干净端到端 latency baseline；
     - 第一遍方案 B（中粒度，stem/stage0..3/head）出稳定的全模型大区域归因；
     - 第二遍方案 C 展开 `stage0/stage2/head` 的热点组件，给阶段三 Plugin 与工程优化候选排序做依据。
     - 第三遍方案 D 细拆 `stage2/context` LiteMLA 内部 `qkv / aggregation / cat / relu_linear_att / proj`，把 Phase 3 候选从"整体 LiteMLA"细化为"局部单段 / 中段组合 / 整体 fallback"三类 Plugin 边界。
   - 生成 **"性能瓶颈与融合机会分析报告"**，同时给出"端到端收益优先级"与"Plugin 展示价值优先级"，避免把二者混为一谈。

**产出物**:
- `phase1/architecture_analysis.md`（架构精读）
- `phase1/scripts/baseline_inference.py`（含 `--nvtx-level {A,B,C,D}` 切换）
- 性能基线 JSON + Nsight attribution Markdown/JSON 汇总表
- **`phase1/bottleneck_analysis_report.md`**（含 Nsight Systems 截图、算子耗时排序、端到端收益排序与 Plugin 展示价值排序）

> ⚠️ **V3.0 删除项**：原 V3.0 中"重点关注 MatMul、Softmax、Scale、LayerNorm"已被证实与 EfficientViT 实际结构不符（**该模型 0 个 Softmax、0 个 LayerNorm**），已在本版本完全替换。

---

### 阶段二：TensorRT 基础部署（截至 2026-06-10，已完成）

**目标**: 完成 ONNX 导出与 TensorRT 推理，产出基础优化数据，并用 Nsight Systems 复核 TensorRT 优化后的残余热点，判断阶段一确定的 Plugin 候选在 TensorRT 后是否仍然成立。

**任务**:
1. **模型导出**: 将 PyTorch 模型导出为 ONNX，确保动态轴和输出节点正确。
   - ⚠️ **架构精读后新增风险点**：LiteMLA 内部有 `H*W > dim ? 线性 : 二次` 的形状自适应分支（`ops.py:662`），ONNX 导出时必须**冻结输入分辨率**，否则两条分支都会被 trace 进去导致图错乱。
   - ✅ **Seg Head Upsample 默认是 bicubic**，当前固定 shape / TensorRT 8.6.1 下已验证 parser/build/runtime 通过；不需要在 Phase 2 改 bilinear。该结论不外推到动态 shape 或其他 TensorRT 版本。
2. **TensorRT 引擎构建**: 使用 TensorRT Python API 构建 FP32 baseline engine，并将 FP16 作为风险实验单独记录。当前 MX250 / TensorRT 8.6.1 实测表明：FP16 可构建且语义一致，但慢于 FP32，因此本机主 baseline 采用 FP32。
   - ⚠️ **数值策略边界**：LiteMLA `relu_linear_att` 标了 `@torch.autocast(enabled=False)` 强制 FP32。Phase 2 已证明 TensorRT FP16 风险实验语义可接受但无速度收益；Phase 3 Plugin 仍需单独设计 FP32 / FP16 / FP32 accumulate 的内部数值策略。
3. **推理验证**: 加载 TensorRT 引擎进行推理，验证输出精度（对齐 PyTorch baseline），测量延迟和吞吐量。
4. **TensorRT Nsight 复核**: 已对 TensorRT engine runtime 采集 Nsight Systems trace，第一版 residual hotspot 排序为 `stage0 > stage2 > stage3 > stage1 > head > stem`。已补 EngineInspector / ONNX node name 映射作为结构辅助证据：ONNX `393` nodes -> TensorRT `155` engine layers；但它不能替代 Nsight runtime 归因。
5. **TensorRT C++ 推理 Demo**（轻量）: 编写一个简单的 C++ 推理 Demo，加载 TensorRT 引擎并执行推理。此 Demo 不必追求极致优化，主要目的是熟悉 TensorRT C++ API，为阶段三的 Plugin 集成验证铺路。

**产出物**:
- `phase2/export_onnx.py`、`phase2/build_trt_engine.py`
- 优化效果对比表（PyTorch vs TensorRT FP32/FP16；本机主 TensorRT baseline 为 FP32）
- TensorRT Nsight attribution 汇总，回答 Phase 1 候选在 TensorRT 后是否仍成立
- EngineInspector / ONNX node name 映射汇总，回答 TensorRT 在结构层面做了哪些 layer 压缩和 fusion pattern
- C++ 推理 Demo 源码及 CMake 编译脚本
- 过程中遇到的问题记录（特别是固定 shape bicubic Resize 验证边界 / LiteMLA 数值精度处理的取舍）

---

### 阶段三：TensorRT 自定义算子开发（6月中旬–7月下旬，下一主线）

> **核心定位**: 这是整个项目最具区分度的部分。基于阶段一的剖析报告，选择一个既有数据支撑、又能展示 C++/CUDA/TensorRT Plugin 能力的融合目标。LiteMLA 仍是默认主线，因为它是非标准线性注意力、TensorRT 难以自动高效融合；但报告必须明确：在当前 B0/MX250 profile 下，LiteMLA 不是全模型最大瓶颈，stage0/head 也是重要优化候选。Plan D 用于把 `stage2/context` LiteMLA 从"整体模块候选"细化为"具体可融合子路径候选"。

**任务**:

**1. 自定义算子设计与实现（核心任务，60-70%精力）**

- **目标算子选择（按"求职展示价值"与"端到端收益"双维度排序）**：
  - 🥇 **P1：stage2 LiteMLA Plugin 主线**（默认主交付物） —— LiteMLA 是论文核心创新，也是 TensorRT 难以自动融合的非标准线性注意力结构；它不一定是当前 PyTorch profile 的最大端到端瓶颈，但最能展示非标准算子分析、CUDA kernel 设计和 TensorRT Plugin 集成能力。
    - **P1a：局部单段 Plugin（MVP 优先）** —— Phase 3 Step 2 已把第一版 MVP 收敛为 `relu_linear_att-only`：真实 contract 为 `[1,384,64,128] -> [1,128,64,128]`，不需要 Plugin 权重，便于先验证 TensorRT Plugin 接入、数值对齐、engine 替换与 Nsight attribution。`aggregation-only` 保留为 fallback / 对照实验。Phase 2 中的 `attention_core = relu_qk + pad + matmul + norm_add_div` 只是 TensorRT layer-name 视角下对 `relu_linear_att` 内部残余路径的 proxy，不反向改写 Phase 1 的 MVP 定义。
    - **P1b：中段组合 Plugin（收益评估主方向）** —— Phase 1 主性能边界仍是 `aggregation + cat + relu_linear_att`，因为 Plan D 显示 `aggregation` 与 `relu_linear_att` 是两大主耗时，且二者之间存在 `cat` 中间拼接。Phase 3 Step 2 确认该边界真实 contract 为 `[1,192,64,128] -> [1,128,64,128]`。Phase 2 TensorRT 后可用 `aggregation + attention_core`（约 `5.443 ms / iter`、`38` launches / iter）作为对应的 residual-runtime 复核 proxy。
    - **P1c：整体 LiteMLA Plugin（fallback / 上限方案）** —— 若单段或中段组合方案收益不足、TensorRT 图集成边界不合适，或希望最大化融合空间，再考虑整体 LiteMLA 级 Plugin。它不是优先 MVP，而是复杂度更高的兜底/上限方案。
  - 🥈 **P2：标准算子链工程优化候选（stage0 / head / stage2-local）** —— 这些区域在 PyTorch Nsight 结果中占比高，端到端收益潜力更直接；Phase 2 TensorRT Nsight 仍显示 `stage0` 是最大 residual hotspot、`stage2` 第二，但这类区域主要由 MBConv / Conv / BN / activation / upsample / add 等标准算子链构成，是否值得手写 Plugin 仍需按展示价值和 TensorRT 已优化程度谨慎排序。
    - **P2a：stage0 early MBConv / Conv 堆叠** —— 当前 PyTorch 与 TensorRT 两条路径下都很重，主要受高分辨率 feature map 和 memory traffic 影响；端到端收益潜力高，但自定义 Plugin 展示区分度低于 LiteMLA。
    - **P2b：Seg Head / head middle** —— `head/middle` 是 MBConv，单项耗时明显但大概率属于标准优化区域；`Conv1x1 + bicubic Upsample + add` 在当前固定 shape / TensorRT 8.6.1 下已能通过，后续只在 TensorRT Nsight 显示 residual hotspot 时再作为工程优化候选。
    - **P2c：stage2-local MBConv** —— 与 `stage2/context` 同属 stage2 热点区，但结构上更接近标准 MBConv，展示价值低于 LiteMLA。
  - 🥉 **P3：低优先级探索项** —— 多尺度 QKV 聚合段（`Conv1x1 + DWConv5×5 + grouped Conv1x1 + concat`）、head resize 替代、INT8/混合精度策略等。B0 scales 只有 1 个，短期收益可能有限；B1/B2 或更大模型上价值更高。
  - 最终选择 P1a / P1b / P1c 中哪种 LiteMLA Plugin 边界，以及是否追加 P2 工程优化，**由阶段一 Plan D + 阶段二 TensorRT Nsight attribution + Phase 2 baseline report** 共同决定，不能只凭"论文模块"、"PyTorch hotspot"或"TensorRT 端到端 speedup"单边判断。
- **融合算子设计**: 撰写算子融合设计文档，包含：
  - 原始计算图与融合后计算图的对比
  - 内存布局优化策略（reshape/transpose 消除、shared memory 使用、epilogue fusion）
  - 预期加速比的理论计算
  - FP16 数值稳定性方案（针对 LiteMLA 的 `eps + 除法` 链路）
- **C++/CUDA 实现**:
  - 编写融合算子的 CUDA Kernel
  - 按照 TensorRT Plugin 规范，实现 `IPluginV2DynamicExt` 接口（包括 `enqueue`、`configurePlugin`、`getOutputDimensions` 等方法）
  - 编写对应的 `PluginCreator`，使 TensorRT 能够识别和加载自定义算子
- **集成与验证**:
  - 将自定义算子集成到 ONNX 模型或直接通过 TensorRT Network API 构建网络
  - 使用阶段二的 C++ 推理 Demo 加载含 Plugin 的引擎，验证正确性和加速效果
  - 对比集成前后的延迟、吞吐量，以及与 PyTorch baseline 的精度差异
  - 对最终采用的 Plugin engine 增加 Cityscapes mIoU / semantic regression accuracy gate：正式 mIoU 使用上游一致的 ImageNet mean/std 预处理；此前 Phase 2/3 的 `[0,1]` 输入口径只用于 latency 和 deployment-style 回归检查，不能替代数据集级精度结论。

**2. 量化探索（辅助任务，20-30%精力）**

- 尝试标准 INT8 PTQ，验证论文结论（精度严重损失）
- 基于敏感性分析，设计混合精度量化（MPQ）方案——LiteMLA 注意力段保留 FP16 或 FP32，纯卷积段应用 INT8
- 撰写量化探索报告，记录成功与失败的数据

**3. 面向 Jetson Orin 的部署迁移方案（轻量，10%精力）**

- 撰写一份简明的迁移方案文档，论证将当前优化成果迁移到 Jetson Orin 平台的路径
- 包含：预期性能估算、潜在问题（功耗、散热、sm_87 与 sm_61 的指令集差异）与解决思路

**4. ROS 2 节点封装 Demo（可选加分项）**

- 将优化后的推理引擎封装成一个最小可行的 ROS 2 节点
- 在 Docker 环境中运行，接收图像消息并发布分割结果
- 成本极低，但能显著提升项目的"机器人系统感"

**产出物**:
- **`phase3/plugin_fusion_design.md`**（算子融合设计文档）
- **`phase3/lite_mla_plugin.cu/.h`**（自定义算子源码，文件名按主选目标定，默认 `lite_mla_plugin`）
- **`phase3/integration_validation_report.md`**（集成验证报告，含性能对比、精度对齐数据）
- `phase3/design_notes/cityscapes_miou_evaluation_design.md` 与 `phase3/scripts/evaluate_cityscapes_miou.py`（Cityscapes mIoU 验收口径与执行脚本；数据集本体因授权和体积原因不入库）
- `phase3/quantization_exploration.md`（量化探索报告）
- `phase3/jetson_migration_plan.md`（Jetson 迁移方案）
- （可选）ROS 2 节点源码

---

## 5. 时间规划与秋招并行策略

> **更新时间：2026-06-10。** Phase 1 与 Phase 2 已提前完成，后续计划应围绕 Phase 3 Plugin 主线压缩推进；时间表以“完成状态 + 下一阶段优先级”为准，而不是继续沿用最初的粗略月份预估。

| 阶段 | 时间 / 状态 | 核心产出 | 秋招投递动作 |
| :--- | :--- | :--- | :--- |
| **阶段一** | 6月上旬，已完成 | 架构精读 + PyTorch baseline + Nsight attribution + 瓶颈分析报告（含双维度候选排序） | 可作为项目第一版经历写入简历草稿 |
| **阶段二** | 6月上旬至 6月10 日，已完成 | ONNX 导出 + TensorRT FP32/FP16 baseline + TensorRT Nsight 复核 + EngineInspector + C++ Runtime Demo + Phase 2 报告 | 更新 GitHub README 与简历项目描述，准备讲清 PyTorch→TensorRT 全链路 |
| **阶段三** | 6月中旬–7月下旬，下一主线 | **LiteMLA TensorRT Plugin MVP（核心）** + Plugin 集成验证 + Nsight 对比 + 必要消融 | 6月中下旬开始用 Phase 1/2 成果投递；7月随着 Plugin MVP 进展更新简历亮点 |
| **扩展与收尾** | 7月下旬–8月中旬 | 量化探索、Jetson 迁移方案、可选 ROS 2 Demo、技术博客与最终 README polish | 面向重点岗位集中投递，并根据面试反馈补强文档和实验 |

Phase 3 的优先级顺序：

1. 先完成 LiteMLA Plugin MVP 的设计与最小可运行实现，优先验证 `relu_linear_att-only`；`aggregation-only` 只作为 fallback / 对照实验。
2. 再评估 `aggregation + cat + relu_linear_att` 中段组合边界的收益与实现成本。
3. 只有在单段 / 中段边界收益不足或 graph 集成不合适时，再考虑整体 LiteMLA Plugin。
4. `stage0/head` 等标准 MBConv/Conv 热点作为工程优化候选保留，但不抢占 Phase 3 第一主线。

---

## 6. 协作方式

- **Codex 需要能够**：
  - 读取、修改项目代码文件（包括 C++/CUDA 源码）
  - 在终端执行命令（Python 脚本、CUDA 编译、TensorRT 引擎构建等）
  - 捕获并记录命令输出与性能数据
  - 根据指导创建 Markdown 文档、图表和报告

- **我会负责**：
  - 做出所有技术决策与方向确认
  - 审核并优化 Codex 产出的代码与文档
  - 与你讨论方案细节，并最终验收每个阶段的成果

---

## 7. 特别说明

- 本项目受限于 MX250 2GB 显存，所有任务设计已将硬件限制转化为展示系统分析能力的优势。
- **自定义算子开发是本项目的核心亮点**，务必投入足够精力做深做透。一个完整、有数据支撑的自定义算子（比如 LiteMLA Plugin），远胜过多个浅尝辄止的优化尝试。但报告中要区分"高区分度 Plugin 目标"与"最大端到端瓶颈"，不要把 LiteMLA 叙述成当前 profile 的最大热点。
- TensorRT FP16 已实测可构建且语义一致，但在 MX250 上慢于 FP32；本机主 baseline 采用 FP32。后续 FP16/混合精度只作为 Phase 3 Plugin 数值策略问题继续讨论。
- 每份报告和源码都是面试中的核心论据，请务必详尽、专业、体现思考深度。

---

## 8. 修订纪要（V3.0 → V3.1，2026-05-26）

| # | V3.0 原内容 | V3.1 修订为 | 修订原因 |
|---|------------|------------|----------|
| 1 | §1 "多尺度线性注意力在多种硬件平台上均取得显著加速效果" | 同上，但补充"用 ReLU 替代 softmax / `(V·K^T)·Q` 顺序乘法"的具体机制说明 | 让读者一眼看出"为什么这个模块适合写 Plugin" |
| 2 | §1 缺少"为什么 LiteMLA 是 Plugin 目标"的论证 | 新增独立段落 **"为什么 LiteMLA 是绝佳的 Plugin 目标"** | 项目讲故事的钩子前置，简历和面试用得上 |
| 3 | §2 PyTorch 未指定版本；Nsight 未指定版本 | 明确为 **PyTorch 2.4.1+cu124** + **Nsight Systems 2026.2.1** | 反映 5/26 环境验证的实际结果，便于复现 |
| 4 | §4 阶段一"重点关注：MatMul、Softmax、Scale、LayerNorm" | 替换为 **LiteMLA 线性注意力核 / Seg Head 多输入融合 / stage0-1 MBConv 堆叠** 三类 | **核心纠错**：EfficientViT 0 个 Softmax、0 个 LayerNorm |
| 5 | §4 阶段一未提 NVTX 策略 | 新增 **四档 NVTX 策略**（A 干净 latency、B 全模型 stage 级归因、C 热点组件级归因、D LiteMLA 内部子路径归因） | 避免 NVTX 开销污染主基线测量，同时给 Plugin 候选提供多层证据，详见 `phase1/README.md` 决策 2 |
| 6 | §4 阶段一产出物未含架构分析 | 新增 `phase1/architecture_analysis.md`| 反映已完成工作 |
| 7 | §4 阶段二未提 ONNX 导出风险 | 新增 **形状自适应分支** 与 **bicubic Upsample** 两个风险点 | 两个点不提前知道，阶段二大概率会卡壳 |
| 8 | §4 阶段三 1.目标算子选择 "MatMul+Softmax+Scale / LayerNorm+残差" | 替换为 **双维度候选融合目标排序**（P1 stage2 LiteMLA 主线：局部单段 MVP / `aggregation+cat+relu_linear_att` 中段组合 / 整体 fallback；P2 stage0/head/stage2-local 标准算子链工程候选；P3 低优先探索项） | 同 #4，纠错；同时吸收 Phase 1 Nsight attribution 的真实结论 |
| 9 | §4 阶段三产出物 `fused_attention_plugin.cu/.h` | 重命名为 `lite_mla_plugin.cu/.h` | 更精确反映模块名，便于面试讲解 |
| 10 | §4 阶段三 2.量化探索 "敏感层 FP16/INT8 混合" | 明确为 **"LiteMLA 注意力段 FP16/FP32，纯卷积 INT8"** | 基于架构精读，量化敏感区段已明确 |
| 11 | §5 阶段一核心产出 | 加上 "架构精读 + 双维度候选排序" | 同步阶段一任务变化 |
| 12 | 文末缺少修订纪要 | 新增 **§8 修订纪要** | 保证文档可追溯，方便后续 V3.2+ |

**后续更新**：时间规划已在 2026-06-10 根据 Phase 1/2 实际完成情况更新，详见 §5；项目定位、技术栈大方向、协作方式、自定义算子作为核心亮点的战略保持不变。
