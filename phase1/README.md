# Phase 1 · Baseline & Profiling

> **阶段目标**：在 PyTorch 原生推理路径上建立 EfficientViT-Seg-B0 的完整性能基线，并用 NVIDIA Nsight Systems 完成系统级剖析。**关键产出是《性能瓶颈与融合机会分析报告》，为阶段三的 TensorRT 自定义算子（C++/CUDA Plugin）开发选定融合目标**。
>
> 📌 **跟进 PROJECT_STRATEGY.md V3.1**：项目核心定位已升级为 **TensorRT 自定义算子开发**。阶段一的剖析目标因此从"找瓶颈"进一步精细化为"找融合机会"。
>
> ⚠️ **2026-05-26 架构精读后修正**：原 V3.0 文档预设的"`MatMul+Softmax+Scale`""`LayerNorm+残差`"两类融合目标 **在 EfficientViT 上不成立**——本模型采用 **LiteMLA 线性注意力**（无 softmax）+ **BN2d**（无 LayerNorm）。真实的融合候选请见 [`architecture_analysis.md`](./architecture_analysis.md) §4。
>
> 📚 **文档分层导航**：
> - 项目级战略：[`../../PROJECT_STRATEGY.md`](../../PROJECT_STRATEGY.md)（仓库外，需在文件系统打开）
> - 横切协作契约：[`../../PROJECT_CONVENTIONS.md`](../../PROJECT_CONVENTIONS.md)（仓库外，AI 协作流程见 §1）
> - 阶段执行（本文件）：`phase1/README.md`
> - 代码级设计：脚本 docstring 或 `phase1/design_notes/xxx_design.md`（按需创建）
> - 阶段一设计纠偏：[`design_notes/phase1_decision_corrections.md`](./design_notes/phase1_decision_corrections.md)

---

## 📁 目录结构

```
phase1/
├── README.md                              ← 本文件，阶段一导航
├── architecture_analysis.md               ← 【NEW】EfficientViT-Seg-B0 架构精读（决策依据）
├── scripts/                               ← 可执行脚本（baseline_inference.py 等）
├── design_notes/
│   ├── baseline_inference_design.md        ← baseline 脚本设计记录
│   └── phase1_decision_corrections.md      ← Phase 1 设计纠偏 / 口径审计记录
├── weights/                               ← 预训练权重（.pt/.pth，不入库）
├── data/                                  ← 固定样图（白名单入库）/ 大型数据集（不入库）
├── results/
│   ├── metrics/                           ← 延迟/显存/归因 JSON/MD（入库，体积小）
│   ├── figures/                           ← Nsight 关键截图（入库，体积小）
│   └── nsight/                            ← .nsys-rep / .sqlite 原始报告（不入库）
└── bottleneck_analysis_report.md          ← 最终交付物（性能瓶颈与融合机会分析）
```

> ⚠️ `weights/`、大型数据集与 `results/nsight/` 已在根 `.gitignore` 中排除；`phase1/data/city_asset_cityscapes_like.png` 作为固定 profiling 样图白名单入库。

---

## 🎯 阶段一任务清单（V3.0 对齐 + 架构精读修订版）

- [x] **Step 0**：搭建 `phase1-baseline` 分支与目录骨架
- [x] **Step 1**：环境验证（PyTorch 2.4.1+cu124 ✅ / Nsight Systems 2026.2.1 ✅）
- [x] **Step 1.5**：**EfficientViT-Seg-B0 源码精读**，产出 `architecture_analysis.md`
- [x] **Step 2**：下载 EfficientViT-Seg-B0 预训练权重（Cityscapes 版）✅
  - 文件：`phase1/weights/efficientvit_seg_b0_cityscapes.pt`（不入库）
  - SHA256：`923d6fdd5e93640cc0c2f3f213764f34e80b477cd98a6b294d870ea6df5acc50`
- [x] **Step 3**：准备固定输入样图放入 `data/` ✅
  - 文件：`phase1/data/city_asset_cityscapes_like.png`
  - 来源：上游仓库自带 `assets/fig/city.png`，用于 Phase 1 latency/profiling，不用于 mIoU 评估
  - SHA256：`34a663391ddeed9bbcc98c605d881fadbf7bb05ff02a8ffe4136d52599efc630`
- [x] **Step 4**：编写 `scripts/baseline_inference.py` ✅ **commit `ec4cda2`**
  - CUDA Event 精确计时（**不能用 time.time()**）✅
  - 预热 20 次 + 正式 100 次（默认值；smoke 用 3+5）✅
  - 记录 avg/p50/p95/p99 延迟、峰值显存、FPS ✅
  - **NVTX 标注**：四档口径，`--nvtx-level {A,B,C,D}` ✅（详见"决策 2"小节）
  - **三档 smoke 全部通过**（MX250, 512×1024, random weights, warmup 3 + measure 5）✅
  - 配套设计文档：[`design_notes/baseline_inference_design.md`](./design_notes/baseline_inference_design.md)✅
  - 使用速查：[`scripts/README.md`](./scripts/README.md)✅
  - ✅ **正式 Plan A baseline 已完成**：真实 Cityscapes 权重 + 固定输入图 + 1024×2048 + warmup 20 + measure 100，结果见 [`results/metrics/baseline_b0_cityscapes_1024x2048_levelA_latency_formal_v1.json`](./results/metrics/baseline_b0_cityscapes_1024x2048_levelA_latency_formal_v1.json)
- [x] **Step 5**：用 Nsight Systems 剖析推理过程
  - 命令模板（Windows Nsight Systems 2026.2.1）：`nsys profile -t cuda,nvtx -o results/nsight/baseline --stats=true python scripts/baseline_inference.py`
  - 注：Windows 版 `nsys` 不接受 `osrt` trace；`wddm` 需要管理员权限，普通终端会被禁用。Phase 1 归因主口径使用 `cuda,nvtx`。
  - ✅ Nsight 关键截图已归档到 [`results/figures/`](./results/figures/)：
    - Plan B：[`planB_timeline_overview.png`](./results/figures/planB_timeline_overview.png)、[`planB_single_forward_nvtx.png`](./results/figures/planB_single_forward_nvtx.png)
    - Plan C：[`planC_timeline_overview.png`](./results/figures/planC_timeline_overview.png)、[`planC_stage0_components.png`](./results/figures/planC_stage0_components.png)、[`planC_stage2_components.png`](./results/figures/planC_stage2_components.png)、[`planC_head_components.png`](./results/figures/planC_head_components.png)
    - Plan D：[`planD_timeline_overview.png`](./results/figures/planD_timeline_overview.png)、[`planD_litemla_aggregation_components.png`](./results/figures/planD_litemla_aggregation_components.png)、[`planD_litemla_relu_linear_att_components.png`](./results/figures/planD_litemla_relu_linear_att_components.png)
  - 截图口径：`Threads -> NVTX` 用于确认逻辑阶段/组件边界；`CUDA HW -> Kernels` 用于观察对应 GPU kernel 执行；`CUDA HW -> NVTX` 仅作 GPU 侧投影趋势参考
  - 分析口径：端到端 latency 以 JSON 中 CUDA Events 为准；NVTX range 只提供结构边界，组件占比应从 Nsight sqlite 中用 CUDA runtime/kernel `correlationId` 归因统计，不能直接用 NVTX range 的 `end-start` 当 GPU 耗时
  - ✅ Plan B/C/D Nsight attribution 表已生成：[`planB_nsys_attribution_summary.md`](./results/metrics/planB_nsys_attribution_summary.md)、[`planC_nsys_attribution_summary.md`](./results/metrics/planC_nsys_attribution_summary.md)、[`planD_nsys_attribution_summary.md`](./results/metrics/planD_nsys_attribution_summary.md)
  - ✅ 显存证据当前采用 JSON 中的 `max_memory_allocated_mb` / `max_memory_reserved_mb` 峰值字段；连续显存曲线不是本轮 Nsight 截图主证据
- [x] **Step 6**：撰写 [`bottleneck_analysis_report.md`](./bottleneck_analysis_report.md)
  - 不只是"哪里慢"，更要标注 **"哪些算子序列适合融合为 Plugin"**
  - 给出每个候选融合点的实测耗时 + 预期加速理论估算
  - 报告必须区分两条排序：**端到端耗时收益**（当前 `stage0` 最大）与 **Plugin 展示价值**（LiteMLA 仍是高区分度主线）

---

## 📌 关键决策记录

### 决策 1：项目核心定位（V3.0 战略对齐）
- **从**：QAT 量化研究
- **到**：**TensorRT 自定义算子（C++/CUDA Plugin）开发**
- **影响**：阶段一从"通用剖析"细化为"为 Plugin 找融合目标"。

### 决策 2：NVTX 标注粒度 ✅ **已确定并扩展到 Plan D**
> **采纳四档口径**：Plan A 无 NVTX，用于干净 latency baseline；Plan B 为 stage-level attribution（`stem/stage0..3/head` 共 6 个 range）；Plan C 为 hotspot component-level attribution（展开 `stage0/stage2/head` 的关键组件）；Plan D 为 stage2 LiteMLA internal attribution（`qkv/aggregation/cat/relu_linear_att/proj`），用于把 Phase 3 Plugin 候选从"整体 LiteMLA"细化为"局部单段 / 中段组合 / 整体 fallback"三类 Plugin 边界。
> 实装方式：`baseline_inference.py --nvtx-level {A,B,C,D}`。
> 详细讨论见下方"NVTX 标注方案"小节。

### 决策 3：模型变种 / 输入分辨率
- **变种**：先选 EfficientViT-Seg-B0（MX250 仅 2GB，B1+ 极易 OOM）
- **分辨率**：先用 Cityscapes 原生 1024×2048。如 OOM 或单次 >2s，降到 512×1024。
- **batch size**：固定 1（边缘实时推理 + 显存约束）。

### 决策 4：阶段一不测精度
- 阶段一聚焦"剖析与融合机会发现"。
- 固定输入图只用于 latency/profiling，不用于 mIoU 或视觉质量结论。
- 精度对齐推迟到阶段二（PyTorch ↔ TRT 对齐）和阶段三（融合 Plugin ↔ 原始算子对齐）；若未来需要 Cityscapes mIoU，应另起 `evaluate.py` 并单独确认设计。

---

## 🔍 NVTX 标注方案（决策 2 候选）

> **共同原则**：NVTX range 只做 `torch.cuda.nvtx.range_push/range_pop`（或等价的 `nvtx.annotate` 上下文管理器），用于 Nsight 归因；**range 内禁止插入 `torch.cuda.synchronize()`**。同步只允许出现在 warmup/measure 边界，以及 latency 模式下 CUDA Event 读取处。NVTX 不是计时工具，latency 以 CUDA Events 为准。

### 方案 A · 干净 latency（无 NVTX）
不注册任何 NVTX range，只用 CUDA Events 测端到端 latency。
- ✅ 最少 profiler/annotation 扰动，作为正式 latency 主表口径；
- ✅ 适合和后续 TensorRT / Plugin 端到端 latency 做公平对比；
- ❌ 不提供模块归因，不能回答"哪里慢"。

### 方案 B · Stage 级归因（6 个 range）✅ **全模型归因主口径**
```
stem
stage0
stage1
stage2
stage3
head
```
- ✅ 覆盖全模型主要结构，用于回答 stage/head 级热点排序；
- ✅ 当前 Plan B sqlite attribution 表明 `stage0` 最大、`stage2` 第二，`head/stage3/stem` 接近；
- ⚠️ 只提供大区域归因，不展开 `stage0/stage2/head` 内部组件。

### 方案 C · 热点组件级（stage0/stage2/head）
在方案 B 基础上，**只展开最值得解释的热点区域**：
```
stage0
├── block0/main           （早期高分辨率 MBConv/ResidualBlock）
└── block1/main

stage2
├── downsample
├── block1/context_module   （LiteMLA + residual）
├── block1/local_module     （MBConv + residual）
├── block2/context_module
└── block2/local_module

head
├── input_stage4
├── input_stage3
├── input_stage2
├── middle
└── output_segout
```
- ✅ **直接回答热点区域里到底是 LiteMLA、MBConv 还是 SegHead 更慢**；
- ✅ `stage0 + stage2 + head` 覆盖 Plan B 约 60% 的 GPU kernel 耗时，同时避免把 Plan C 膨胀成全模型递归 profiler；
- ✅ 支撑 Phase 3 的双维度判断：`stage0` 端到端收益最高，`stage2/LiteMLA` 自定义算子展示价值最高，`head/middle` 是工程优化候选；
- ✅ 使用 forward hooks，不改写 forward 数值路径，因此不需要 sanity check；
- ⚠️ `head` 内部的 merge add 不是独立 module，当前 hook-only 方案不单独计入一个 range；
- ⚠️ NVTX range 本身有 ~1μs/次开销，B0 stage4 注意力本来就只有几百 μs，**range 太密可能反而扰动测量**。

### 方案 D · stage2 LiteMLA 内部子路径级（Plugin 候选细化）
只针对 Plan C 已确认的 `stage2/context` LiteMLA，拆第一层子路径，不进入 `relu_linear_att()` 函数体：
```
stage2/block1/litemla/qkv
stage2/block1/litemla/aggregation
stage2/block1/litemla/cat
stage2/block1/litemla/relu_linear_att
stage2/block1/litemla/proj

stage2/block2/litemla/qkv
...
```
- ✅ 直接回答 LiteMLA 内部到底是 qkv、multi-scale aggregation、linear attention 还是 proj 更值得融合；
- ✅ `relu_linear_att()` 保持黑盒调用，因此其原始 `@torch.autocast(device_type="cuda", enabled=False)` 与 dtype 逻辑不被破坏；
- ✅ 使用实例级 `LiteMLA.forward` patch，并在 profiling 前做 patched-vs-original `torch.allclose(atol=1e-5, rtol=1e-5)` sanity check；
- ⚠️ Plan D 是 Phase 3 Plugin 候选细化实验，不替代 Plan B 的全模型归因，也不替代 Plan C 的热点组件归因；
- ℹ️ Plan D 结果显示 `aggregation` 与 `relu_linear_att` 是 stage2 LiteMLA 内部两大主耗时，且二者之间存在 `cat` 中间拼接；Phase 3 应比较三类 Plugin 边界：局部单段（`aggregation-only` / `relu_linear_att-only`）、中段组合（`aggregation + cat + relu_linear_att`）、整体 LiteMLA fallback。

### 🎯 我的明确建议
> **分四档使用**：用 **方案 A** 得到干净端到端 latency baseline（提交报告主表）；用 **方案 B** 得到全模型 stage/head 级归因；用 **方案 C** 展开 `stage0/stage2/head` 的热点组件；用 **方案 D** 细拆 stage2/context 中的 LiteMLA 内部子路径，专门为阶段三 Plugin 选定具体可融合目标。
>
> 这样既不让 NVTX 开销污染主 latency 基线，又能拿到 Plugin 设计需要的全模型、热点组件、stage2 LiteMLA 内部三层证据，**是性价比最高的策略**。

> **Phase 3 叙事约束**：Plan B/C 结果不能支撑"LiteMLA 是全模型最大瓶颈"。更严谨的说法是：`stage0/block*/main`、`head/middle`、`stage2/local` 主要是 MBConv/Conv 系列，耗时高很大程度来自分辨率与 feature map 尺寸，TensorRT/cuDNN 可能已有较好处理；`stage2/context` 中的 LiteMLA 不是最大端到端瓶颈，但更符合自定义 Plugin 的高区分度主线。Plan D 进一步说明：`relu_linear_att-only` 适合作为 MVP，`aggregation + cat + relu_linear_att` 是更有潜在收益的主评估方向，整体 LiteMLA 是复杂度更高的 fallback / 上限方案。

**[已实装 ✅]** 四档口径已写进 `baseline_inference.py` 的 `--nvtx-level {A,B,C,D}` 参数。A/B/C/D 均已有正式结果与 attribution 表。

---

## 🧭 阶段间依赖关系（V3.0）

```
Phase 1 (剖析报告 + 架构精读)
    │
    │ 输出：候选融合算子序列 + 各段 CUDA kernel 归因耗时占比 + 加速比理论估算
    ▼
Phase 2 (TRT 基础部署 + C++ Demo)
    │
    │ 输出：可加载 Plugin 的 C++ 推理框架
    ▼
Phase 3 (Plugin 开发) ← 项目核心亮点
    │
    │ 输出：lite_mla_plugin.cu/.h + 加速验证报告
    ▼
秋招简历核心论据
```
