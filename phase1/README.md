# Phase 1 · Baseline & Profiling

> **阶段目标**：在 PyTorch 原生推理路径上建立 EfficientViT-Seg-B0 的完整性能基线，并用 NVIDIA Nsight Systems 完成系统级剖析。**关键产出是《性能瓶颈与融合机会分析报告》，为阶段三的 TensorRT 自定义算子（C++/CUDA Plugin）开发选定融合目标**。
>
> 📌 **跟进 Floatboat.md V3.0**：项目核心定位已升级为 **TensorRT 自定义算子开发**。阶段一的剖析目标因此从"找瓶颈"进一步精细化为"找融合机会"。
>
> ⚠️ **2026-05-26 架构精读后修正**：原 V3.0 文档预设的"`MatMul+Softmax+Scale`""`LayerNorm+残差`"两类融合目标 **在 EfficientViT 上不成立**——本模型采用 **LiteMLA 线性注意力**（无 softmax）+ **BN2d**（无 LayerNorm）。真实的融合候选请见 [`architecture_analysis.md`](./architecture_analysis.md) §4。
>
> 📚 **文档分层导航**：
> - 项目级战略：[`../../Floatboat.md`](../../../Floatboat.md)（仓库外，需在文件系统打开）
> - 横切协作契约：[`../../PROJECT_CONVENTIONS.md`](../../../PROJECT_CONVENTIONS.md)（仓库外，AI 协作流程见 §1）
> - 阶段执行（本文件）：`phase1/README.md`
> - 代码级设计：脚本 docstring 或 `phase1/design_notes/xxx_design.md`（按需创建）

---

## 📁 目录结构

```
phase1/
├── README.md                              ← 本文件，阶段一导航
├── architecture_analysis.md               ← 【NEW】EfficientViT-Seg-B0 架构精读（决策依据）
├── scripts/                               ← 可执行脚本（baseline_inference.py 等）
├── weights/                               ← 预训练权重（.pt/.pth，不入库）
├── data/                                  ← Cityscapes 样图（不入库）
├── results/
│   ├── metrics/                           ← 延迟/显存/吞吐 csv（入库，体积小）
│   └── nsight/                            ← .nsys-rep 报告 + 截图（不入库）
└── bottleneck_analysis_report.md          ← 最终交付物（V3.0 重命名，待编写）
```

> ⚠️ `weights/`、`data/`、`results/nsight/` 已在根 `.gitignore` 中排除，每个目录用 `.gitkeep` 占位保持结构。

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
  - **NVTX 标注**：双跑策略，`--nvtx-level {A,B,C}` ✅（详见"决策 3"小节）
  - **三档 smoke 全部通过**（MX250, 512×1024, random weights, warmup 3 + measure 5）：
    - Plan A: mean ≈ 23.9 ms
    - Plan B: mean ≈ 23.6 ms（mid-grain hooks 已注入，hook_count=12）
    - Plan C: hotspot component-level hooks 已注入；当前范围为 `stage0/stage2/head`（约 12 个 range / 24 hooks），正式 nsys 需按新版定义重跑
  - 配套设计文档：[`design_notes/baseline_inference_design.md`](./design_notes/baseline_inference_design.md)（575 行）
  - 使用速查：[`scripts/README.md`](./scripts/README.md)
  - ✅ **正式 Plan A baseline 已完成**：真实 Cityscapes 权重 + 固定输入图 + 1024×2048 + warmup 20 + measure 100，结果见 [`results/metrics/baseline_b0_cityscapes_1024x2048_levelA_latency_formal_v1.json`](./results/metrics/baseline_b0_cityscapes_1024x2048_levelA_latency_formal_v1.json)
- [ ] **Step 5**：用 Nsight Systems 剖析推理过程
  - 命令模板（Windows Nsight Systems 2026.2.1）：`nsys profile -t cuda,nvtx -o results/nsight/baseline --stats=true python scripts/baseline_inference.py`
  - 注：Windows 版 `nsys` 不接受 `osrt` trace；`wddm` 需要管理员权限，普通终端会被禁用。Phase 1 归因主口径使用 `cuda,nvtx`。
  - 截 3 类关键图：CPU↔GPU 时间线、**CUDA kernel 耗时归因排序（重点）**、显存使用曲线
  - 分析口径：端到端 latency 以 JSON 中 CUDA Events 为准；NVTX range 只提供结构边界，组件占比应从 Nsight sqlite 中用 CUDA runtime/kernel `correlationId` 归因统计，不能直接用 NVTX range 的 `end-start` 当 GPU 耗时
- [ ] **Step 6**：撰写 `bottleneck_analysis_report.md`
  - 不只是"哪里慢"，更要标注 **"哪些算子序列适合融合为 Plugin"**
  - 给出每个候选融合点的实测耗时 + 预期加速理论估算

---

## 🛠️ 环境快照（2026-05-26）

| 组件 | 版本 |
|---|---|
| GPU | NVIDIA GeForce MX250 (Pascal, sm_61, 2GB) |
| Driver / CUDA Toolkit | 560.81 / 12.6 |
| Conda env | `efficientvit` @ `D:\software\anaconda3\envs\efficientvit` |
| Python | 3.10.20 |
| PyTorch | 2.4.1+cu124 (最后一批官方支持 sm_61 的版本) |
| cuDNN | 9.1.0 |
| Nsight Systems | 2026.2.1 @ `D:\software\nsight_systems\target-windows-x64` |

---

## 📌 关键决策记录

### 决策 1：项目核心定位（V3.0 战略对齐）
- **从**：QAT 量化研究
- **到**：**TensorRT 自定义算子（C++/CUDA Plugin）开发**
- **影响**：阶段一从"通用剖析"细化为"为 Plugin 找融合目标"。

### 决策 2：PyTorch 版本选择
- **选择**：PyTorch 2.4.1+cu124
- **原因**：PyTorch 2.7+ 已放弃 Pascal 架构（sm_61）预编译 wheel，2.4.x 是最后一批官方支持 MX250 的版本。

### 决策 3：NVTX 标注粒度 ✅ **已确定（commit `ec4cda2` 实装）**
> **采纳双跑策略**：正式 baseline 用 Plan B（mid-grain，stem/stage0..3/head 共 6 个 range），Plugin 设计用 Plan C（hotspot component-level，展开 `stage0/stage2/head` 的关键组件）。
> 实装方式：`baseline_inference.py --nvtx-level {A,B,C}`；Plan A 无 NVTX 用于干净 latency 参考。
> 详细讨论见下方"NVTX 标注方案"小节。

### 决策 4：模型变种 / 输入分辨率
- **变种**：先选 EfficientViT-Seg-B0（MX250 仅 2GB，B1+ 极易 OOM）
- **分辨率**：先用 Cityscapes 原生 1024×2048。如 OOM 或单次 >2s，降到 512×1024。
- **batch size**：固定 1（边缘实时推理 + 显存约束）。

### 决策 5：阶段一不测精度
- 阶段一聚焦"剖析与融合机会发现"。
- 精度对齐推迟到阶段二（PyTorch ↔ TRT 对齐）和阶段三（融合 Plugin ↔ 原始算子对齐）。

---

## 🔍 NVTX 标注方案（决策 3 候选）

> **共同原则**：NVTX range 只做 `torch.cuda.nvtx.range_push/range_pop`（或等价的 `nvtx.annotate` 上下文管理器），用于 Nsight 归因；**range 内禁止插入 `torch.cuda.synchronize()`**。同步只允许出现在 warmup/measure 边界，以及 latency 模式下 CUDA Event 读取处。NVTX 不是计时工具，latency 以 CUDA Events 为准。

### 方案 A · 最粗（3 个 range）
仅区分 `total / backbone / head`。
- ✅ 实现 5 行代码，最快出图；
- ❌ 看不出注意力 vs 卷积谁主导，**无法支撑 Plugin 选型**。
- 适用：阶段一第一次跑通的 sanity check。

### 方案 B · 中等粒度（≈10 个 range）✅ **默认推荐**
```
total
├── backbone
│   ├── input_stem
│   ├── stage1
│   ├── stage2
│   ├── stage3                       ← 含 2 个 EfficientViTBlock
│   │   └── attn_block_0/1           （仅在含注意力的 stage）
│   │        ├── lite_mla
│   │        └── mbconv_ffn
│   └── stage4                       ← 含 2 个 EfficientViTBlock（结构同上）
└── head
    ├── inputs_merge                 （1×1 Conv + 2× upsample + add）
    ├── middle                       （1× MBConv）
    └── output_proj                  （final_expand + cls head）
```
- ✅ 能直接回答"注意力 vs 卷积谁主导""head 的 upsample 是不是瓶颈"；
- ✅ 输出报告里每个候选融合点都有对应数据；
- ⚠️ 需要给 backbone / EfficientViTBlock / SegHead 注册 forward hooks。

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
- ✅ 使用 forward hooks，不改写 forward 数值路径，因此不需要 sanity check；
- ⚠️ `head` 内部的 merge add 不是独立 module，当前 hook-only 方案不单独计入一个 range；
- ℹ️ 若后续需要 LiteMLA 内部 `qkv / aggregation / attention matmul / proj` 子算子级耗时，应另写专门的 `litemla_internal_profile.py`；
- ⚠️ NVTX range 本身有 ~1μs/次开销，B0 stage4 注意力本来就只有几百 μs，**range 太密可能反而扰动测量**。

### 🎯 我的明确建议
> **跑两遍**：第一遍用 **方案 B** 得到稳定的端到端 baseline 数字（提交报告主表）；第二遍用 **方案 C** 展开 `stage0/stage2/head` 的热点组件，专门给阶段三 Plugin 设计提供组件级耗时。
>
> 这样既不让 NVTX 开销污染主基线，又能拿到 Plugin 设计需要的细粒度数据，**是性价比最高的策略**。

**[已实装 ✅]** 双跑策略已写进 `baseline_inference.py` 的 `--nvtx-level {A,B,C}` 参数（commit `ec4cda2`）。三档 smoke test 全部通过，详见上方 Step 4。

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
