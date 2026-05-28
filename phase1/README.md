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
- [ ] **Step 2**：下载 EfficientViT-Seg-B0 预训练权重（Cityscapes 版）
- [ ] **Step 3**：准备 1~2 张 Cityscapes 样图放入 `data/`
- [x] **Step 4**：编写 `scripts/baseline_inference.py` ✅ **commit `ec4cda2`**
  - CUDA Event 精确计时（**不能用 time.time()**）✅
  - 预热 20 次 + 正式 100 次（默认值；smoke 用 3+5）✅
  - 记录 avg/p50/p95/p99 延迟、峰值显存、FPS ✅
  - **NVTX 标注**：双跑策略，`--nvtx-level {A,B,C}` ✅（详见"决策 3"小节）
  - **三档 smoke 全部通过**（MX250, 512×1024, random weights, warmup 3 + measure 5）：
    - Plan A: mean ≈ 23.9 ms
    - Plan B: mean ≈ 23.6 ms（mid-grain hooks 已注入，hook_count=14）
    - Plan C: mean ≈ 23.5 ms（4 个 LiteMLA monkey-patch + sanity_check 全过，max_abs_diff=0.0）
  - 配套设计文档：[`design_notes/baseline_inference_design.md`](./design_notes/baseline_inference_design.md)（575 行）
  - 使用速查：[`scripts/README.md`](./scripts/README.md)
  - ⚠️ **以上仅为脚本链路验证**。**正式 baseline 仍需**：真实 Cityscapes 权重（Step 2）+ 固定输入图（Step 3）+ 1024×2048 + warmup 20 + measure 100，并将 JSON 落到 `results/metrics/`。在 Step 2/3 完成前，禁止把 smoke 的 24 ms / 42 FPS 写入任何性能结论。
- [ ] **Step 5**：用 Nsight Systems 剖析推理过程
  - 命令模板：`nsys profile -t cuda,nvtx,osrt -o results/nsight/baseline --stats=true python scripts/baseline_inference.py`
  - 截 3 类关键图：CPU↔GPU 时间线、**算子序列耗时排序（重点）**、显存使用曲线
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
> **采纳双跑策略**：正式 baseline 用 Plan B（mid-grain，stem/stage0..4/head 共 ~7 个 range），Plugin 设计用 Plan C（LiteMLA-internal，4 个实例级 monkey-patch + sanity check）。
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

> **共同原则**：所有方案都用 `torch.cuda.nvtx.range_push/pop`（或 `nvtx.annotate` 上下文管理器）插入；每次都包住 `torch.cuda.synchronize()` 之后才结束 range，否则 timeline 会错位。

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
- ⚠️ 需要给 backbone / EfficientViTBlock / SegHead 各打 ~3 个 hook 或直接 monkey-patch forward。

### 方案 C · 最细（≈25 个 range，含 LiteMLA 内部）
在方案 B 基础上，**进一步细化 LiteMLA 内部**：
```
lite_mla
├── qkv_conv                    （Conv1x1 → 3*total_dim）
├── multi_scale_aggreg          （Conv5x5 DW + Conv1x1 grouped + concat）
├── reshape_split_qkv
├── relu_qk
├── linear_matmul_vkt           （V·K^T）
├── linear_matmul_vkq           （VK·Q）
├── norm_divide                 （末尾归一化除法）
└── proj_conv                   （Conv1x1 + BN）
```
- ✅ **直接指出 Plugin 内部每个子操作的耗时占比**，对阶段三 kernel 设计极有价值；
- ⚠️ 需要短暂"侵入式"修改 `LiteMLA.forward`（写一个带 NVTX 的 patched 版本，仅用于剖析）；
- ⚠️ NVTX range 本身有 ~1μs/次开销，B0 stage4 注意力本来就只有几百 μs，**range 太密可能反而扰动测量**。

### 🎯 我的明确建议
> **跑两遍**：第一遍用 **方案 B** 得到稳定的端到端 baseline 数字（提交报告主表）；第二遍用 **方案 C 仅针对 LiteMLA** 单独打开（其他 range 关掉），专门给阶段三 Plugin 设计提供子算子级耗时。
>
> 这样既不让 NVTX 开销污染主基线，又能拿到 Plugin 设计需要的细粒度数据，**是性价比最高的策略**。

**[已实装 ✅]** 双跑策略已写进 `baseline_inference.py` 的 `--nvtx-level {A,B,C}` 参数（commit `ec4cda2`）。三档 smoke test 全部通过，详见上方 Step 4。

---

## 🧭 阶段间依赖关系（V3.0）

```
Phase 1 (剖析报告 + 架构精读)
    │
    │ 输出：候选融合算子序列 + 各段耗时占比 + 加速比理论估算
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
