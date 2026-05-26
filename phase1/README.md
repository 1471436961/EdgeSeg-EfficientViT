# Phase 1 · Baseline & Profiling

> **阶段目标**：在 PyTorch 原生推理路径上建立 EfficientViT-Seg 的完整性能基线，并用 NVIDIA Nsight Systems 完成系统级剖析。**关键产出是《性能瓶颈与算子融合机会分析报告》，为阶段三的 TensorRT 自定义算子（C++/CUDA Plugin）开发选定融合目标**。
>
> 📌 **跟进 Floatboat.md V3.0**：项目核心定位已升级为 **TensorRT 自定义算子开发**。阶段一的剖析目标因此从"找瓶颈"进一步精细化为"找融合机会"，重点关注 MatMul+Softmax+Scale、LayerNorm+残差 等算子序列。

---

## 📁 目录结构

```
phase1/
├── README.md                              ← 本文件，阶段一导航
├── scripts/                               ← 可执行脚本（baseline_inference.py 等）
├── weights/                               ← 预训练权重（.pt/.pth，不入库）
├── data/                                  ← Cityscapes 样图（不入库）
├── results/
│   ├── metrics/                           ← 延迟/显存/吞吐 csv（入库，体积小）
│   └── nsight/                            ← .nsys-rep 报告 + 截图（不入库）
└── bottleneck_analysis_report.md          ← 最终交付物（V3.0 重命名，待编写）
```

> ⚠️ `weights/`、`data/`、`results/nsight/` 已在根 `.gitignore` 中排除，每个目录用 `.gitkeep` 占位保持结构。所有 `.pt/.onnx/.engine/.nsys-rep` 等大文件默认不入库，请放心存放本地大文件。

---

## 🎯 阶段一任务清单（V3.0 对齐版）

- [x] **Step 0**：搭建 `phase1-baseline` 分支与目录骨架
- [x] **Step 1**：环境验证（PyTorch 2.4.1+cu124 ✅ / Nsight Systems 2026.2.1 ✅）
- [ ] **Step 2**：下载 EfficientViT-Seg-B0 预训练权重
- [ ] **Step 3**：准备 1~2 张 Cityscapes 样图放入 `data/`
- [ ] **Step 4**：编写 `scripts/baseline_inference.py`
  - CUDA Event 精确计时（**不能用 time.time()**）
  - 预热 20 次 + 正式 100 次
  - 记录 avg/p50/p95/p99 延迟、峰值显存、FPS
  - **细粒度 NVTX 标注**（这是 V3.0 的关键升级）：
    - backbone / head 级别
    - 关键算子序列：`MatMul + Softmax + Scale`（多尺度线性注意力内）
    - `LayerNorm + 残差` 序列
    - 这些标注直接对应阶段三 Plugin 的候选融合目标
- [ ] **Step 5**：用 Nsight Systems 剖析推理过程
  - 命令模板：`nsys profile -t cuda,nvtx,osrt -o results/nsight/baseline --stats=true python scripts/baseline_inference.py`
  - 截 3 类关键图：CPU↔GPU 时间线、**算子序列耗时排序（重点）**、显存使用曲线
- [ ] **Step 6**：撰写 `bottleneck_analysis_report.md`
  - 不只是"哪里慢"，更要标注 **"哪些算子序列适合融合为 Plugin"**
  - 给出每个候选融合点的预期加速理论估算

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

- **【V3.0 战略对齐】项目核心定位**：从 QAT 量化研究 → **TensorRT 自定义算子（C++/CUDA Plugin）开发**。阶段一的所有产出必须服务于这个新目标。
- **为什么选 PyTorch 2.4.1+cu124**：PyTorch 2.7+ 已放弃 Pascal 架构（sm_61）预编译 wheel，2.4.x 是最后一批官方支持 MX250 的版本。
- **为什么先选 EfficientViT-Seg-B0**：MX250 仅 2GB 显存，更大的 B1/B2/B3 极易 OOM。先跑通 B0，余量充足再尝试 B1。
- **为什么 batch size = 1**：项目目标是**边缘设备实时推理**，bs=1 是真实部署场景；同时贴合 2GB 显存限制。
- **为什么阶段一不测精度**：阶段一聚焦"剖析与融合机会发现"。精度对齐推迟到阶段二（PyTorch ↔ TRT 对齐）和阶段三（融合 Plugin ↔ 原始算子对齐）。

---

## 🧭 阶段间依赖关系（V3.0）

```
Phase 1 (剖析报告)
    │
    │ 输出：候选融合算子序列 + 加速比理论估算
    ▼
Phase 2 (TRT 基础部署 + C++ Demo)
    │
    │ 输出：可加载 Plugin 的 C++ 推理框架
    ▼
Phase 3 (Plugin 开发) ← 项目核心亮点
    │
    │ 输出：fused_attention_plugin.cu/.h + 加速验证报告
    ▼
秋招简历核心论据
```
