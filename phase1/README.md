# Phase 1 · Baseline & Profiling

> **阶段目标**：在 PyTorch 原生推理路径上建立 EfficientViT-Seg 的完整性能基线，并用 NVIDIA Nsight Systems 完成系统级剖析，产出《性能瓶颈分析报告》作为阶段二/三优化方向的事实依据。

---

## 📁 目录结构

```
phase1/
├── README.md                       ← 本文件，阶段一导航
├── scripts/                        ← 可执行脚本（baseline_inference.py 等）
├── weights/                        ← 预训练权重（.pt/.pth，不入库）
├── data/                           ← Cityscapes 样图（不入库）
├── results/
│   ├── metrics/                    ← 延迟/显存/吞吐 csv（入库，体积小）
│   └── nsight/                     ← .nsys-rep 报告 + 截图（不入库）
└── profiling_analysis_report.md    ← 最终交付物（待编写）
```

> ⚠️ `weights/`、`data/`、`results/nsight/` 已在根 `.gitignore` 中排除，每个目录用 `.gitkeep` 占位保持结构。**所有 `.pt/.onnx/.engine/.nsys-rep` 等大文件默认不入库**，请放心存放本地大文件。

---

## 🎯 阶段一任务清单

- [ ] **Step 1**：环境验证（PyTorch CUDA ✅ / Nsight Systems 安装）
- [ ] **Step 2**：下载 EfficientViT-Seg 预训练权重（先选 `b0` 适配 2GB 显存）
- [ ] **Step 3**：准备 1~2 张 Cityscapes 样图放入 `data/`
- [ ] **Step 4**：编写 `scripts/baseline_inference.py`
  - CUDA Event 精确计时（**不能用 time.time()**）
  - 预热 20 次 + 正式 100 次
  - 记录 avg/p50/p95/p99 延迟、峰值显存、FPS
  - 添加 NVTX range 标注（供 Nsight 识别推理阶段）
- [ ] **Step 5**：用 Nsight Systems 剖析推理过程
  - 命令模板：`nsys profile -t cuda,nvtx,osrt -o results/nsight/baseline --stats=true python scripts/baseline_inference.py`
  - 截 3 张关键图：CPU↔GPU 时间线、Top-10 耗时 Kernel、显存使用曲线
- [ ] **Step 6**：撰写 `profiling_analysis_report.md`

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
| Nsight Systems | 2026.2.1（待装） |

---

## 📌 关键决策记录

- **为什么选 PyTorch 2.4.1+cu124**：PyTorch 2.7+ 已放弃 Pascal 架构（sm_61）预编译 wheel，2.4.x 是最后一批官方支持 MX250 的版本。
- **为什么先选 EfficientViT-Seg-b0**：MX250 仅 2GB 显存，更大的 b1/b2/b3 极易 OOM。先跑通 b0，余量充足再尝试 b1。
- **为什么 batch size = 1**：项目目标是**边缘设备实时推理**，bs=1 是真实部署场景；同时贴合 2GB 显存限制。
