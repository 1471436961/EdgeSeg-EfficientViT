# Phase 2 — ONNX / TensorRT 基础部署

> **阶段目标**：在 Phase 1 PyTorch baseline 与 Nsight attribution 的基础上，建立 `PyTorch -> ONNX -> TensorRT` 的基础部署链路，产出可和 Phase 1 对比的 TensorRT baseline，并为 Phase 3 LiteMLA Plugin 选择提供新的证据。
>
> **当前状态**：ONNX 固定 shape 导出与 ONNXRuntime 对齐验证已完成；下一步进入 TensorRT engine 构建与 benchmark。

---

## 1. 阶段边界

Phase 2 做：

- 导出 EfficientViT-Seg-B0 为 ONNX。
- 验证 ONNX 输出与 PyTorch 输出数值一致性。
- 构建 TensorRT FP32 / FP16 baseline engine。
- 运行 TensorRT inference benchmark，与 Phase 1 PyTorch baseline 对比。
- 重新观察 Phase 1 中的 P1/P2 候选在 TensorRT 后是否仍然成立。

Phase 2 不做：

- 不实现自定义 TensorRT Plugin。
- 不把 LiteMLA 内部算子改写为 CUDA kernel。
- 不以 Cityscapes 全量 mIoU 作为第一验收条件。
- 不追求动态输入分辨率；第一版固定 `1x3x1024x2048`。

---

## 2. 输入与输出约定

| 项目 | 约定 |
|---|---|
| 模型 | EfficientViT-Seg-B0 Cityscapes pretrained |
| 权重 | `phase1/weights/efficientvit_seg_b0_cityscapes.pt`（不入 git） |
| 输入图 | `phase1/data/city_asset_cityscapes_like.png` |
| 输入分辨率 | 固定 `1024x2048` |
| Batch size | 1 |
| 输出 | `segout`，shape 预期为 `(1, 19, 128, 256)` |
| 主 dtype | FP32 baseline 优先，FP16 为第二轮对比 |

固定分辨率的原因：LiteMLA 内部存在 `H*W > dim` 的形状分支。Phase 2 第一版先冻结 Cityscapes 原生分辨率，避免动态 shape 让 ONNX/TensorRT 图语义变复杂。

---

## 3. 任务清单

- [x] Step 0：创建 `phase2-tensorrt` 分支，并从已合并 Phase 1 的 `master` 开始。
- [x] Step 1：建立 Phase 2 目录骨架与 ONNX 导出设计文档。
- [x] Step 2：实现 `phase2/scripts/export_onnx.py`。
  - 加载 Phase 1 权重与固定输入。
  - 导出 ONNX。
  - 写入导出元信息 JSON。
- [x] Step 3：ONNX 基础验证。
  - `onnx.checker` 结构检查。
  - ONNXRuntime 推理输出与 PyTorch 输出对齐。
  - 记录 `max_abs_diff`、`mean_abs_diff`、`cosine_similarity`。
- [x] Step 4：实现 `phase2/scripts/build_trt_engine.py`。
  - 第一版先构建 FP32 engine。
  - FP16 作为 FP32 构建和 benchmark 成功后的第二轮实验。
  - 记录 parser / builder 日志、engine 大小、构建配置。
- [ ] Step 5：实现 TensorRT benchmark。
  - 复用 Phase 1 CUDA Events 计时口径。
  - 记录 latency、显存、输出误差。
- [ ] Step 6：撰写 `phase2/tensorrt_baseline_report.md`。
  - PyTorch vs ONNXRuntime vs TensorRT 对比。
  - TensorRT 后热点是否变化。
  - Phase 3 Plugin 候选是否需要调整。

---

## 4. 目录结构

```text
phase2/
├── README.md
├── scripts/
│   ├── _compat.py
│   ├── build_trt_engine.py
│   └── export_onnx.py
├── design_notes/
│   ├── build_trt_engine_design.md
│   └── onnx_export_design.md
├── results/
│   ├── onnx/
│   │   ├── .gitkeep
│   │   └── efficientvit_seg_b0_cityscapes_1024x2048.onnx  # 运行产物，不入 git
│   ├── engines/
│   │   ├── .gitkeep
│   │   └── efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine  # 运行产物，不入 git
│   └── metrics/
│       ├── .gitkeep
│       ├── onnx_export_b0_cityscapes_1024x2048.json
│       └── trt_build_b0_cityscapes_1024x2048_fp32.json
└── logs/
    └── .gitkeep
```

---

## 5. 验收标准

第一阶段 ONNX 导出验收：

- `export_onnx.py` 能从真实权重构建模型并导出 ONNX。
- ONNX 文件通过 `onnx.checker.check_model`。
- ONNXRuntime 输出 shape 与 PyTorch 输出 shape 一致。
- ONNXRuntime 与 PyTorch 输出误差在设计文档定义的阈值内。
- 导出 JSON 记录模型、权重 hash、输入 hash、torch / onnx / onnxruntime 版本、固定 shape、输出节点等复现信息。

当前 ONNX 导出口径：

- opset：`17`（PyTorch 2.4.1 的默认 ONNX opset）。
- exporter：legacy `torch.onnx.export`。
- compat：`phase2/scripts/_compat.py` 提供 import-only `triton_stub` 与 `wandb_stub`。

当前 ONNX 验证结果：

| 项目 | 结果 |
|---|---|
| ONNX 文件 | `phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx`（约 2.92 MB，不入 git） |
| Metadata | `phase2/results/metrics/onnx_export_b0_cityscapes_1024x2048.json` |
| `onnx.checker` | 通过 |
| ONNXRuntime provider | `CPUExecutionProvider` |
| 输出 shape | `[1, 19, 128, 256]` |
| `allclose_pass` | `true` (`atol=1e-4`, `rtol=1e-4`) |
| `max_abs_diff` | `3.44e-4` |
| `mean_abs_diff` | `1.81e-5` |
| `cosine_similarity` | `0.99999999999` |

说明：当前对齐比较的是 PyTorch CUDA 输出与 ONNXRuntime CPU 输出，不要求逐 bit 完全一致；以上误差属于可接受范围。`max_rel_diff` 对接近 0 的输出值敏感，不作为单独否决指标。

TensorRT baseline 验收：

- TensorRT parser 错误完整落日志，不吞错。
- FP32 engine 可构建并可推理。
- TensorRT 输出与 PyTorch / ONNXRuntime 输出误差可解释。
- TensorRT latency 使用与 Phase 1 可比的 CUDA Events 口径。
- 若 FP16 engine 构建或数值对齐失败，报告中明确记录原因，不强行包装成成功。

当前 TensorRT 构建环境口径：

| 项目 | 结果 |
|---|---|
| GPU | NVIDIA GeForce MX250 (`sm_61`) |
| 最新 pip TensorRT | `tensorrt-cu12==11.0.0.114` 可安装，但 `trt.Builder` 报 `Unsupported SM: 0x601`，不可用于本机 |
| 当前可用 TensorRT | NVIDIA archived TensorRT `8.6.1` Windows zip |
| TensorRT root | `E:\NVIDIA\TensorRT-8.6.1.6` |
| Python wheel | `tensorrt-8.6.1-cp310-none-win_amd64.whl` |
| 补充 runtime | `nvidia-cudnn-cu12==8.9.7.29`、`nvidia-cublas-cu12==12.9.2.10`、`nvidia-cuda-nvrtc-cu12==12.9.86` |
| Builder smoke | `trt.Builder(...)` 创建成功 |

说明：MX250 是 Pascal `sm_61`，需要旧版 TensorRT 路线。`build_trt_engine.py` 会在 import `tensorrt` 前显式注入 TensorRT / cuDNN / cuBLAS / NVRTC DLL 目录，避免依赖系统全局 PATH。

当前 FP32 engine 构建结果：

| 项目 | 结果 |
|---|---|
| 构建脚本 | `phase2/scripts/build_trt_engine.py` |
| 设计文档 | `phase2/design_notes/build_trt_engine_design.md` |
| ONNX 输入 | `phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx` |
| Engine 输出 | `phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine`（约 3.67 MB，不入 git） |
| Metadata | `phase2/results/metrics/trt_build_b0_cityscapes_1024x2048_fp32.json` |
| TensorRT | `8.6.1` |
| Precision | FP32 |
| Workspace | 1024 MiB |
| Network IO | input `[1, 3, 1024, 2048]` -> segout `[1, 19, 128, 256]` |
| TensorRT network layers | 331 |
| Parser errors | 0 |

构建日志中出现两个需要记录但不阻断的提示：ONNX 中存在 INT64 权重，TensorRT 会尝试 cast 到 INT32，且有超出 INT32 范围的权重被 clamp；另外 TensorRT 默认启用的 TF32 flag 因 MX250 不支持 TF32 被禁用。后续 benchmark / 输出对齐时需要确认这些转换是否影响 logits 误差。

---

## 6. 已知风险

| 风险 | 影响 | 初始策略 |
|---|---|---|
| Windows 上 `triton` import 问题 | import EfficientViT 失败 | 复用 Phase 1 延迟注入 stub 思路，必要时抽成 `_compat.py` |
| `wandb` 退出清理 PermissionError | 脚本退出噪声 | Phase 2 脚本启动时禁用/隔离 wandb |
| LiteMLA shape-adaptive branch | 动态 shape 导出不稳定 | 第一版固定 `1024x2048` |
| SegHead bicubic upsample | TensorRT parser/build 可能不支持 | 第一版保持上游结构并记录失败；后续再决定 bilinear 替代或 Plugin |
| LiteMLA autocast-disabled 语义 | FP16 输出误差可能变大 | 先建立 FP32 baseline，FP16 作为第二轮对比 |

---

## 7. 与 Phase 1 / Phase 3 的关系

Phase 1 给出 PyTorch 路径证据：

- `stage0` 是当前 PyTorch 最大 GPU kernel 热点。
- `stage2/context` LiteMLA 是最高区分度 Plugin 主线。
- Plan D 将 LiteMLA 候选细化为 `aggregation-only` / `relu_linear_att-only`、`aggregation + cat + relu_linear_att`、整体 LiteMLA fallback。

Phase 2 要验证：

- TensorRT 是否已经自动优化 P2 标准算子链热点。
- LiteMLA 在 TensorRT 后是否仍然是值得手写 Plugin 的残余热点。
- `bicubic upsample` 和 FP16 数值策略是否成为新的工程约束。

Phase 3 才开始真正实现 Plugin。
