# Phase 2 — ONNX / TensorRT 基础部署

> **阶段目标**：在 Phase 1 PyTorch baseline 与 Nsight attribution 的基础上，建立 `PyTorch -> ONNX -> TensorRT` 的基础部署链路，产出可和 Phase 1 对比的 TensorRT baseline，并为 Phase 3 LiteMLA Plugin 选择提供新的证据。
>
> **当前状态**：ONNX 固定 shape 导出、ONNXRuntime 对齐、TensorRT 8.6.1 FP32/FP16 engine 构建与 benchmark、TensorRT Nsight Systems runtime attribution 均已完成；下一步进入 TensorRT C++ runtime demo，然后撰写 Phase 2 baseline report。

---

## 1. 阶段边界

Phase 2 做：

- 导出 EfficientViT-Seg-B0 为 ONNX。
- 验证 ONNX 输出与 PyTorch 输出数值一致性。
- 构建 TensorRT FP32 / FP16 baseline engine。
- 运行 TensorRT inference benchmark，与 Phase 1 PyTorch baseline 对比。
- 重新观察 Phase 1 中的 P1/P2 候选在 TensorRT 后是否仍然成立。
- 使用 Nsight Systems 分析 TensorRT engine runtime，复核 TensorRT 后的残余热点与 Plugin 候选排序。
- 实现轻量 TensorRT C++ 推理 Demo，验证 FP32 engine 能被 C++ Runtime API 加载和执行，为 Phase 3 Plugin 集成铺路。

Phase 2 不做：

- 不实现自定义 TensorRT Plugin。
- 不把 LiteMLA 内部算子改写为 CUDA kernel。
- 不以 Cityscapes 全量 mIoU 作为阶段验收条件；Phase 2 只做 PyTorch / ONNXRuntime / TensorRT 转换一致性验证。
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
- [x] Step 5：实现 TensorRT benchmark。
  - 复用 Phase 1 CUDA Events 计时口径。
  - 记录 latency、显存、输出误差。
- [x] Step 5.5：TensorRT FP16 风险实验。
  - 构建 FP16 engine。
  - 复用 benchmark 脚本记录 latency 与输出误差。
  - 结论：FP16 可构建且语义一致，但在 MX250 上慢于 FP32，不作为本机主 baseline。
- [x] Step 6：TensorRT Nsight Systems profiling / attribution。
  - 对 `benchmark_trt_engine.py` 的 FP32 engine execute 路径采集 Nsight Systems trace。
  - 使用 `cuda,nvtx` 为主口径；Windows 下 CPU sampling / WDDM tracing 需管理员权限，若不采集则作为限制项记录。
  - 对比 Phase 1 Plan B/C/D：kernel 类型分布、launch 密度、TensorRT 后残余热点，以及 LiteMLA / stage0 / head 候选是否仍成立。
  - EngineInspector / verbose layer dump 只作为解释 engine 结构的辅助证据，不替代 Nsight runtime 归因。
  - 当前正式结果：沿用 `warmup=20 / measure=100`，`trt/execute` kernel avg `54.454 ms/iter`，layer attribution 覆盖 `100.00%` execute kernel time；残余热点排序为 `stage0 > stage2 > stage3 > stage1 > head > stem`。
  - 已补 EngineInspector / ONNX node name 映射：ONNX `393` nodes -> TensorRT `155` engine layers，总体 layer-count reduction `60.56%`。
- [ ] Step 7：TensorRT C++ 推理 Demo。
  - 使用 TensorRT C++ Runtime API 加载 FP32 engine。
  - 分配固定 shape input / output buffer，执行一次或多次 inference。
  - 输出 binding 信息、输出 checksum / 简单统计，不追求独立性能优化。
  - 记录 CMake / MSVC / TensorRT include/lib 路径，为 Phase 3 Plugin 集成做工程预热。
- [ ] Step 8：撰写 `phase2/tensorrt_baseline_report.md`。
  - PyTorch vs ONNXRuntime vs TensorRT 对比。
  - TensorRT 后热点是否变化。
  - Phase 3 Plugin 候选是否需要调整。

---

## 4. 目录结构

```text
phase2/
??? README.md
??? scripts/
?   ??? _compat.py
?   ??? analyze_trt_nsys_attribution.py
?   ??? benchmark_trt_engine.py
?   ??? build_trt_engine.py
?   ??? export_onnx.py
?   ??? inspect_trt_engine.py
??? design_notes/
?   ??? benchmark_trt_engine_design.md
?   ??? build_trt_engine_design.md
?   ??? onnx_export_design.md
?   ??? trt_nsys_attribution_design.md
??? results/
?   ??? onnx/
?   ?   ??? .gitkeep
?   ?   ??? efficientvit_seg_b0_cityscapes_1024x2048.onnx  # ??????? git
?   ??? engines/
?   ?   ??? .gitkeep
?   ?   ??? efficientvit_seg_b0_cityscapes_1024x2048_fp16.engine  # ??????? git
?   ?   ??? efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine  # ??????? git
?   ??? metrics/
?   ?   ??? .gitkeep
?   ?   ??? onnx_export_b0_cityscapes_1024x2048.json
?   ?   ??? trt_engine_inspection_summary.md
?   ?   ??? trt_engine_inspection_summary.json
?   ?   ??? trt_nsys_attribution_summary.md
?   ?   ??? trt_nsys_attribution_summary.json
?   ?   ??? trt_benchmark_b0_cityscapes_1024x2048_fp32_nsys.json
?   ?   ??? trt_benchmark_b0_cityscapes_1024x2048_fp16.json
?   ?   ??? trt_benchmark_b0_cityscapes_1024x2048_fp32.json
?   ?   ??? trt_build_b0_cityscapes_1024x2048_fp16.json
?   ?   ??? trt_build_b0_cityscapes_1024x2048_fp32.json
?   ??? nsight/
?       ??? TensorRT Nsight trace / sqlite ???????????? git?
??? cpp_demo/
?   ??? TensorRT C++ runtime demo?Step 7 ???
??? logs/
    ??? .gitkeep
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
- TensorRT 后候选复核必须补充 Nsight Systems runtime attribution，不能只用端到端 speedup 代替瓶颈分析。
- 若 FP16 engine 构建或数值对齐失败，报告中明确记录原因，不强行包装成成功。

精度 / 语义一致性口径：

- Phase 2 不做完整 Cityscapes mIoU，不把数据集精度评测作为阶段完成条件。
- Phase 2 验证的是部署转换一致性：logits diff、relaxed allclose、cosine similarity、argmax pixel agreement。
- 完整 mIoU 或更大样本集评估放到 Phase 3 Plugin 集成验证或最终验收阶段。

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

当前 TensorRT benchmark 口径：

- 脚本：`phase2/scripts/benchmark_trt_engine.py`
- 设计文档：`phase2/design_notes/benchmark_trt_engine_design.md`
- 计时工具：CUDA Events
- 计时范围：只测 `context.execute_async_v2(...)`，不包含 preprocess / H2D / D2H / PyTorch reference
- 对齐对象：PyTorch CUDA logits reference
- 输出 JSON：`phase2/results/metrics/trt_benchmark_b0_cityscapes_1024x2048_fp32.json`

当前 TensorRT FP32 benchmark 结果：

| 项目 | 结果 |
|---|---|
| Engine | `phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine` |
| Metadata | `phase2/results/metrics/trt_benchmark_b0_cityscapes_1024x2048_fp32.json` |
| warmup / measure | 20 / 100 |
| TensorRT FP32 p50 / p95 / p99 | `54.44 ms` / `55.43 ms` / `55.68 ms` |
| Phase 1 PyTorch Plan A formal p50 | `85.70 ms` |
| p50 speedup | `1.57x` |
| TensorRT max / mean abs diff vs PyTorch | `2.69e-4` / `2.54e-5` |
| strict `1e-4` allclose | `false` |
| relaxed `1e-3` allclose | `true` |
| argmax pixel agreement | `100%` (`0 / 32768` mismatch) |
| PyTorch CUDA allocator peak during benchmark process | `170.76 MB allocated`, `228.0 MB reserved`（不代表 TensorRT 内部显存峰值） |

说明：TensorRT build 阶段存在 INT64 -> INT32 cast / clamp 提示，因此 `1e-4` logits allclose 未通过需要保守记录；但误差量级较小，`1e-3` 通过，且当前样图的分割 argmax 完全一致。Phase 2 报告中应把它表述为“FP32 TensorRT runtime 数值接近且语义输出一致”，而不是“逐元素严格一致”。

FP16 风险实验口径：

- FP16 不是默认主线结论，而是风险实验。
- MX250 是 Pascal `sm_61`，没有 Tensor Core，FP16 不一定比 FP32 快。
- LiteMLA 存在 FP32 数值保护语义，FP16 需要重点检查 logits diff 和 argmax pixel agreement。
- 判断标准：若 FP16 latency 明显优于 FP32 且输出误差可解释，可纳入 Phase 2 baseline；否则记录为“不建议在本机 MX250 路线启用 FP16”。

当前 TensorRT FP16 风险实验结果：

| 项目 | 结果 |
|---|---|
| FP16 build | 成功，`BuilderFlag.FP16` 已启用 |
| Engine | `phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp16.engine`（约 3.55 MB，不入 git） |
| Build metadata | `phase2/results/metrics/trt_build_b0_cityscapes_1024x2048_fp16.json` |
| Benchmark metadata | `phase2/results/metrics/trt_benchmark_b0_cityscapes_1024x2048_fp16.json` |
| TensorRT FP16 p50 / p95 / p99 | `59.39 ms` / `65.34 ms` / `66.85 ms` |
| TensorRT FP32 p50 / p95 / p99 | `54.44 ms` / `55.43 ms` / `55.68 ms` |
| FP16 vs FP32 latency | FP16 更慢，p50 约 `0.92x` FP32 |
| FP16 max / mean abs diff vs PyTorch | `2.69e-4` / `2.54e-5` |
| FP16 relaxed `1e-3` allclose | `true` |
| FP16 argmax pixel agreement | `100%` (`0 / 32768` mismatch) |

结论：当前 MX250 / TensorRT 8.6.1 路线下，FP16 engine 可构建且语义输出一致，但没有速度收益，反而比 FP32 慢。因此 Phase 2 主 baseline 应采用 FP32 TensorRT；FP16 作为风险实验记录，不建议作为本机主线优化结论。

当前 TensorRT Nsight attribution 结果：

| 项目 | 结果 |
|---|---|
| Nsight trace | `phase2/results/nsight/trt_fp32_fullres.nsys-rep`（运行产物，不入 git） |
| SQLite export | `phase2/results/nsight/trt_fp32_fullres.sqlite`（运行产物，不入 git） |
| Nsys benchmark metadata | `phase2/results/metrics/trt_benchmark_b0_cityscapes_1024x2048_fp32_nsys.json` |
| Attribution summary | `phase2/results/metrics/trt_nsys_attribution_summary.md` / `.json` |
| warmup / measure | `20 / 100` |
| CUDA Events latency mean / p50 | `55.242 ms` / `55.237 ms` |
| `trt/execute` kernel avg | `54.454 ms / iter` |
| `trt/execute` launches | `185.0 / iter` |
| Layer attribution / execute kernel time | `100.00%` |
| Residual hotspot order | `stage0 > stage2 > stage3 > stage1 > head > stem` |

说明：Step 6 使用 TensorRT/NVTX layer range -> CUDA runtime launch -> CUDA kernel `correlationId` 的归因口径，不把 NVTX range end-start 当作 GPU component time。该结果说明 TensorRT 后 `stage0` 仍是最大 residual hotspot，`stage2` 仍是第二大 residual hotspot 且 launch 密度高；但 TensorRT layer range 与 Phase 1 PyTorch Plan B/C/D 的模块范围不是一一对应关系，Phase 2 report 中需要按“趋势复核”而非“逐模块复刻”来表述。

TensorRT 后 `stage2/context` 细粒度 runtime 归因：

| Component | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---:|---:|---:|
| `matmul` | `1.947` | `3.58%` | `4.0` |
| `aggregation` | `1.754` | `3.22%` | `26.0` |
| `pad` | `0.695` | `1.28%` | `2.0` |
| `relu_qk` | `0.685` | `1.26%` | `4.0` |
| `qkv` | `0.544` | `1.00%` | `2.0` |
| `proj_add` | `0.396` | `0.73%` | `2.0` |
| `norm_add_div` | `0.361` | `0.66%` | `2.0` |

候选边界聚合：

| Candidate boundary | Includes | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---|---:|---:|---:|
| `full_stage2_context` | `qkv + aggregation + relu_qk + pad + matmul + norm_add_div + proj_add` | `6.383` | `11.72%` | `42.0` |
| `aggregation_plus_attention_core` | `aggregation + relu_qk + pad + matmul + norm_add_div` | `5.443` | `10.00%` | `38.0` |
| `attention_core` | `relu_qk + pad + matmul + norm_add_div` | `3.689` | `6.77%` | `12.0` |
| `aggregation_only` | `aggregation` | `1.754` | `3.22%` | `26.0` |

说明：这张表来自 `trt_nsys_attribution_summary.md` 的 TensorRT layer-name heuristic mapping，真实时间仍是 CUDA kernel `correlationId` attribution。它显示 TensorRT 后 `matmul` 与 `aggregation` 仍是 `stage2/context` 的主要 residual runtime；`relu_qk` 已被 pointwise fusion 压低，不应单独当作最高收益候选。Phase 3 的 LiteMLA MVP 更适合表述为 `attention_core`，而不是狭义的 `relu_qk`。

当前 TensorRT EngineInspector / ONNX node name 映射结果：

| 项目 | 结果 |
|---|---|
| 脚本 | `phase2/scripts/inspect_trt_engine.py` |
| Summary | `phase2/results/metrics/trt_engine_inspection_summary.md` / `.json` |
| ONNX nodes | `393` |
| TensorRT engine layers | `155` |
| Overall layer-count reduction | `60.56%` |
| EngineInspector detail | `layer_names_only` |
| 结构证据 | `PWN(...)` 表明 pointwise/activation fusion；layer name 中的 ` + ` 表明 TensorRT 将多个 ONNX-named ops 合并成一个 engine layer |

说明：EngineInspector / ONNX 映射只能说明结构变化，例如 layer 数减少、PWN pointwise fusion、Conv+Add 融合、stage2 context 中仍存在 MatMul / Reformat / Cast 等结构；它不提供真实 GPU 耗时。真实耗时仍以 `trt_nsys_attribution_summary.md` 的 Nsight SQLite `correlationId` 归因为准。

SegHead bicubic upsample 验证：

- ONNX 中存在 2 个 SegHead `Resize` 节点，属性为 `mode=cubic`、`coordinate_transformation_mode=half_pixel`、`cubic_coeff_a=-0.75`。
- TensorRT 8.6.1 parser/build 对这两个节点无 parser error，FP32 engine 构建和 runtime benchmark 均通过。
- 因此，`SegHead bicubic upsample` 对当前固定 shape / TensorRT 8.6.1 路线不是阻塞项；该结论不外推到动态 shape、其他 cubic 参数组合或 TensorRT 10/11。

---

## 6. 已知风险

| 风险 | 影响 | 初始策略 |
|---|---|---|
| Windows 上 `triton` import 问题 | import EfficientViT 失败 | 复用 Phase 1 延迟注入 stub 思路，必要时抽成 `_compat.py` |
| `wandb` 退出清理 PermissionError | 脚本退出噪声 | Phase 2 脚本启动时禁用/隔离 wandb |
| LiteMLA shape-adaptive branch | 动态 shape 导出不稳定 | 第一版固定 `1024x2048` |
| SegHead bicubic upsample | 已验证当前固定 shape / TensorRT 8.6.1 支持；动态 shape 或其他 TensorRT 版本仍需复验 | 不改 bilinear，不作为当前 Plugin 候选；在报告中记录验证边界 |
| LiteMLA autocast-disabled 语义 | FP16 输出误差可能变大 | 已建立 FP32 baseline；FP16 作为风险实验单独验证 |

---

## 7. 与 Phase 1 / Phase 3 的关系

Phase 1 给出 PyTorch 路径证据：

- `stage0` 是当前 PyTorch 最大 GPU kernel 热点。
- `stage2/context` LiteMLA 是最高区分度 Plugin 主线。
- Phase 1 Plan D 将 LiteMLA 候选细化为 `aggregation-only` / `relu_linear_att-only`、`aggregation + cat + relu_linear_att`、整体 LiteMLA fallback；Phase 2 TensorRT 细分后进一步把第一版 MVP 候选收敛为 `attention_core`，并把收益评估边界改写为 `aggregation + attention_core`。

Phase 2 要验证：

- TensorRT 是否已经自动优化 P2 标准算子链热点。
- LiteMLA 在 TensorRT 后是否仍然是值得手写 Plugin 的残余热点。
- FP16 数值策略是否成为新的工程约束；`bicubic upsample` 已在当前固定 shape / TensorRT 8.6.1 下通过。

这些问题不能只靠端到端 latency 或 engine build 成功来回答。Phase 2 的候选复核必须以 TensorRT Nsight Systems runtime 归因为主证据，EngineInspector / verbose build log 只能说明 engine 结构和 layer 名称，不能替代真实 GPU 时间线。

Phase 3 才开始真正实现 Plugin。
