# `trt_nsys_attribution` 设计说明

> **关联阶段**：[`phase2/README.md`](../README.md) Step 6  
> **前置产物**：`phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine`  
> **状态**：v1.0，Step 6 第一版已执行完成；已产出 TensorRT Nsight SQLite attribution summary，用于复核 Phase 1 候选在 TensorRT 后的残余热点。

---

## 1. 设计目标

Phase 2 已经证明 TensorRT FP32 baseline 能构建、能推理、比 PyTorch baseline 更快。但这只能说明 **TensorRT 端到端优化有效**，不能说明：

- TensorRT 自动优化后，Phase 1 的 `stage0` / `stage2 LiteMLA` / `head` 候选是否仍是残余热点。
- `aggregation + cat + relu_linear_att` 是否仍然值得进入 Phase 3 Plugin 主线。
- P2 标准算子链热点是否已经被 TensorRT / cuDNN 充分处理。

因此 Step 6 的职责是：用 Nsight Systems 对 TensorRT runtime 做归因，回答 **Phase 1 候选在 TensorRT 后是否仍成立**。

---

## 2. 主证据与辅助证据

### 主证据：Nsight Systems runtime trace

Phase 2 继续沿用 Phase 1 的方法论：以 Nsight Systems 的真实运行时间线和 CUDA kernel 事件为主证据。

采集口径：

- trace：`cuda,nvtx`
- 目标脚本：`phase2/scripts/benchmark_trt_engine.py`
- 主对象：FP32 engine
- 输入：固定 `phase1/data/city_asset_cityscapes_like.png`
- 分辨率：`1024x2048`
- 计时范围：engine execute only，与 benchmark JSON 口径一致

Windows 普通权限下，CPU sampling / context switch / WDDM tracing 可能不可用；这类信息只能作为限制项记录，不能作为主要归因依据。

### 辅助证据：EngineInspector / verbose layer dump

EngineInspector、verbose build log 或 layer dump 只能回答 engine 结构问题，例如：

- TensorRT engine 有哪些 layer。
- layer type / name / tensor shape 大致如何。
- 某些 ONNX 节点是否被融合、重排或消失。

它们不能替代 Nsight runtime profiling，因为 layer 存在不等于耗时高，layer 融合也不等于残余热点消失。

---

## 3. 预期产物

```text
phase2/results/nsight/
  trt_fp32_fullres.nsys-rep         # 运行产物，不入 git
  trt_fp32_fullres.sqlite           # 运行产物，不入 git

phase2/results/metrics/
  trt_benchmark_b0_cityscapes_1024x2048_fp32_nsys.json
  trt_nsys_attribution_summary.md   # 入 git
  trt_nsys_attribution_summary.json # 入 git
```

若导出 SQLite 失败，报告中必须记录原因，并保留 `.nsys-rep` 截图/观察作为降级证据；但不得把截图当作定量归因表的替代品。

---

## 4. 关键问题

Step 6 最终至少回答以下问题：

1. TensorRT FP32 后，kernel 类型分布相比 PyTorch Plan B 是否明显变化。
2. TensorRT 后是否仍存在大量小 kernel / launch 密度问题。
3. `stage0` / `head` 这类标准 MBConv / Conv 链是否仍是残余热点。
4. LiteMLA 相关路径是否仍有可观察的残余 kernel time。
5. Phase 3 的 P1a / P1b / P1c / P2 排序是否需要调整。

---

## 5. 与 Phase 1 的可比性边界

Phase 1 的 PyTorch 模型可以通过 Python hooks / monkey patch 插入模块级 NVTX；TensorRT engine 内部不能用同样方式直接插入 `stage0/stage2/head` 模块级 NVTX。

因此 Phase 2 的可比性不是“相同 NVTX 层级一一对齐”，而是：

- 使用相同 Nsight Systems 工具链。
- 使用相同输入、分辨率、权重和 batch size。
- 使用相同 `cuda,nvtx` 主 trace 口径。
- 使用 benchmark JSON 的 CUDA Events latency 作为端到端执行时间锚点。
- 结合 TensorRT layer dump 辅助解释 kernel / layer 对应关系。

报告中必须明确这一限制，避免把 TensorRT engine 内部归因解释得比证据更细。

---

## 6. 建议执行顺序

1. 给 `benchmark_trt_engine.py` 增加最小 NVTX 标记：
   - `trt/warmup`
   - `trt/measure`
   - `trt/execute`
2. 用 Nsight Systems 采集 FP32 engine benchmark。
3. 导出 SQLite。
4. 先做 kernel type / CUDA API / memory copy 汇总。
5. 如可行，再结合 EngineInspector layer dump 做 layer 结构解释。
6. 写入 `trt_nsys_attribution_summary.md`，并在 `tensorrt_baseline_report.md` 中引用。

---

## 7. 验收标准

- Nsight trace 能打开，能看到 TensorRT engine execute 区间。
- CUDA kernel 事件足够完整，能形成 kernel type 级别统计。
- 汇总表明确说明哪些结论是 Nsight 运行时证据，哪些只是 EngineInspector 辅助解释。
- 明确回答 Phase 3 候选是否仍沿用 Phase 1 的 P1a/P1b/P1c 排序，或需要调整。

---

## 8. 当前 Step 6 实测结果

本轮使用 Nsight Systems `cuda,nvtx` trace 采集 FP32 TensorRT engine execute 路径，并导出 SQLite 后由 `phase2/scripts/analyze_trt_nsys_attribution.py` 汇总。

运行口径：

- benchmark metadata：`phase2/results/metrics/trt_benchmark_b0_cityscapes_1024x2048_fp32_nsys.json`
- Nsight SQLite：`phase2/results/nsight/trt_fp32_fullres.sqlite`
- 汇总表：`phase2/results/metrics/trt_nsys_attribution_summary.md`
- warmup / measure：`20 / 100`
- attribution 方法：TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`
- 重要纪律：不使用 NVTX range duration 作为 GPU component time。

关键结果：

| 项目 | 结果 |
|---|---:|
| CUDA Events latency mean / p50 | `55.242 ms` / `55.237 ms` |
| `trt/execute` kernel avg | `54.454 ms / iter` |
| `trt/execute` launches | `185.0 / iter` |
| Layer-attributed kernel avg | `54.454 ms / iter` |
| Layer attribution / execute kernel time | `100.00%` |

TensorRT 后的 residual hotspot 排序：

| Group | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---|---:|---:|---:|
| `stage0` | `14.669` | `26.94%` | `10.0` |
| `stage2` | `12.179` | `22.37%` | `57.0` |
| `stage3` | `7.511` | `13.79%` | `88.0` |
| `stage1` | `7.431` | `13.65%` | `10.0` |
| `head` | `6.608` | `12.14%` | `15.0` |
| `stem` | `6.056` | `11.12%` | `5.0` |

结论边界：

- TensorRT 后 `stage0` 仍是最大 residual hotspot，但主要仍是标准卷积 / pointwise / activation 路径，不自动等价于 Phase 3 Plugin MVP。
- `stage2` 仍是第二大 residual hotspot，且 launches / iter 明显高，说明 Phase 1 的 LiteMLA / stage2 候选仍值得保留。
- TensorRT layer ranges 与 Phase 1 PyTorch Plan B/C/D 模块范围不是一一对应关系；因此 Step 6 支撑的是 residual hotspot 趋势复核，不是 PyTorch 模块耗时的逐项复刻。
