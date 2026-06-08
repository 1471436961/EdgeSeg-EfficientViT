# `benchmark_trt_engine.py` — 设计文档

> **关联阶段**：[`phase2/README.md`](../README.md)
> **输入 engine**：`phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine`
> **状态**：v1.1，第一版 FP32 TensorRT engine benchmark 已完成。

---

## 1. 设计目标

`benchmark_trt_engine.py` 的职责是加载已构建的 TensorRT engine，测量 engine runtime latency，并与 PyTorch CUDA reference 输出做数值对齐。

它不构建 engine，也不做完整数据 pipeline benchmark。输入预处理、GPU buffer 分配、PyTorch reference 计算均不计入 TensorRT latency。

---

## 2. 执行流程

```text
parse_args()
prepare_runtime_paths()
load_engine()
build_input_tensor()
run_pytorch_reference()
allocate_trt_bindings()
warmup()
measure_execute_async_v2_with_cuda_events()
compare_trt_output_vs_pytorch()
write_metadata_json()
print_summary()
```

默认输出：

```text
phase2/results/metrics/trt_benchmark_b0_cityscapes_1024x2048_fp32.json
```

---

## 3. 关键取舍

### D1：计时范围

**选择：只测 TensorRT engine execute。**

计时范围是：

```text
context.execute_async_v2(...)
```

不包含：

- 图像读取；
- PIL resize / `[0,1]` 预处理；
- H2D 输入拷贝；
- D2H 输出拷贝；
- PyTorch reference。

原因：Phase 1 PyTorch baseline 也是 forward-only latency，TensorRT baseline 必须保持可比口径。

### D2：计时工具

**选择：CUDA Events。**

沿用 Phase 1 的主口径：

- warmup 后再测量；
- 每次 measured iteration 用 `start.record()` / `end.record()`；
- `end.synchronize()` 只用于读取该次 CUDA Event；
- 报告 mean / std / min / max / p50 / p95 / p99 和 samples。

### D3：buffer 管理方式

**选择：用 PyTorch CUDA tensor 作为 TensorRT binding buffer。**

第一版不引入 `pycuda` / `cuda-python` 新依赖，而是：

```python
bindings[input_idx] = int(input_tensor.data_ptr())
bindings[output_idx] = int(output_tensor.data_ptr())
context.execute_async_v2(bindings, stream_handle)
```

这样能复用当前 PyTorch CUDA runtime 和 CUDA stream，减少额外依赖。但必须保证 tensor：

- 在执行期间生命周期有效；
- shape / dtype 与 TensorRT binding 完全一致；
- contiguous；
- 位于 CUDA device。

### D3.1：TensorRT import 顺序

**选择：先准备 TensorRT DLL 路径并 import TensorRT，再 import PyTorch / `export_onnx.py`。**

实测如果脚本顶部先 import PyTorch，PyTorch 自带的 CUDA/cuDNN DLL 可能先进入进程，导致 TensorRT 8.6.1 import 报 Windows DLL 解析错误。第一版脚本因此延迟 import PyTorch reference 相关函数，确保 TensorRT 8.6.1 的 `nvinfer` / cuDNN 8 / cuBLAS 路径先完成解析。

### D4：输出对齐对象

**选择：第一版对齐 PyTorch CUDA reference。**

原因：

- Phase 1/2 的真实权重与输入预处理都以 PyTorch 为源头。
- ONNXRuntime 已经在 `export_onnx.py` 中完成过第一层对齐。
- TensorRT build 日志出现 INT64 -> INT32 cast / clamp 提示，必须直接检查 TensorRT vs PyTorch logits 差异。

实测第一版 FP32 benchmark 的严格 `atol=1e-4, rtol=1e-4` allclose 未通过，但 `max_abs_diff≈2.69e-4`、`mean_abs_diff≈2.54e-5`、`cosine≈0.99999999998`，并且当前样图的 segmentation argmax pixel agreement 为 100%。因此 Phase 2 报告应使用“数值接近且语义输出一致”的保守表述，而不是“逐元素严格一致”。

---

## 4. JSON 口径

benchmark metadata 记录：

- engine 路径、大小、SHA256；
- 输入图 hash 和输入 tensor hash；
- TensorRT version、binding 信息、network layer 数；
- latency samples 与统计指标；
- PyTorch vs TensorRT 输出误差；
- CUDA memory peak；
- TensorRT / cuDNN / cuBLAS / NVRTC 版本；
- runtime DLL 目录注入情况。

---

## 5. 已知风险

| 风险 | 影响 | 处理 |
|---|---|---|
| TensorRT build 中 INT64 -> INT32 cast / clamp | 输出可能和 PyTorch 有差异 | benchmark 必须做 logits diff |
| PyTorch tensor binding 用法依赖 tensor 生命周期 | 若 tensor 被释放会导致非法访问 | 输入/输出 tensor 在整个 benchmark 函数内持有 |
| 不含 H2D/D2H | 不是完整 pipeline latency | JSON `scope` 明确写 `engine_execute_only_no_preprocess_no_h2d_no_d2h` |
| TensorRT engine 与 GPU/版本绑定 | 不可跨机器复用 | `.engine` 不入 git，metadata 入 git |

---

## 6. 当前结果与下一步

已完成：

1. FP32 benchmark 使用 warmup 20 / measure 100 跑通。
2. TensorRT FP32 p50 / p95 / p99 为 `54.44 ms` / `55.43 ms` / `55.68 ms`。
3. 相比 Phase 1 PyTorch Plan A formal p50 `85.70 ms`，p50 speedup 约 `1.57x`。
4. 严格 `1e-4` allclose 未通过，但 `1e-3` allclose 通过，argmax pixel agreement 为 100%。
5. Phase 2 README Step 5 已更新。

下一步：

1. 撰写 `phase2/tensorrt_baseline_report.md`。
2. 报告中使用“数值接近且语义输出一致”的保守表述。
3. 决定是否继续 FP16 engine build / benchmark；如果继续，需单独记录 LiteMLA FP32 数值保护语义和 MX250 FP16 加速不确定性。
