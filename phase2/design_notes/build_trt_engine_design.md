# `build_trt_engine.py` — 设计文档

> **关联阶段**：[`phase2/README.md`](../README.md)
> **输入产物**：`phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx`
> **状态**：v1.7，FP32 / FP16 engine 均已成功构建；FP16 仅作为风险实验记录；TensorRT Nsight attribution、C++ runtime demo 与 Phase 2 baseline report 均已完成。

---

## 1. 设计目标

`build_trt_engine.py` 的职责是把 Phase 2 已验证的固定 shape ONNX 转换为 TensorRT engine，并产出可复现的构建 metadata。

它不做 benchmark，也不做输出对齐验证。原因是 TensorRT parser/build 失败本身就是重要证据，应该和 runtime latency 测量分开。

---

## 2. 执行流程

```text
parse_args()
prepare_runtime_paths()
import tensorrt
create_builder_network_parser()
parse_onnx()
create_builder_config()
set_workspace_limit()
build_serialized_network()
write_engine()
write_metadata_json()
```

默认输入输出：

```text
input ONNX:  phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx
engine:      phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine
metadata:    phase2/results/metrics/trt_build_b0_cityscapes_1024x2048_fp32.json
precision:   fp32
workspace:   1024 MiB
```

FP16 风险实验输出：

```text
engine:      phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp16.engine
metadata:    phase2/results/metrics/trt_build_b0_cityscapes_1024x2048_fp16.json
precision:   fp16
```

---

## 3. 关键取舍

### D1：TensorRT 11 pip 包 vs TensorRT 8.6.1 手动 zip

**选择：TensorRT 8.6.1 手动 zip。**

实测 `tensorrt-cu12==11.0.0.114` 可以安装，但在 MX250 上创建 builder 失败：

```text
Unsupported SM: 0x601
```

MX250 是 Pascal `sm_61`，需要旧版 TensorRT 路线。`tensorrt==8.6.1` 的 pip 包当前依赖已不可用，因此采用 NVIDIA archived Windows zip：

```text
E:\NVIDIA\TensorRT-8.6.1.6
```

并安装其中的：

```text
python/tensorrt-8.6.1-cp310-none-win_amd64.whl
```

### D2：第一版 FP32 vs 同时做 FP32/FP16

**选择：先构建 FP32，再把 FP16 作为单独风险实验。**

原因：

- Phase 1 PyTorch baseline 与 Phase 2 ONNX 对齐都基于 FP32。
- LiteMLA 有 FP32 数值保护语义，FP16 应单独做风险实验。
- MX250 没有 Tensor Core，`platform_has_fast_fp16=True` 不等于 FP16 一定更快。

脚本实现上，`--precision fp16` 会设置 `BuilderFlag.FP16`，并切换默认输出路径到 `_fp16.engine` / `_fp16.json`。FP16 是否进入报告结论，要等 benchmark 的 latency 与输出误差共同判断。

### D3：build 与 benchmark 是否合并

**选择：拆分。**

`build_trt_engine.py` 只负责 parser/build；后续 `benchmark_trt_engine.py` 再负责 runtime latency 和输出误差。

这样可以把错误分清：

- parser 不支持某个 ONNX op；
- builder tactic 选择或 workspace 失败；
- engine 可构建但 runtime 输出不一致；
- engine 可运行但 latency 不理想。

### D4：DLL 路径显式处理

**选择：脚本在 import `tensorrt` 前显式添加 DLL 目录。**

TensorRT 8.6.1 依赖：

```text
E:\NVIDIA\TensorRT-8.6.1.6\lib
D:\software\anaconda3\envs\efficientvit\Lib\site-packages\nvidia\cudnn\bin
D:\software\anaconda3\envs\efficientvit\Lib\site-packages\nvidia\cublas\bin
D:\software\anaconda3\envs\efficientvit\Lib\site-packages\nvidia\cuda_nvrtc\bin
```

这些路径不应假设已经永久写入系统 PATH。脚本使用 `os.add_dll_directory()` 和进程级 `PATH` 更新，使构建命令自包含。

---

## 4. 元信息 JSON 口径

构建成功时记录：

- ONNX 路径、大小、SHA256。
- engine 路径、大小、SHA256。
- TensorRT 版本、workspace、network IO、layer 数。
- builder 的 `platform_has_fast_fp16` / `platform_has_fast_int8`。
- DLL 目录注入情况。
- Python / TensorRT / cuDNN / cuBLAS / NVRTC 版本。
- 已知风险。

构建失败时仍写 metadata，记录：

- `status="failed"`
- 错误类型与错误信息
- ONNX / engine / TensorRT root 路径
- 时间戳与脚本版本

---

## 5. 已知风险

| 风险 | 影响 | 处理 |
|---|---|---|
| TensorRT 8.6.1 是 archived 版本 | 与当前 CUDA/cuDNN/PyTorch 生态不完全一致 | JSON 记录版本和 DLL 路径 |
| MX250 `sm_61` 只能走旧 TensorRT | 不具备现代 TensorRT 10/11 行为代表性 | 报告中将其定义为本机 baseline，不外推到新 GPU |
| `Resize` bicubic | 当前固定 shape / TensorRT 8.6.1 已验证通过；动态 shape 或其他 TRT 版本仍需复验 | 不改 bilinear，不作为当前阻塞项 |
| LiteMLA 子图 parser 失败 | 当前 FP32 engine 已构建成功，但后续版本/shape 仍可能变化 | parser 错误完整写入 metadata |
| engine 绑定当前 GPU / TensorRT 版本 | 不可跨机器复用 | `.engine` 不入 git，只保留 metadata |
| FP16 build | 已成功构建，但出现 subnormal FP16 weights warning | 通过 benchmark 判断是否可接受 |

---

## 6. 当前结果与后续使用

已完成：

1. `build_trt_engine.py --help` 通过。
2. FP32 engine 构建成功。
3. parser error 为空。
4. engine metadata 已写入 `phase2/results/metrics/trt_build_b0_cityscapes_1024x2048_fp32.json`。
5. FP16 engine 也已构建成功，metadata 写入 `phase2/results/metrics/trt_build_b0_cityscapes_1024x2048_fp16.json`。

后续使用：

1. Phase 3 若引入 Plugin，应以当前 FP32 engine 作为无 Plugin TensorRT baseline。
2. 含 Plugin engine 的构建脚本应复用本脚本的 TensorRT root / DLL path / metadata 记录口径。
3. 若切换 TensorRT 版本、GPU、workspace 或 precision strategy，需要重新生成 engine metadata，并在 Phase 3 报告中与本 baseline 区分。
