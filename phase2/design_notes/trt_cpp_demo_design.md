# `trt_runtime_demo.cpp` — 设计文档

> **关联阶段**：[`phase2/README.md`](../README.md) Step 7  
> **目标产物**：`phase2/cpp_demo/trt_runtime_demo.cpp`  
> **状态**：v1.1，第一版 TensorRT C++ Runtime smoke demo 已构建并运行通过

---

## 1. 设计目标

Step 7 的目标不是重新做一套 C++ 性能 benchmark，而是验证 Phase 3 TensorRT Plugin 集成前必须具备的 C++ Runtime 链路：

```text
load serialized FP32 engine
-> deserialize TensorRT engine
-> create execution context
-> inspect input/output bindings
-> allocate CUDA device buffers
-> fill deterministic dummy input
-> enqueue inference on CUDA stream
-> copy output back to host
-> print checksum / min / max / mean
```

只要这条链路跑通，后续 Phase 3 的 Plugin registry、engine build、engine load 和 C++ inference demo 就有了可复用骨架。

---

## 2. 关键取舍

### D1：真实图片预处理 vs deterministic dummy input

**选择：第一版使用 deterministic dummy input。**

Phase 2 的 PyTorch / ONNXRuntime / TensorRT 数值一致性已经由 Python 脚本负责。C++ demo 的职责是验证 TensorRT Runtime API 和 CUDA buffer 链路，不应引入 OpenCV、PNG 解码或 Cityscapes 预处理依赖。

### D2：buffer 管理方式

**选择：C++ 原生 CUDA API。**

使用：

```text
cudaMalloc
cudaMemcpyAsync
cudaStreamCreate
IExecutionContext::enqueueV2
cudaStreamSynchronize
```

这样最接近 Phase 3 Plugin 的实际集成环境，也避免把 Python/PyTorch 的 buffer 语义带入 C++ demo。

### D3：支持范围

**选择：第一版只要求 FP32 engine。**

当前 Phase 2 baseline 已确认 FP32 是 MX250 / TensorRT 8.6.1 的主线；FP16 是风险实验且慢于 FP32。C++ demo 第一版只验证 FP32 engine，若传入非 FP32 binding 会报错退出。

### D4：构建系统

**选择：CMake + MSVC。**

CMakeLists 显式要求用户传入：

```text
-DTENSORRT_ROOT=E:/NVIDIA/TensorRT-8.6.1.6
```

当前已为 `efficientvit` conda 环境安装 `cmake 4.2.3`，并将最小 MSVC C++ Build Tools 组件安装到：

```text
E:\VSBuildTools
```

构建时通过以下脚本激活 MSVC 环境：

```text
E:\VSBuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64
```

---

## 3. 已知风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| MSVC 工具链未激活 | CMake configure/build 失败 | 通过 `VsDevCmd.bat` 激活后再调用 CMake |
| TensorRT/cuDNN/cuBLAS/NVRTC DLL 不在 PATH | 运行 demo 时找不到 DLL 或静默退出 | 运行前把 TensorRT `lib/bin`、CUDA `bin`、conda env 下的 `nvidia/*/bin` 加入 PATH |
| C++ demo 使用 dummy input | 不能证明语义一致性 | 语义一致性仍引用 Python benchmark；C++ demo 只做 runtime smoke |
| 只支持 FP32 binding | FP16 engine 不能直接复用 | Phase 2 主线本来采用 FP32；FP16 后续可扩展 |

---

## 4. 验收标准

第一版 Step 7 通过条件：

- CMake 工程文件存在，并能在 MSVC 工具链可用时配置。
- C++ 源码能加载 TensorRT FP32 engine。
- 程序能打印所有 binding 的 name / role / dtype / shape / bytes。
- 程序能执行至少 1 次 inference。
- 程序能打印 output checksum / min / max / mean / first values。

当前已通过：

```text
phase2/cpp_demo/build/trt_runtime_demo.exe phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine 1
```

验证结果：engine 反序列化成功，binding 信息正确，单次 enqueue 推理成功，输出 summary 正常。
