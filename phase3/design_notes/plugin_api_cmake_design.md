# Plugin API 与 CMake 构建设计

> **状态**：v0.2，已吸收 Phase 3 Step 4 skeleton 实测结果。
>
> **目的**：在开始写 C++/CUDA/TensorRT Plugin 代码前，先固定第一版 `relu_linear_att-only` Plugin 的 API、序列化字段、构建目标、DLL 加载方式和验证路径。本文不实现 kernel。

---

## 1. 输入依据

| 依据 | 文件 | 用途 |
|---|---|---|
| Stage2 tensor contract | [`stage2_context_tensor_contract.md`](stage2_context_tensor_contract.md) | 确认第一版 Plugin 输入输出 shape |
| Phase 2 C++ demo | [`../../phase2/cpp_demo/CMakeLists.txt`](../../phase2/cpp_demo/CMakeLists.txt) | 复用 Windows + TensorRT + CUDA + MSVC 构建口径 |
| TensorRT 8.6.1 headers | `E:/NVIDIA/TensorRT-8.6.1.6/include` | 确认本地可用 `IPluginV2DynamicExt` / `IPluginCreator` / `REGISTER_TENSORRT_PLUGIN` |
| Phase 2 benchmark | [`../../phase2/scripts/benchmark_trt_engine.py`](../../phase2/scripts/benchmark_trt_engine.py) | 后续复用同口径 latency 与输出对齐 |

---

## 2. 第一版 Plugin 边界

第一版只做 P1a MVP：`relu_linear_att-only`。

| 项 | 约定 |
|---|---|
| Plugin 名称 | `EdgesegReluLinearAttention_TRT` |
| Plugin 版本 | `1` |
| TensorRT namespace | `edgeseg` |
| 替换范围 | `Concat_output_0 -> Cast_1_output_0` |
| 输入 | cat 后 qkv，`[1,384,64,128]` |
| 输出 | attention 输出，`[1,128,64,128]` |
| layout | NCHW / TensorRT `kLINEAR` |
| dtype | FP32 only |
| 是否带权重 | 否 |
| 后续接入 | 输出继续喂给原有 `proj/conv/Conv` |

第一版明确不做：

- 不支持 FP16 / INT8。
- 不支持 dynamic shape。
- 不支持 batch > 1。
- 不把 aggregation 或 proj 纳入 Plugin。
- 不在 Step 3 写 CUDA kernel。

---

## 3. TensorRT Plugin 接口选择

### 3.1 选择 `IPluginV2DynamicExt`

本地 TensorRT 8.6.1 头文件确认存在：

- `nvinfer1::IPluginV2DynamicExt`
- `nvinfer1::IPluginCreator`
- `REGISTER_TENSORRT_PLUGIN`

因此第一版采用 `IPluginV2DynamicExt`，原因：

1. TensorRT 8.6.1 Windows zip 明确支持，兼容当前环境。
2. Phase 2 engine 使用 explicit batch，`IPluginV2DynamicExt` 与 explicit batch / shape expression API 匹配。
3. 虽然第一版 fixed shape，但使用 dynamic interface 可以让 `getOutputDimensions()` 与后续扩展保持一致。

暂不使用更高版本 Plugin V3，因为当前环境锁定 TensorRT 8.6.1。

### 3.2 必须实现的方法

第一版 Plugin 类需要实现：

| 方法 | 第一版职责 |
|---|---|
| `getNbOutputs()` | 返回 1 |
| `getOutputDimensions()` | 输入 `[N,C,H,W]` 输出 `[N,C/3,H,W]` |
| `supportsFormatCombination()` | 只接受 FP32 + `TensorFormat::kLINEAR`，输入输出一致 |
| `configurePlugin()` | 校验输入 shape 为 `[1,384,64,128]`，输出 shape 为 `[1,128,64,128]` |
| `getWorkspaceSize()` | 第一版返回 0，除非 kernel 设计阶段需要 workspace |
| `enqueue()` | 调用 CUDA kernel，使用 TensorRT 传入的 `cudaStream_t` |
| `getSerializationSize()` / `serialize()` | 序列化 `dim`、`eps`、固定 shape 元信息 |
| `clone()` | 复制 Plugin 配置 |
| `getPluginType()` / `getPluginVersion()` | 返回固定名称与版本 |
| `destroy()` | `delete this` |
| `setPluginNamespace()` / `getPluginNamespace()` | 支持 namespace |
| `getOutputDataType()` | 返回 FP32 |
| `initialize()` / `terminate()` | 第一版可无状态 |

对应 Creator 需要实现：

| 方法 | 第一版职责 |
|---|---|
| `getPluginName()` | `EdgesegReluLinearAttention_TRT` |
| `getPluginVersion()` | `1` |
| `getFieldNames()` | 暴露 `dim`、`eps`、`input_c`、`height`、`width` |
| `createPlugin()` | 从 `PluginFieldCollection` 创建新实例 |
| `deserializePlugin()` | 从 engine 序列化数据恢复实例 |
| `setPluginNamespace()` / `getPluginNamespace()` | 固定 `edgeseg` |

---

## 4. Plugin 字段与序列化

第一版 Plugin 不带权重，但仍需要记录少量参数，避免把所有内容硬编码进 kernel。

| 字段 | 类型 | 默认值 | 用途 |
|---|---|---|---|
| `dim` | int32 | 16 | LiteMLA split 维度，q/k/v 每段维度 |
| `eps` | float32 | `1e-15` | normalization 分母保护 |
| `input_c` | int32 | 384 | 固定输入通道数，用于校验 |
| `height` | int32 | 64 | 固定 H，用于校验 |
| `width` | int32 | 128 | 固定 W，用于校验 |

派生值：

- `output_c = input_c / 3 = 128`
- `heads_after_cat = input_c / (3 * dim) = 8`
- `spatial = height * width = 8192`

序列化只保存字段值，不保存任何 device buffer。

---

## 5. C++ / CUDA 文件布局

Step 4 开始实现时建议使用以下布局：

```text
phase3/plugin/
|-- CMakeLists.txt
|-- include/
|   `-- edgeseg_relu_linear_attention_plugin.h
`-- src/
    |-- edgeseg_relu_linear_attention_plugin.cpp
    `-- relu_linear_attention_kernel.cu
```

命名原则：

- 文件名使用 `relu_linear_attention`，对应第一版 MVP。
- Plugin 类型名使用 `EdgesegReluLinearAttentionPlugin`。
- Creator 类型名使用 `EdgesegReluLinearAttentionPluginCreator`。
- CMake target 使用 `edgeseg_relu_linear_attention_plugin`。

---

## 6. CMake 构建方案

### 6.1 构建目标

第一版构建一个 Windows 动态库：

```text
phase3/plugin/build/edgeseg_relu_linear_attention_plugin.dll
```

对应 import library：

```text
phase3/plugin/build/edgeseg_relu_linear_attention_plugin.lib
```

### 6.2 CMake 口径

沿用 Phase 2 C++ demo 的工具链约束：

| 项 | 约定 |
|---|---|
| CMake minimum | 3.20 |
| C++ standard | C++17 |
| CUDA standard | C++17 |
| TensorRT root | `E:/NVIDIA/TensorRT-8.6.1.6` |
| CUDA Toolkit | CUDA 12.4 |
| Windows compiler | MSVC Build Tools |
| GPU arch | `sm_61`，即 `CMAKE_CUDA_ARCHITECTURES=61` |

目标链接：

- `nvinfer`
- `nvinfer_plugin`
- `CUDA::cudart`

MSVC 编译选项：

- `/W4`
- `/permissive-`
- `/utf-8`

CUDA 编译选项第一版保持保守，不启用 fast math，避免数值对齐阶段引入额外变量。

### 6.3 构建命令草案

后续 Step 4 可采用与 Phase 2 类似的命令：

```powershell
cmd /c "`"E:/VSBuildTools/Common7/Tools/VsDevCmd.bat`" -arch=x64 -host_arch=x64 && D:/software/anaconda3/envs/efficientvit/Library/bin/cmake.exe -S phase3/plugin -B phase3/plugin/build -G `"NMake Makefiles`" -DTENSORRT_ROOT=E:/NVIDIA/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES=61 && D:/software/anaconda3/envs/efficientvit/Library/bin/cmake.exe --build phase3/plugin/build --config Release"
```

---

## 7. Plugin 注册与加载策略

### 7.1 注册

TensorRT 8.6.1 的 `REGISTER_TENSORRT_PLUGIN` 宏会调用内置 registrar，并默认注册到空 namespace。第一版 Plugin 需要固定 namespace 为 `edgeseg`，因此实际实现采用自定义静态 registrar：

```cpp
namespace {
edgeseg::EdgesegReluLinearAttentionPluginCreator gCreator;

struct EdgesegReluLinearAttentionRegistrar {
    EdgesegReluLinearAttentionRegistrar() {
        getPluginRegistry()->registerCreator(gCreator, "edgeseg");
    }
};

EdgesegReluLinearAttentionRegistrar gRegistrar;
} // namespace
```

这会在 DLL 被加载时把 Creator 注册到 TensorRT Plugin Registry 的 `edgeseg` namespace。

### 7.2 Python builder 加载

后续构建含 Plugin 的 engine 前，Python 脚本必须先加载 DLL：

```python
import ctypes
ctypes.CDLL(str(plugin_dll_path))
trt.init_libnvinfer_plugins(logger, "")
```

加载顺序必须早于 ONNX parser / Network API 查找 Plugin Creator。

### 7.3 C++ runtime 加载

C++ runtime demo 后续需要在反序列化含 Plugin 的 engine 前加载 DLL：

```cpp
LoadLibraryA("edgeseg_relu_linear_attention_plugin.dll");
initLibNvInferPlugins(&logger, "");
```

否则 `deserializeCudaEngine()` 可能因为找不到 Plugin Creator 失败。

---

## 8. Graph 集成策略

Step 3 只确定 Plugin API 与构建方案，不立刻做 graph surgery。后续建议顺序：

1. **Step 4：Plugin skeleton**
   - 先能编译出 DLL。
   - 能被 TensorRT registry 找到 Creator。
   - 能在最小 toy network 中加入 Plugin layer 并 build engine。

2. **Step 5：实现真实 CUDA kernel 与单层对齐**
   - 用真实 `relu_linear_att` 数学替换 skeleton zero-fill。
   - 在 toy/plugin 单层层面与 PyTorch reference 对齐。
   - 本步不做完整 EfficientViT graph surgery。

3. **Step 6：接入真实 EfficientViT subgraph**
   - 优先用 ONNX graph surgery 把 `Concat_output_0 -> Cast_1_output_0` 子图替换为 custom op。
   - 若 ONNX parser custom op 路线不稳定，再评估 TensorRT Network API 手动重建局部网络。

4. **Step 7：benchmark + correctness**
   - 复用 Phase 2 `benchmark_trt_engine.py` 的 execute-only CUDA Events 口径。
   - 对比 TensorRT FP32 baseline 与 Plugin engine。

5. **Step 8：Nsight attribution + report**
   - 采集 Plugin engine Nsight trace，更新 attribution summary。
   - 汇总到 `integration_validation_report.md`。

---

## 9. 数值与性能验收

第一版 `relu_linear_att-only` Plugin 分层通过条件：

| 层级 | 通过条件 |
|---|---|
| Build | DLL 编译成功，导出后能被 TensorRT 加载 |
| Registry | Python/C++ 中能找到 `EdgesegReluLinearAttention_TRT` Creator |
| Engine build | 含 Plugin 的 engine 构建成功 |
| Runtime | engine 能执行 `20 warmup + 100 measure` |
| Correctness | 与 TensorRT FP32 baseline 记录 `max_abs_diff`、`mean_abs_diff`、cosine similarity、argmax agreement |
| Latency | 使用 Phase 2 相同 CUDA Events execute-only 口径 |
| Nsight | 能看到 Plugin layer/kernel，且 residual hotspot 变化可解释 |

第一版不要求一定加速。若 Plugin engine 没有加速，但完成注册、替换、数值对齐和 Nsight 可解释，也仍然是有效工程里程碑。

---

## 10. Step 4 实测结果

Step 4 已完成 skeleton 实现与 toy engine build：

| 项 | 结果 |
|---|---|
| Plugin DLL | `phase3/plugin/build/edgeseg_relu_linear_attention_plugin.dll` |
| Plugin Creator | `EdgesegReluLinearAttention_TRT` |
| Namespace | `edgeseg` |
| Toy input | `[1,384,64,128]` |
| Toy output | `[1,128,64,128]` |
| Toy engine | `phase3/results/engines/relu_linear_attention_toy_fp32.engine` |
| Metadata | [`../results/metrics/relu_linear_attention_toy_build.json`](../results/metrics/relu_linear_attention_toy_build.json) |

注意：Step 4 skeleton 的 CUDA enqueue 当前只做 zero-fill，用于验证 Plugin 调用链路；真实 `relu_linear_att` 数学在 Step 5 实现。

---

## 11. Step 3/4 结论

1. 第一版 Plugin API 采用 `IPluginV2DynamicExt`。
2. 第一版只支持 FP32、NCHW、fixed shape `[1,384,64,128] -> [1,128,64,128]`。
3. 第一版 Plugin 不带权重，只序列化 `dim/eps/input_c/height/width`。
4. 构建目标是 Windows DLL：`edgeseg_relu_linear_attention_plugin.dll`。
5. Step 4 已证明 Plugin skeleton 可编译、Creator 可注册、toy engine 可构建。
6. 后续 Step 5 只实现真实 CUDA kernel 与单层对齐；Step 6 再做真实 EfficientViT graph 集成，不直接把 graph surgery 与数学实现混在一次改动里。
