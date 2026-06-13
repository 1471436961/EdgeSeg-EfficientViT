# Plugin Graph Integration Design

> **状态**：v0.1，Phase 3 Step 6 落盘前方案的执行版。
>
> **目的**：把 Step 5/5.5 已验证的 P1a `relu_linear_att-only` Plugin 接入真实 EfficientViT TensorRT graph。本文只覆盖 P1a graph replacement 与 engine build，不扩大到 P1b，也不继续优化 CUDA kernel。

---

## 1. 设计目标

Step 6 要回答的问题是：

> 已经通过 toy engine 验证的 `EdgesegReluLinearAttention_TRT` Plugin，能否替换真实 EfficientViT ONNX 图中两个 `stage2/context` 的 `relu_linear_att` 子图，并成功构建真实 TensorRT engine？

因此 Step 6 分成两个脚本：

| 脚本 | 职责 |
|---|---|
| [`../scripts/integrate_relu_linear_attention_plugin_onnx.py`](../scripts/integrate_relu_linear_attention_plugin_onnx.py) | 对 Phase 2 ONNX 做 graph surgery，输出 patched ONNX 与 metadata |
| [`../scripts/build_plugin_engine.py`](../scripts/build_plugin_engine.py) | 加载 Plugin DLL，解析 patched ONNX，构建真实 EfficientViT Plugin FP32 engine |

---

## 2. 替换边界

只替换 `backbone.stages.2` 中两个 LiteMLA context block 的 attention core：

| block | Plugin 输入 | Plugin 输出 |
|---|---|---|
| `op_list.1` | `/backbone/stages.2/op_list.1/context_module/main/Concat_output_0` | `/backbone/stages.2/op_list.1/context_module/main/Cast_1_output_0` |
| `op_list.2` | `/backbone/stages.2/op_list.2/context_module/main/Concat_output_0` | `/backbone/stages.2/op_list.2/context_module/main/Cast_1_output_0` |

保留内容：

- qkv Conv
- aggregation depthwise / grouped Conv
- Concat
- proj Conv
- residual Add

删除内容：

- `Concat_output_0 -> Cast_1_output_0` 之间的 Reshape / Slice / Relu / Transpose / Pad / MatMul / Div / Cast 子图，以及该子图专用 Constant / Shape / Gather / Unsqueeze 等辅助节点。

---

## 3. ONNX Plugin 节点约定

新增 ONNX node 使用 TensorRT ONNX parser 的 custom plugin 约定：

| 字段 | 值 |
|---|---|
| `op_type` | `EdgesegReluLinearAttention_TRT` |
| `domain` | 空字符串 |
| `plugin_version` | `"1"` |
| `plugin_namespace` | `"edgeseg"` |
| `dim` | `16` |
| `eps` | `1e-15` |
| `input_c` | `384` |
| `height` | `64` |
| `width` | `128` |

依据：TensorRT 8.6.1 samples 中 `onnx_packnet/post_processing.py` 使用 `plugin_version` / `plugin_namespace` 属性让 ONNX parser 查找 Plugin Creator。

---

## 4. 子图定位策略

第一版不依赖 `onnx-graphsurgeon`，而使用原生 `onnx` API。

定位方式：

1. 对每个 block 固定 `plugin_input` 和 `plugin_output` tensor name。
2. 从 `plugin_output` 的 producer node 反向 DFS。
3. 遇到 `plugin_input` 时停止。
4. 收集反向路径上所有 producer nodes。
5. 删除这些 nodes，并追加一个 Plugin node：`plugin_input -> plugin_output`。

这个策略比硬编码节点序号稳，因为它适应 op_list.1 和 op_list.2 中静态 / 动态 reshape 展开差异。

---

## 5. Engine build 策略

`build_plugin_engine.py` 基于 Phase 2 `build_trt_engine.py`，但额外处理：

- 在 `import tensorrt` 后、ONNX parser 前加载 `edgeseg_relu_linear_attention_plugin.dll`。
- 调用 `trt.init_libnvinfer_plugins(logger, "")`。
- 确认 Plugin Creator 在 registry 中存在。
- 解析 patched ONNX 并构建 engine。

输出：

| 产物 | 路径 |
|---|---|
| patched ONNX | `phase3/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048_relu_linear_att_plugin.onnx` |
| graph surgery metadata | `phase3/results/metrics/relu_linear_attention_plugin_onnx_integration.json` |
| Plugin engine | `phase3/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_relu_linear_att_plugin_fp32.engine` |
| build metadata | `phase3/results/metrics/relu_linear_attention_plugin_engine_build.json` |

---

## 6. 已知风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| TensorRT ONNX parser 找不到 Plugin Creator | engine build 失败 | metadata 记录 parser errors；优先检查 `plugin_namespace` 与 DLL 加载 |
| ONNX graph surgery 删除范围过宽 | 图结构损坏 | 使用反向 DFS，且只针对两个固定 block |
| ONNX checker 不认识 custom op | checker 可能失败 | checker 仅作为 warning 记录，不阻塞 parser 实测 |
| Plugin 输出 shape 推断缺失 | parser 失败 | Plugin `getOutputDimensions()` 固定返回 `[1,128,64,128]` |
| Step 6 build 成功但数值未验证 | 不能进入性能结论 | Step 7 再复用 Phase 2 benchmark 做端到端 correctness / latency |

---

## 7. 通过条件

Step 6 通过条件：

1. patched ONNX 成功生成，且两个 target block 均被替换。
2. patched ONNX 中 Plugin node 数量为 2。
3. TensorRT registry 能找到 `EdgesegReluLinearAttention_TRT` Creator。
4. TensorRT parser 能解析 patched ONNX。
5. serialized Plugin engine 成功写入 `phase3/results/engines/`。

Step 6 不要求：

- 输出数值与 baseline 对齐；
- latency 加速；
- Nsight attribution 完整解释。

这些属于 Step 7/8。

---

## 8. Step 6 实测结果

Step 6 已按本文方案完成：

| 项 | 结果 |
|---|---|
| patched ONNX | `phase3/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048_relu_linear_att_plugin.onnx` |
| ONNX metadata | [`../results/metrics/relu_linear_attention_plugin_onnx_integration.json`](../results/metrics/relu_linear_attention_plugin_onnx_integration.json) |
| Plugin engine | `phase3/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_relu_linear_att_plugin_fp32.engine` |
| Engine metadata | [`../results/metrics/relu_linear_attention_plugin_engine_build.json`](../results/metrics/relu_linear_attention_plugin_engine_build.json) |
| 原始 ONNX nodes | `393` |
| patched ONNX nodes | `262` |
| Plugin nodes | `2` |
| TensorRT parser errors | `[]` |
| TensorRT network layers | `241` |

TensorRT parser 日志确认：

- `No importer registered for op: EdgesegReluLinearAttention_TRT. Attempting to import as plugin.`
- `Searching for plugin: EdgesegReluLinearAttention_TRT, plugin_version: 1, plugin_namespace: edgeseg`
- `Successfully created plugin: EdgesegReluLinearAttention_TRT`

注意：本次 build 命令在工具层面触发 timeout，但 stdout 已打印 engine build complete，且 metadata `status=ok`、engine 文件存在、parser errors 为空。因此本阶段按文件与 metadata 判定 Step 6 通过。
