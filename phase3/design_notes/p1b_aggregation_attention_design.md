# P1b Aggregation + Attention Plugin Design

> **关联阶段**：[`../README.md`](../README.md)
>
> **状态**：v0.3，P1b skeleton / parser toy 与真实 EfficientViT ONNX surgery build smoke 均已通过。本文确定 `aggregation + cat + relu_linear_att` 的替换边界、权重输入方式和验证顺序；当前只证明 TensorRT parser/build 闭环，不承诺 CUDA kernel 数学正确性或性能。

---

## 1. 设计目标

P1b 的目标是在 P1a `relu_linear_att-only` 已经跑通的基础上，把替换边界向前扩展到 LiteMLA 的中段组合：

```text
qkv output
  -> aggregation depthwise 5x5
  -> aggregation grouped 1x1
  -> cat(original qkv, aggregation output)
  -> relu_linear_att
  -> attention output
```

它不是整体 LiteMLA Plugin：`qkv Conv` 仍保留在 TensorRT/cuDNN 路径中，`proj Conv` 和 residual add 也保留在 Plugin 外部。

---

## 2. 为什么进入 P1b

Phase 2 TensorRT baseline 与 Phase 3 P1a 结果共同说明：

- Phase 2 中 `stage2/context` 的 `aggregation + attention_core` proxy 约为 `5.443 ms / iter`、`38 launches / iter`。
- P1a-3b 后，`relu_linear_att` Plugin layer 约为 `1.310 ms / iter`、`4 launches / iter`。
- P1a-3b 后，`aggregation + plugin` proxy 约为 `3.062 ms / iter`、`30 launches / iter`。
- 其中 `aggregation_preserved` 仍约为 `1.752 ms / iter`、`26 launches / iter`。

因此 P1a 已经证明 `relu_linear_att-only` 可以减少目标边界的 kernel time 和 launch 数，但剩余 runtime 很大一部分还留在 aggregation 与中间 tensor 流转上。P1b 的价值是验证：把 aggregation 与 attention 放进同一个 Plugin 边界后，是否能进一步减少中间写回、读取、concat 相关开销和 launch 数。

---

## 3. 目标范围

P1b 只覆盖两个 `stage2/context` 实例：

| 模块 | ONNX 路径前缀 | 语义 |
|---|---|---|
| block1 context | `/backbone/stages.2/op_list.1/context_module/main` | stage2 第一个 LiteMLA context |
| block2 context | `/backbone/stages.2/op_list.2/context_module/main` | stage2 第二个 LiteMLA context |

固定 Cityscapes 输入 `1x3x1024x2048` 下，P1b 的 tensor contract 为：

| 项 | shape | 说明 |
|---|---:|---|
| runtime input | `[1, 192, 64, 128]` | `qkv/conv/Conv_output_0` |
| output | `[1, 128, 64, 128]` | 原 `Cast_1_output_0`，继续喂给现有 `proj/conv/Conv` |
| dtype | `float32` | 第一版不走 FP16 / Tensor Core |
| layout | `NCHW` | 固定 shape、batch=1 |

每个 block 有独立权重，不能假设共享：

| 权重 | shape | 语义 |
|---|---:|---|
| `aggreg.0.0.weight` | `[192, 1, 5, 5]` | depthwise 5x5，groups=192，padding=2 |
| `aggreg.0.1.weight` | `[192, 16, 1, 1]` | grouped 1x1，groups=12，每组 16 输入通道到 16 输出通道 |

当前 ONNX 中 aggregation 无 bias。

---

## 4. 关键决策

### D1：新增 Plugin 类型，不扩展 P1a 类型

P1b 使用新的 op type：

```text
EdgesegAggregationReluLinearAttention_TRT
```

而不是把现有 `EdgesegReluLinearAttention_TRT` 扩展成多模式 Plugin。原因是 P1a 已经有完整的验证结果和可复现实验链，P1b 权重、输入数量、替换边界、序列化字段都会改变。拆成新类型可以保留 P1a 的 ABI 和历史结果，降低回归风险。

### D2：第一版继续放在同一个 DLL / CMake target

P1b 先复用当前 `edgeseg_relu_linear_attention_plugin.dll` 的构建、加载和注册路径，但新增独立 Creator 与独立 C++ 类。这样可以减少 Windows DLL、TensorRT registry、Python loader 的变量，同时保持 Plugin 类型隔离。

### D3：先做 parser/build 可行性，再写性能 kernel

P1b 的第一风险不是 CUDA 算法，而是 TensorRT ONNX parser 是否能稳定创建一个带权重输入的 Plugin node。因此第一阶段只要求：

1. P1b custom op 可以被 TensorRT parser 识别。
2. depthwise / grouped 1x1 权重能以可追溯方式进入 Plugin。
3. engine 可以 build / deserialize / execute smoke。

在 parser/build 闭环通过前，不进入 P1b CUDA 性能实现。

### D4：优先尝试“权重 initializer 作为 Plugin 输入”

优先方案是让 P1b Plugin node 有 3 个输入：

```text
input0: qkv runtime tensor [1,192,64,128]
input1: aggregation depthwise weight initializer [192,1,5,5]
input2: aggregation grouped pointwise weight initializer [192,16,1,1]
```

这样权重仍保留在 ONNX initializer 中，来源、shape、hash 更容易追溯，也避免把几千个 float 手写进 PluginField attribute。

该方案已通过 toy ONNX parser 验证：TensorRT 8.6.1 可以创建 `EdgesegAggregationReluLinearAttention_TRT` Plugin，`parser_errors=[]`，构建后的网络只有 `qkv` 一个 runtime input，两个 aggregation 权重保持为 initializer / constant 路径，不暴露为外部 engine input。

### D5：Graph surgery 边界

P1b surgery 的替换边界是：

```text
qkv/conv/Conv_output_0 -> Cast_1_output_0
```

被移除的子图包括：

- aggregation depthwise Conv；
- aggregation grouped 1x1 Conv；
- Concat；
- P1a 已替换过的 `relu_linear_att` 子图。

保留在 Plugin 外部的节点包括：

- qkv Conv；
- proj Conv；
- residual Add；
- 其它 stage0/stage1/stage3/head 标准算子。

---

## 5. 验证顺序

P1b 按下面顺序推进：

1. **设计与 contract 落盘**：本文即本步产物。
2. **P1b parser toy / skeleton**：新增 Plugin 类型，先用 zero-fill 或简单 placeholder 输出验证 parser/build，不宣称数值正确。
3. **真实 ONNX surgery build smoke**：替换两个 stage2 context 的 P1b 边界，确认 TensorRT engine 可构建。
4. **单 block 数值验证**：用 PyTorch LiteMLA 子模块输出作为 reference，验证 P1b Plugin 的 aggregation + attention 数学。
5. **端到端 correctness / latency**：复用 Phase 2/3 benchmark 口径，与 TensorRT FP32 baseline 和 P1a engine 对比。
6. **Nsight attribution**：比较 `aggregation + attention` proxy 在 P1b 后的 kernel time / launch 数变化。

---

## 6. 已知风险

| 风险 | 说明 | 应对 |
|---|---|---|
| TensorRT parser 不接受 initializer 作为 Plugin 输入 | 3-input Plugin 是最干净的权重输入方式，但 TensorRT 8.6.1 parser 行为需要实测 | fallback 到 PluginField / serialized weights |
| aggregation 权重布局弄错 | depthwise 5x5 与 grouped 1x1 的 group 语义不同，不能按普通 dense Conv 处理 | 单 block PyTorch reference 验证必须覆盖 aggregation 输出与最终 attention 输出 |
| P1b 可能比 P1a 慢 | aggregation 是标准 Conv 路径，TensorRT/cuDNN 已有优化；naive 自写 aggregation 可能抵消 attention 收益 | 先做 parser/build，再做 microbenchmark 与 Nsight，避免凭直觉扩大边界 |
| 两个 block 权重不同 | block1/block2 shape 相同但数值不同 | surgery 与 Plugin 创建必须逐 block 绑定权重 |
| P1b 不等于 P1c | qkv/proj/residual 仍在 Plugin 外 | 文档和报告中继续区分 P1b、整体 LiteMLA、整网端到端收益 |

---

## 7. 下一步文件

P1b skeleton / parser toy 已新增：

- `phase3/plugin/include/edgeseg_aggregation_relu_linear_attention_plugin.h`
- `phase3/plugin/src/edgeseg_aggregation_relu_linear_attention_plugin.cpp`
- `phase3/scripts/build_p1b_plugin_toy_engine.py`
- `phase3/results/metrics/p1b_aggregation_attention_toy_build.json`
- `phase3/scripts/integrate_p1b_aggregation_attention_plugin_onnx.py`
- `phase3/scripts/build_p1b_plugin_engine.py`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json`
- `phase3/results/metrics/p1b_aggregation_attention_plugin_engine_build.json`

第一版 P1b 实现的验收口径是 parser/build feasibility JSON，而不是 latency 优化结论。

---

## 8. Parser Toy 验证结果

本轮已完成 P1b skeleton / parser toy：

| 项 | 结果 |
|---|---|
| Plugin op type | `EdgesegAggregationReluLinearAttention_TRT` |
| Plugin namespace / version | `edgeseg` / `1` |
| Plugin DLL | `phase3/plugin/build/edgeseg_relu_linear_attention_plugin.dll` |
| Toy ONNX | `phase3/results/onnx/p1b_aggregation_attention_toy.onnx` |
| Toy engine | `phase3/results/engines/p1b_aggregation_attention_toy_fp32.engine` |
| Metadata | [`../results/metrics/p1b_aggregation_attention_toy_build.json`](../results/metrics/p1b_aggregation_attention_toy_build.json) |
| TensorRT parser | 通过，`parser_errors=[]` |
| TensorRT network IO | input `qkv [1,192,64,128]`，output `attention_out [1,128,64,128]` |
| Network layer count | `3` |

解释：

- ONNX checker 对自定义 op 报 “No Op registered” 是预期 warning；本步骤以 TensorRT parser/build 为权威检查。
- TensorRT 日志显示 `Successfully created plugin: EdgesegAggregationReluLinearAttention_TRT`。
- TensorRT build 日志显示网络只有 `1 inputs and 1 output network tensors`，说明两个 aggregation weight initializer 没有变成外部 runtime binding。
- 当前 skeleton 的 `enqueue()` 只 zero-fill 输出，用于验证 Plugin 创建、序列化、shape 和 parser/build 路径；它不是数学正确版本，也不是性能结果。

---

## 9. 真实 ONNX Surgery / Engine Build Smoke

真实 EfficientViT P1b graph surgery 与 TensorRT build smoke 已通过：

| 项 | 结果 |
|---|---|
| ONNX surgery script | [`../scripts/integrate_p1b_aggregation_attention_plugin_onnx.py`](../scripts/integrate_p1b_aggregation_attention_plugin_onnx.py) |
| Engine build script | [`../scripts/build_p1b_plugin_engine.py`](../scripts/build_p1b_plugin_engine.py) |
| Integration metadata | [`../results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json`](../results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json) |
| Engine build metadata | [`../results/metrics/p1b_aggregation_attention_plugin_engine_build.json`](../results/metrics/p1b_aggregation_attention_plugin_engine_build.json) |
| Patched graph | `393 -> 256` nodes |
| Plugin node count | `2` |
| Removed subgraph | block1: `58` nodes；block2: `81` nodes |
| Plugin inputs | `qkv` runtime tensor + depthwise weight initializer + grouped pointwise weight initializer |
| TensorRT parser | 通过，`parser_errors=[]` |
| TensorRT network IO | input `input [1,3,1024,2048]`，output `segout [1,19,128,256]` |
| TensorRT layer count | `239` |
| Engine size | `3,585,252` bytes |

解释：

- 真实 ONNX surgery 删除了每个目标 block 中从 aggregation depthwise Conv 到 `Cast_1_output_0` 的子图，并用 P1b Plugin node 替换。
- 两个 aggregation 权重仍作为 initializer 输入进入 Plugin node：`aggreg.0.0.weight [192,1,5,5]` 与 `aggreg.0.1.weight [192,16,1,1]`。
- TensorRT 日志显示两个 P1b Plugin node 都被成功创建。
- 构建命令在 Codex 工具层面触发了超时，但在超时前已经打印 engine build complete，且 metadata/engine 文件均已落盘并通过 JSON 断言校验。
- 当前 P1b Plugin skeleton 仍只 zero-fill 输出，所以该 engine 只能证明真实 graph parser/build 可行，不能用于 correctness、latency 或 Nsight attribution 结论。

下一步应进入 P1b 单 block 数值验证设计：在真实 CUDA 实现之前，先构造 PyTorch LiteMLA block-level reference，确保 aggregation 权重布局与 P1b 替换边界完全正确。
