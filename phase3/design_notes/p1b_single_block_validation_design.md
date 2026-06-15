# P1b 单 Block 数值验证设计

> **关联阶段**：[`../README.md`](../README.md) Step 8.10 / Step 8.11
>
> **状态**：v0.3，PyTorch reference 捕获已通过并写入 [`../results/metrics/p1b_stage2_reference_capture.json`](../results/metrics/p1b_stage2_reference_capture.json)；P1b 第一版 CUDA 数学路径已通过两个真实 `stage2/context` block 的 toy/plugin correctness 验证，结果写入 [`../results/metrics/p1b_aggregation_attention_plugin_validation.json`](../results/metrics/p1b_aggregation_attention_plugin_validation.json)。

---

## 1. 设计目标

P1b 的目标边界是：

```text
qkv output
  -> aggregation depthwise 5x5
  -> aggregation grouped 1x1
  -> cat(original qkv, aggregation output)
  -> relu_linear_att
  -> attention output
```

P1b skeleton 已经证明 TensorRT parser/build 可以接受这个边界；随后第一版 CUDA 数学路径已经证明 block-local 输出可与 PyTorch reference 对齐。单 block 验证的目的不是测端到端性能，而是回答一个更基础的问题：

```text
Plugin(qkv, depthwise_weight, pointwise_weight)
  ~= module.relu_linear_att(torch.cat([qkv, module.aggreg[0](qkv)], dim=1))
```

这一步通过后，才有资格进入真实 EfficientViT P1b engine correctness / latency / Nsight attribution。

---

## 2. 为什么先做单 Block 验证

P1b 比 P1a 多了 aggregation 卷积分支，新增风险不在 TensorRT parser，而在数学边界：

- depthwise 5x5 的 `groups=192` 语义不能按普通 dense conv 处理。
- grouped 1x1 的 `groups=12`，每组输入/输出通道为 16。
- `cat` 的输入顺序必须是 `[original qkv, aggregation output]`，不能反过来。
- P1b 输出边界是 `relu_linear_att` 的输出，**不包含**后续 `proj` Conv，也不包含 residual Add。

如果不先做 block-local 验证，端到端输出一旦不一致，很难判断错误来自 aggregation、cat、linear attention、proj 还是后续 segmentation head。

---

## 3. 目标模块

固定只验证两个真实 `stage2/context` LiteMLA 实例：

| 模块名 | 语义 |
|---|---|
| `backbone.stages.2.op_list.1.context_module.main` | stage2 第一个 LiteMLA context block |
| `backbone.stages.2.op_list.2.context_module.main` | stage2 第二个 LiteMLA context block |

索引口径：

- `backbone.stages.2` 是源码 / ONNX / NVTX 的 0-indexed 路径。
- 它对应模型语义中的 stage3 特征层。
- 每个 block 源码顺序是先 `context_module`，再 `local_module`；P1b 只覆盖 `context_module.main`，不覆盖后面的 MBConv local module。

---

## 4. Tensor 与权重契约

固定 Cityscapes `1024x2048` 输入、batch=1、FP32：

| 名称 | shape | 来源 |
|---|---:|---|
| `context_input` | `[1,64,64,128]` | LiteMLA 输入 `x` |
| `qkv` | `[1,192,64,128]` | `module.qkv(context_input)` |
| `aggregated_qkv` | `[1,192,64,128]` | `module.aggreg[0](qkv)` |
| `cat_qkv` | `[1,384,64,128]` | `torch.cat([qkv, aggregated_qkv], dim=1)` |
| `attention_out` | `[1,128,64,128]` | `module.relu_linear_att(cat_qkv).to(cat_qkv.dtype)` |

权重：

| 名称 | shape | 语义 |
|---|---:|---|
| `aggreg.0.0.weight` | `[192,1,5,5]` | depthwise 5x5，`groups=192` |
| `aggreg.0.1.weight` | `[192,16,1,1]` | grouped 1x1，`groups=12` |

当前模型配置中 aggregation 无 bias；验证脚本显式检查 bias 为 `None`。

---

## 5. Reference 捕获流程

脚本：

```text
phase3/scripts/capture_p1b_stage2_reference.py
```

执行顺序：

1. 加载 EfficientViT-Seg B0 Cityscapes 真实权重。
2. 构造固定输入，复用 Phase 1/2 的正式输入口径。
3. 设置 `eval()` + `torch.inference_mode()`。
4. 注册临时 forward hook 捕获两个 `context_module.main` 的输入 `context_input`。
5. 对每个目标模块单独计算：
   - `qkv = module.qkv(context_input)`
   - `aggregated_qkv = module.aggreg[0](qkv)`
   - `cat_qkv = torch.cat([qkv, aggregated_qkv], dim=1)`
   - `attention_out = module.relu_linear_att(cat_qkv).to(cat_qkv.dtype)`
6. 记录每个 tensor 的 shape、dtype、contiguous、sha256、min/max/mean/std。
7. 记录两个 aggregation weight 的 shape、dtype、sha256。
8. 写出 JSON：

```text
phase3/results/metrics/p1b_stage2_reference_capture.json
```

本地 tensor bundle：

```text
phase3/results/tensors/p1b_stage2_reference_capture.npz
```

该文件较大，不入 git；它是后续 block-level Plugin correctness 的输入来源。

---

## 6. Plugin 数值验证流程

脚本：

```text
phase3/scripts/validate_p1b_aggregation_attention_plugin.py
```

验证方式：

- 对 block1 / block2 分别构造 toy P1b ONNX。
- toy engine runtime input 只有 `qkv [1,192,64,128]`。
- depthwise / grouped pointwise weights 作为 initializer 固化在 engine 中。
- 输出直接是 `attention_out [1,128,64,128]`。
- 使用 TensorRT Plugin DLL 执行后，与 PyTorch reference 中的 `attention_out` 对齐。

选择 toy P1b engine 而不是完整 EfficientViT engine，是为了让失败定位更直接：如果输出不一致，只可能来自 aggregation、cat 或 attention，而不是 proj/head/后处理。

---

## 7. 数值指标与验收阈值

每个 block 至少记录：

| 指标 | 作用 |
|---|---|
| `max_abs_diff` | 捕获最坏点误差 |
| `mean_abs_diff` | 判断整体漂移 |
| `cosine_similarity` | 判断整体方向是否一致 |
| `allclose_pass` | 给出工程可接受的 pass/fail |
| `argmax_channel_agreement` | 中间特征的粗粒度结构一致性，仅作 sanity signal |

第一版 FP32 验收阈值：

```text
max_abs_diff <= 1e-3
mean_abs_diff <= 1e-5
cosine_similarity >= 0.99999
torch.allclose(atol=1e-3, rtol=1e-3) == true
```

说明：

- P1b 聚合中包含 Conv 和 MatMul，TensorRT / CUDA 实现顺序可能与 PyTorch 不完全一致，不能要求 bitwise equal。
- 当前阶段不验证 FP16；FP16/混合精度需要另立阈值。
- `argmax_channel_agreement` 对中间 feature 不等价于语义正确性，只作为快速 sanity signal。

---

## 8. Reference 捕获结果

Reference 捕获已通过：

| 项 | 结果 |
|---|---|
| Metadata | [`../results/metrics/p1b_stage2_reference_capture.json`](../results/metrics/p1b_stage2_reference_capture.json) |
| 目标模块 | `backbone.stages.2.op_list.1.context_module.main`、`backbone.stages.2.op_list.2.context_module.main` |
| P1b runtime input | `qkv [1,192,64,128]` |
| P1b output reference | `attention_out [1,128,64,128]` |
| depthwise weight | `[192,1,5,5]`，`groups=192` |
| pointwise weight | `[192,16,1,1]`，`groups=12` |
| projection sanity check | 两个 block 的 `module.proj(attention_out)` 均与原模块输出 `allclose(atol=1e-5, rtol=1e-5)` |

该结果证明 PyTorch block-level reference、权重布局和输出边界可复现。

---

## 9. Plugin correctness 结果

P1b 第一版 CUDA 数学路径已通过：

| 项 | block1 | block2 |
|---|---:|---:|
| `max_abs_diff` | `1.311302e-06` | `2.384186e-06` |
| `mean_abs_diff` | `9.504385e-08` | `7.170572e-08` |
| `cosine_similarity` | `0.9999999999996935` | `0.9999999999997492` |
| `allclose(atol=1e-3, rtol=1e-3)` | `true` | `true` |
| `argmax_channel_agreement` | `1.0` | `1.0` |

Metadata：

```text
phase3/results/metrics/p1b_aggregation_attention_plugin_validation.json
```

结论：

- 两个真实 `stage2/context` block 的 P1b Plugin 输出均与 PyTorch `attention_out` reference 对齐。
- 当前验证仍是 block-local toy/plugin correctness，不代表完整 EfficientViT P1b engine 端到端 correctness。
- 全部通过后，可以进入 P1b 真实 engine rebuild、correctness、latency 与 Nsight attribution。

---

## 10. 下一步实现顺序

1. 用当前第一版 P1b CUDA 数学路径重新构建真实 P1b patched engine。
2. 复用 Phase 2 / Phase 3 benchmark 口径做端到端 correctness。
3. correctness 通过后，再跑 latency 与 Nsight attribution。
4. 若 P1b 端到端收益不稳定，再回到 P1a 或拆分 aggregation-only / attention-only 对照实验。
