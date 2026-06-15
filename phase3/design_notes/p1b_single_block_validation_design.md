# P1b 单 Block 数值验证设计

> **关联阶段**：[`../README.md`](../README.md) Step 8.10
>
> **状态**：v0.2，验证口径已确认，PyTorch reference 捕获已通过并写入 [`../results/metrics/p1b_stage2_reference_capture.json`](../results/metrics/p1b_stage2_reference_capture.json)。本文只定义 P1b 单 block 验证口径，不宣称当前 P1b skeleton 已经数值正确。

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

当前 P1b skeleton 已经证明 TensorRT parser/build 可以接受这个边界，但 `enqueue()` 仍然只是 zero-fill 输出。因此下一步不能直接跑端到端 latency 或 correctness，而应先做**单 block 数值验证**：

1. 用真实 EfficientViT `stage2/context` 模块生成 PyTorch reference。
2. 固定两个目标 block 的输入、权重、aggregation 中间结果与最终 attention 输出。
3. 后续 P1b CUDA 实现完成后，用同一批捕获数据验证 Plugin 输出是否对齐 reference。

---

## 2. 为什么先做单 Block 验证

P1b 比 P1a 多了 aggregation 卷积分支，新增风险不在 TensorRT parser，而在数学边界：

- depthwise 5x5 的 `groups=192` 语义不能按普通 dense conv 处理。
- grouped 1x1 的 `groups=12`，每组输入/输出通道为 16。
- `cat` 的输入顺序必须是 `[original qkv, aggregation output]`，不能反过来。
- P1b 输出边界是 `relu_linear_att` 的输出，**不包含**后续 `proj` Conv，也不包含 residual Add。

因此 P1b 必须先在 block-local 层面证明：

```text
Plugin(qkv, depthwise_weight, pointwise_weight)
  ~= module.relu_linear_att(torch.cat([qkv, module.aggreg[0](qkv)], dim=1))
```

这一步通过后，才有资格进入真实 ONNX Plugin engine correctness / latency。

---

## 3. 目标模块

固定只验证两个真实 `stage2/context` LiteMLA 实例：

| 模块名 | 语义 |
|---|---|
| `backbone.stages.2.op_list.1.context_module.main` | stage2 第一个 LiteMLA context block |
| `backbone.stages.2.op_list.2.context_module.main` | stage2 第二个 LiteMLA context block |

注意索引口径：

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
| `aggreg.0.0.weight` | `[192,1,5,5]` | depthwise 5x5，`groups=192`，padding=2 |
| `aggreg.0.1.weight` | `[192,16,1,1]` | grouped 1x1，`groups=12` |

当前模型配置下 aggregation 无 bias；验证脚本应显式断言 bias 为 `None`。

---

## 5. Reference 捕获流程

第一版脚本已落盘为：

```text
phase3/scripts/capture_p1b_stage2_reference.py
```

执行顺序：

1. 加载 EfficientViT-Seg B0 Cityscapes 真实权重。
2. 构造固定输入，优先复用 Phase 1/2 的正式 dummy/image 输入口径。
3. 将模型置于 `eval()` + `torch.inference_mode()`。
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

这一步只验证 PyTorch reference 和 P1b 边界，不加载 P1b Plugin engine。

---

## 6. Plugin 数值验证流程

P1b CUDA 数学实现完成后，再新增：

```text
phase3/scripts/validate_p1b_aggregation_attention_plugin.py
```

推荐先使用 toy P1b engine，而不是完整 EfficientViT engine：

- toy engine runtime input 只有 `qkv [1,192,64,128]`；
- depthwise / grouped pointwise weights 作为 initializer 固化在 engine 中；
- 输出直接是 `attention_out [1,128,64,128]`；
- 验证失败时更容易定位是 aggregation、cat 还是 linear attention 出错。

验证输入来自 `p1b_stage2_reference_capture.json` 记录的两个 block。每个 block 应分别构建或选择对应权重的 toy engine，不能假设两个 block 权重共享。

---

## 7. 数值指标与验收阈值

每个 block 至少记录：

| 指标 | 作用 |
|---|---|
| `max_abs_diff` | 捕获最坏点误差 |
| `mean_abs_diff` | 判断整体漂移 |
| `cosine_similarity` | 判断整体方向是否一致 |
| `relaxed_allclose` | 给出工程可接受的 pass/fail |
| `argmax_channel_agreement` | 中间特征的粗粒度结构一致性，仅作参考 |

第一版 FP32 验收建议：

```text
max_abs_diff <= 1e-3
mean_abs_diff <= 1e-5
cosine_similarity >= 0.99999
relaxed_allclose(atol=1e-3, rtol=1e-3) == true
```

说明：

- P1b 聚合中包含 Conv 和 MatMul，TensorRT / CUDA 实现顺序可能与 PyTorch 不完全一致，不能要求 bitwise equal。
- 当前阶段不验证 FP16；FP16/混合精度需要另立阈值。
- `argmax_channel_agreement` 对中间 feature 不等价于语义正确性，只作为快速 sanity signal。

---

## 8. 成功标准

P1b 单 block 数值验证分两阶段验收：

### 8.1 Reference 捕获完成

必须满足：

- 两个目标模块都被找到。
- 两个模块的 `qkv / aggregated_qkv / cat_qkv / attention_out` shape 与本文契约一致。
- 两个模块的 aggregation 权重 shape 与 group 语义一致。
- JSON 中记录输入、权重和 reference 输出 hash。

### 8.2 Plugin correctness 完成

必须满足：

- block1 和 block2 均通过数值阈值。
- JSON 明确记录 Plugin DLL、toy engine、权重 hash、输入 hash。
- 若任一 block 失败，不能进入端到端 P1b engine benchmark。

---

## 9. 已知风险

| 风险 | 说明 | 应对 |
|---|---|---|
| skeleton zero-fill 被误用 | 当前 P1b engine 能 build，但输出必错 | 文档和脚本中显式标记 `skeleton_only`，未实现 CUDA 前禁止 benchmark |
| block2 输入依赖前序 block | 随机 block-local 输入不能代表真实运行分布 | reference 捕获优先从完整模型 forward 中取真实 `context_input` |
| 权重绑定错误 | block1/block2 权重 shape 相同但数值不同 | 每个 block 记录独立权重 hash，toy engine 不共享权重 |
| autocast 口径漂移 | `relu_linear_att` 源码禁用 autocast | 验证全程使用 FP32，后续 FP16 单独设计 |
| Windows 兼容噪声 | EfficientViT import 可能触发 Triton/W&B 兼容问题 | 复用 Phase 2/3 已有 compat 路径，不在验证脚本里引入新依赖 |

---

## 10. 下一步实现顺序

1. 新增 `capture_p1b_stage2_reference.py`，先只捕获 PyTorch reference。
2. 用 reference JSON 检查两个 block 的 tensor / weight contract。
3. 实现 P1b CUDA 数学路径。
4. 新增 `validate_p1b_aggregation_attention_plugin.py`，对 block1/block2 分别做 toy engine 数值验证。
5. 全部通过后，再进入 P1b 真实 engine correctness / latency / Nsight attribution。
