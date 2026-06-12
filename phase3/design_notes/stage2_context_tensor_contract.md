# Stage2 Context Tensor Contract

> **状态**：v0.1，Phase 3 Step 2 产物。
>
> **目的**：基于 ONNX graph、TensorRT EngineInspector 和 EfficientViT 源码，确定 `stage2/context` LiteMLA Plugin 候选的真实输入输出边界。本文只定义 tensor contract，不开始写 Plugin / CUDA kernel。

---

## 1. 证据来源

| 证据 | 文件 | 用途 |
|---|---|---|
| EfficientViT B0 backbone 配置 | [`../../efficientvit/models/efficientvit/backbone.py`](../../efficientvit/models/efficientvit/backbone.py) | 确认 `width_list=[8,16,32,64,128]`、`dim=16` |
| LiteMLA 源码 | [`../../efficientvit/models/nn/ops.py`](../../efficientvit/models/nn/ops.py) | 确认 `qkv -> aggregation -> cat -> relu_linear_att -> proj` 的语义顺序 |
| ONNX 导出结果 | [`../../phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx`](../../phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx) | 确认实际 node name 与 tensor shape |
| TensorRT EngineInspector | [`../../phase2/results/metrics/trt_engine_inspection_summary.md`](../../phase2/results/metrics/trt_engine_inspection_summary.md) | 确认 TensorRT 后 `stage2/context` 仍拆成多个 layers |
| TensorRT Nsight attribution | [`../../phase2/results/metrics/trt_nsys_attribution_summary.md`](../../phase2/results/metrics/trt_nsys_attribution_summary.md) | 确认 TensorRT 后 residual runtime 仍集中在 stage2 context 子路径 |

---

## 2. 命名与索引口径

本文中的 `stage2/context` 指 ONNX / TensorRT layer name 里的：

- `/backbone/stages.2/op_list.1/context_module/main`
- `/backbone/stages.2/op_list.2/context_module/main`

它们对应源码 `EfficientViTBlock` 中的两个 LiteMLA context block。需要注意两个索引口径：

- ONNX / ModuleList 使用 0-indexed `stages.2`。
- Backbone forward 输出字典使用 `stage1/stage2/stage3/stage4` 这种 1-indexed 命名；因此 ONNX `stages.2` 对应语义上的 backbone `stage3` 特征层。

源码顺序是先 `context_module`，再 `local_module`：

```python
x = self.context_module(x)
x = self.local_module(x)
```

因此 Phase 3 的 LiteMLA Plugin 目标应锁定 `context_module/main`，而不是同一 block 后面的 MBConv `local_module`。

---

## 3. 固定形状与模型参数

当前 contract 只覆盖 Phase 2 已构建的固定输入 TensorRT FP32 engine：

| 项 | 值 |
|---|---|
| 模型 | `efficientvit-seg-b0-cityscapes` |
| 输入分辨率 | `1x3x1024x2048` |
| layout | NCHW |
| dtype | FP32 |
| batch size | 1 |
| 目标 ONNX stage | `backbone.stages.2` |
| context 输入特征 | `[1,64,64,128]` |
| LiteMLA `dim` | 16 |
| 原始 qkv heads | `64 / 16 = 4` |
| aggregation scale | `(5,)` |
| cat 后 attention heads | 8 |

---

## 4. 单个 stage2/context block 的 tensor 链路

两个目标 block 的形状一致，区别只在输入 tensor name：

| block | context 输入 |
|---|---|
| `op_list.1` | `/backbone/stages.2/op_list.0/main/point_conv/conv/Conv_output_0` |
| `op_list.2` | `/backbone/stages.2/op_list.1/local_module/Add_output_0` |

核心链路如下：

| 段 | ONNX / 源码语义 | shape |
|---|---|---|
| context input | LiteMLA 输入 `x` | `[1,64,64,128]` |
| qkv Conv | `self.qkv(x)` | `[1,192,64,128]` |
| aggregation depthwise 5x5 | `aggreg.0.0(qkv)` | `[1,192,64,128]` |
| aggregation grouped 1x1 | `aggreg.0.1(...)` | `[1,192,64,128]` |
| cat | `torch.cat([qkv, aggreg(qkv)], dim=1)` | `[1,384,64,128]` |
| reshape | `(B, -1, 3*dim, H*W)` | `[1,8,48,8192]` |
| q / k / v split | split `48 -> 16+16+16` | each `[1,8,16,8192]` |
| ReLU(q), ReLU(k) | `kernel_func` | each `[1,8,16,8192]` |
| transpose k | `k.transpose(-1,-2)` | `[1,8,8192,16]` |
| pad v | `F.pad(v, (0,0,0,1), value=1)` | `[1,8,17,8192]` |
| matmul 1 | `vk = matmul(v_pad, k^T)` | `[1,8,17,16]` |
| matmul 2 | `out = matmul(vk, q)` | `[1,8,17,8192]` |
| normalize | `out[:,:,:-1] / (out[:,:,-1:] + eps)` | `[1,8,16,8192]` |
| reshape back | `(B, -1, H, W)` | `[1,128,64,128]` |
| cast | `.to(qkv.dtype)`，FP32 下等价 | `[1,128,64,128]` |
| proj Conv | `self.proj(out)` | `[1,64,64,128]` |
| residual Add | `context_module/Add` | `[1,64,64,128]` |

ONNX shape inference 对部分 `Pad/MatMul/Slice` 输出保留了 `unk__*` 符号，但根据 LiteMLA 源码和固定输入 shape，可以静态推出上表形状。

---

## 5. Plugin 候选边界

### 5.1 P1a MVP：`relu_linear_att-only`

**第一版优先候选。**

| 项 | contract |
|---|---|
| 替换范围 | `Concat_output_0 -> Cast_1_output_0` |
| 输入 | cat 后 qkv：`[1,384,64,128]` |
| 输出 | attention 输出：`[1,128,64,128]` |
| dtype | FP32 |
| 是否需要 Plugin 权重 | 不需要 |
| 后续接入 | 输出继续喂给现有 `proj/conv/Conv` |

选择理由：

- 边界最小，适合先验证 TensorRT Plugin Creator、engine build、runtime enqueue 和数值对齐。
- 直接覆盖 LiteMLA 的非标准线性注意力核心，而不是标准 Conv/MBConv 链。
- 不需要携带 qkv / aggregation / proj 权重，第一版工程复杂度最低。

局限：

- 不消除 qkv Conv、aggregation Conv 和 cat 的中间 tensor 落地。
- 端到端收益上限低于 P1b，但更适合作为 MVP。

### 5.2 P1a 对照 / fallback：`aggregation-only`

| 项 | contract |
|---|---|
| 替换范围 | `qkv/conv/Conv_output_0 -> aggreg.0/aggreg.0.1/Conv_output_0` |
| 输入 | qkv Conv 输出：`[1,192,64,128]` |
| 输出 | aggregation 分支输出：`[1,192,64,128]` |
| dtype | FP32 |
| 需要权重 | depthwise 5x5 `[192,1,5,5]`；grouped 1x1 `[192,16,1,1]` |

选择理由：

- 图边界清晰，TensorRT residual runtime 仍可见。
- 可作为 `relu_linear_att-only` 无法顺利 graph replacement 时的 fallback。

局限：

- 本质更接近标准 depthwise/group convolution，TensorRT/cuDNN 已有较成熟优化路径。
- 展示“非标准注意力 Plugin”的区分度低于 `relu_linear_att-only`。

### 5.3 P1b 主性能边界：`aggregation + cat + relu_linear_att`

| 项 | contract |
|---|---|
| 替换范围 | `qkv/conv/Conv_output_0 -> Cast_1_output_0` |
| 输入 | qkv Conv 输出：`[1,192,64,128]` |
| 输出 | attention 输出：`[1,128,64,128]` |
| dtype | FP32 |
| 需要权重 | aggregation depthwise/grouped conv 权重 |
| 后续接入 | 输出继续喂给现有 `proj/conv/Conv` |

选择理由：

- 覆盖 `aggregation`、`cat`、`relu_linear_att` 三段，贴合 Phase 1 Plan D 和 Phase 2 residual attribution 的主性能候选。
- 有机会减少 aggregation 输出、cat 输出、attention 输入之间的全局显存写回/读取。

风险：

- Graph 替换范围更大，Plugin 内部需要处理 aggregation 权重和 attention 计算。
- 共享内存/寄存器是否能承接中间结果需要 Phase 3 kernel 设计阶段再评估。

### 5.4 P1c fallback / 上限：整体 LiteMLA

| 项 | contract |
|---|---|
| 替换范围 | `context input -> proj/conv/Conv_output_0`，或进一步包括 residual Add |
| 输入 | context input：`[1,64,64,128]` |
| 输出 | main 输出或 residual 后输出：`[1,64,64,128]` |
| dtype | FP32 |
| 需要权重 | qkv / aggregation / proj 全部权重 |

整体 LiteMLA 融合空间最大，但实现、调试和数值验证成本最高，不作为第一版。

---

## 6. Step 2 结论

1. **第一版 MVP 选择 `relu_linear_att-only`**。它的真实 contract 是 `[1,384,64,128] -> [1,128,64,128]`，不需要 Plugin 权重，最适合先跑通 TensorRT Plugin 接入闭环。
2. **`aggregation-only` 保留为 fallback / 对照实验**。它的真实 contract 是 `[1,192,64,128] -> [1,192,64,128]`，但属于标准卷积类路径，展示区分度较低。
3. **主性能边界是 P1b：`aggregation + cat + relu_linear_att`**。它的真实 contract 是 `[1,192,64,128] -> [1,128,64,128]`，是 MVP 成功后的优先扩展方向。
4. **当前 contract 是固定 shape contract**。它只承诺 batch=1、Cityscapes `1024x2048`、FP32、TensorRT 8.6.1 engine；动态 shape / FP16 / batch>1 需要另立 contract。

