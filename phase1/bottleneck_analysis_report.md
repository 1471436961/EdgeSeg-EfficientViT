# Phase 1 性能瓶颈与融合机会分析报告

> **分析对象**：EfficientViT-Seg-B0 在 NVIDIA GeForce MX250 上的 PyTorch 原生推理路径。
>
> **报告目标**：建立 Phase 1 端到端性能基线，使用 Nsight Systems 将 CUDA kernel 耗时归因到模型结构，并为 Phase 3 TensorRT Plugin / 工程优化候选排序。
>
> **关键口径**：NVTX range 只提供结构边界，不是计时工具。本文所有组件耗时均来自 Nsight SQLite 中 CUDA runtime/kernel `correlationId` 归因统计，不直接使用 NVTX range 的 `end-start` 作为 GPU 耗时。
>
> **方法论复盘**：Phase 1 中被人工 review 纠正过的关键测量/归因/Plugin 候选取舍，见 [`design_notes/phase1_decision_corrections.md`](./design_notes/phase1_decision_corrections.md)。

---

## 1. 结论摘要

EfficientViT-Seg-B0 可以在 MX250 2GB 上以 Cityscapes 原生分辨率 `1024x2048` 跑通。干净 PyTorch baseline 如下：

| 指标 | 数值 |
|---|---:|
| Mean latency | 85.76 ms |
| P50 latency | 85.70 ms |
| P95 latency | 86.51 ms |
| P99 latency | 87.63 ms |
| Peak allocated memory | 1378 MB |
| Peak reserved memory | 1434 MB |

Phase 1 的核心结论必须分成两条线：

- **端到端耗时最大热点**：`stage0`，主要是早期高分辨率 MBConv / Conv 风格计算。
- **Plugin 展示价值最高主线**：`stage2/context` LiteMLA，因为它是论文核心非标准线性注意力模块，更难被 TensorRT 自动高效融合。

因此，报告中不能说“LiteMLA 是全模型最大瓶颈”。更准确的说法是：

> `stage0/head/stage2-local` 是重要工程热点，但主要由标准 Conv / MBConv 链构成；`stage2/context` LiteMLA 不是最大端到端热点，但它是最高区分度的 TensorRT Plugin 主线。

Plan D 进一步细化了 LiteMLA Plugin 边界：在 `stage2/context` LiteMLA 内部，`aggregation` 与 `relu_linear_att` 是两大主耗时，中间夹着 `cat`。因此 Phase 3 应比较三类 LiteMLA Plugin 边界：

1. **局部单段 Plugin**：`aggregation-only` 或 `relu_linear_att-only`。
2. **中段组合 Plugin**：`aggregation + cat + relu_linear_att`。
3. **整体 LiteMLA Plugin fallback**。

---

## 2. 实验设置

| 项目 | 数值 |
|---|---|
| 模型 | EfficientViT-Seg-B0，Cityscapes pretrained |
| 权重文件 | `phase1/weights/efficientvit_seg_b0_cityscapes.pt`（不入库，见 SHA256） |
| 权重 SHA256 | `923d6fdd5e93640cc0c2f3f213764f34e80b477cd98a6b294d870ea6df5acc50` |
| 输入图像 | [`data/city_asset_cityscapes_like.png`](./data/city_asset_cityscapes_like.png) |
| 输入 SHA256 | `34a663391ddeed9bbcc98c605d881fadbf7bb05ff02a8ffe4136d52599efc630` |
| 输入分辨率 | `1024x2048` |
| Batch size | 1 |
| Dtype | FP32 |
| GPU | NVIDIA GeForce MX250，Pascal `sm_61`，2GB VRAM |
| PyTorch | 2.4.1+cu124 |
| Nsight Systems | 2026.2.1 |
| Warmup / measure | 20 / 100 |
| 端到端计时 | CUDA Events |
| Nsight trace | `cuda,nvtx` |

Windows Nsight 说明：CPU sampling / context switch tracing 需要管理员权限，本阶段不作为主证据。Phase 1 归因主口径为 CUDA/NVTX 数据。

---

## 3. 测量方法

Phase 1 使用四档 profiling plan：

| Plan | 目的 | NVTX 范围 | 定量用途 |
|---|---|---|---|
| A | 干净 latency baseline | 无 NVTX | 端到端 latency / 显存锚点 |
| B | 全模型大区域归因 | `stem`, `stage0..3`, `head` | stage/head 级瓶颈排序 |
| C | 热点组件归因 | `stage0`, `stage2`, `head` 内部关键组件 | 解释热点区域内部构成 |
| D | stage2 LiteMLA 内部归因 | `qkv`, `aggregation`, `cat`, `relu_linear_att`, `proj` | 选择 Phase 3 LiteMLA Plugin 边界 |

解释规则：

- 端到端 latency 使用 JSON 中 CUDA Event 统计。
- 组件耗时使用 `analyze_nsys_attribution.py`，通过 runtime/kernel `correlationId` 将 CUDA kernel duration 归因到 NVTX range。
- 不把 NVTX range 的 `start/end` duration 当作 GPU 组件耗时。
- [`results/figures/`](./results/figures/) 中截图只作为可视化证据，定量结论以 JSON / attribution 表为准。

---

## 4. 端到端基线

| Run | Mean ms | P50 ms | P95 ms | P99 ms | Peak allocated MB | 说明 |
|---|---:|---:|---:|---:|---:|---|
| Plan A | 85.76 | 85.70 | 86.51 | 87.63 | 1378 | 干净 latency baseline |
| Plan B | 88.20 | 88.02 | 89.95 | 90.44 | 1378 | stage-level NVTX |
| Plan C | 90.22 | 89.98 | 93.22 | 93.60 | 1378 | hotspot component NVTX |
| Plan D | 88.04 | 87.51 | 89.44 | 89.61 | 见说明 | LiteMLA internal NVTX |

报告主 baseline 使用 Plan A。Plan B/C/D 用于归因，不作为最终 latency anchor。

显存说明：主显存证据采用 Plan A 的 `max_memory_allocated_mb` 与 `max_memory_reserved_mb`。Plan D 的 memory 字段不作为主显存结论，因为 Plan D patch/sanity 路径会改变 PyTorch 对 peak memory 的记录方式。

---

## 5. Plan B：Stage 级归因

Plan B 回答：**全模型哪个大区域 CUDA kernel 耗时最多？**

| 排名 | 区域 | Avg kernel ms / iter | Share of forward mean |
|---:|---|---:|---:|
| 1 | `stage0` | 24.528 | 27.81% |
| 2 | `stage2` | 18.458 | 20.93% |
| 3 | `stage1` | 12.223 | 13.86% |
| 4 | `head` | 10.882 | 12.34% |
| 5 | `stage3` | 10.403 | 11.80% |
| 6 | `stem` | 10.324 | 11.71% |

证据：

- Attribution 表：[planB_nsys_attribution_summary.md](./results/metrics/planB_nsys_attribution_summary.md)
- Timeline 总览：[planB_timeline_overview.png](./results/figures/planB_timeline_overview.png)
- 单次 forward NVTX：[planB_single_forward_nvtx.png](./results/figures/planB_single_forward_nvtx.png)

结论：

`stage0` 是当前 PyTorch 路径最大 GPU hotspot，`stage2` 第二。因此，Phase 1 不能把 LiteMLA 叙述为“最大瓶颈”。正确叙事是：LiteMLA 是高区分度 Plugin 主线，而 stage0/head 是重要工程热点。

---

## 6. Plan C：热点组件归因

Plan C 回答：**在选中的热点区域里，具体哪个组件解释了耗时？**

| 排名 | 组件 | Avg kernel ms / iter | Share of forward mean |
|---:|---|---:|---:|
| 1 | `stage0/block0/main` | 12.294 | 13.63% |
| 2 | `stage0/block1/main` | 12.216 | 13.54% |
| 3 | `head/middle` | 6.399 | 7.09% |
| 4 | `stage2/block1/context` | 4.992 | 5.53% |
| 5 | `stage2/block2/context` | 4.989 | 5.53% |
| 6 | `stage2/downsample` | 3.016 | 3.34% |
| 7 | `stage2/block2/local` | 3.006 | 3.33% |
| 8 | `stage2/block1/local` | 3.005 | 3.33% |
| 9 | `head/output_segout` | 2.577 | 2.86% |

证据：

- Attribution 表：[planC_nsys_attribution_summary.md](./results/metrics/planC_nsys_attribution_summary.md)
- 截图：
  - [planC_timeline_overview.png](./results/figures/planC_timeline_overview.png)
  - [planC_stage0_components.png](./results/figures/planC_stage0_components.png)
  - [planC_stage2_components.png](./results/figures/planC_stage2_components.png)
  - [planC_head_components.png](./results/figures/planC_head_components.png)

关键解释：

1. `stage0/block0/main` 和 `stage0/block1/main` 位居前二，主要原因是它们作用在高分辨率 feature map 上，且主体是 MBConv / Conv 风格计算。
2. `head/middle` 耗时明显，但它同样是 MBConv 风格模块，属于标准算子链工程优化候选。
3. `stage2` 内部，`context` 比 `local` 更耗时：
   - `stage2/block1/context`: 4.992 ms
   - `stage2/block1/local`: 3.005 ms
   - `stage2/block2/context`: 4.989 ms
   - `stage2/block2/local`: 3.006 ms

这正是进入 Plan D 的依据：`stage2/context` 是 LiteMLA 所在路径，并且在 stage2 内比 local MBConv 更重。

---

## 7. Plan D：stage2 LiteMLA 内部归因

Plan D 回答：**stage2 LiteMLA 内部哪个子路径更适合成为 TensorRT Plugin 边界？**

| 排名 | LiteMLA internal range | Avg kernel ms / iter | Share of forward mean |
|---:|---|---:|---:|
| 1 | `stage2/block1/litemla/aggregation` | 1.840 | 2.09% |
| 2 | `stage2/block2/litemla/aggregation` | 1.840 | 2.09% |
| 3 | `stage2/block2/litemla/relu_linear_att` | 1.805 | 2.05% |
| 4 | `stage2/block1/litemla/relu_linear_att` | 1.802 | 2.05% |
| 5 | `stage2/block1/litemla/cat` | 0.528 | 0.60% |
| 6 | `stage2/block2/litemla/cat` | 0.528 | 0.60% |
| 7 | `stage2/block1/litemla/qkv` | 0.296 | 0.34% |
| 8 | `stage2/block2/litemla/qkv` | 0.289 | 0.33% |
| 9 | `stage2/block1/litemla/proj` | 0.272 | 0.31% |
| 10 | `stage2/block2/litemla/proj` | 0.272 | 0.31% |

按候选边界聚合：

| 候选边界 | 包含范围 | 约 ms / iter | 约占 Plan D attributed time | 约占 forward mean |
|---|---|---:|---:|---:|
| `aggregation-only` | 两个 block 的 aggregation | 3.680 | 38.85% | 4.18% |
| `relu_linear_att-only` | 两个 block 的 linear attention | 3.607 | 38.08% | 4.10% |
| `aggregation + cat + relu_linear_att` | 两个 aggregation + 两个 cat + 两个 linear attention | 8.343 | 88.09% | 9.48% |
| 整体 Plan D LiteMLA internal ranges | 全部 Plan D ranges | 9.471 | 100.00% | 10.76% |

证据：

- Attribution 表：[planD_nsys_attribution_summary.md](./results/metrics/planD_nsys_attribution_summary.md)
- 截图：
  - [planD_timeline_overview.png](./results/figures/planD_timeline_overview.png)
  - [planD_litemla_aggregation_components.png](./results/figures/planD_litemla_aggregation_components.png)
  - [planD_litemla_relu_linear_att_components.png](./results/figures/planD_litemla_relu_linear_att_components.png)

结论：

Plan D 不支持“只看 `relu_linear_att` 就够了”的结论。`aggregation` 与 `relu_linear_att` 几乎并列主耗时，中间还有 `cat`。因此 Phase 3 应比较三类 LiteMLA Plugin 边界：

1. 局部单段 Plugin：`aggregation-only` 或 `relu_linear_att-only`。
2. 中段组合 Plugin：`aggregation + cat + relu_linear_att`。
3. 整体 LiteMLA Plugin fallback。

---

## 8. 主要瓶颈结论

### 8.1 `stage0` 是当前 PyTorch 最大端到端热点

`stage0` 占 24.528 ms / iter，占 Plan B forward mean 的 27.81%。这是当前 PyTorch profile 下最大的 GPU hotspot。

但 `stage0` 主要是高分辨率 feature map 上的标准 Conv / MBConv 风格计算。它的耗时真实存在，但不能直接推出“优先手写 Plugin”。TensorRT/cuDNN 可能已经能较好处理这类标准算子链。

### 8.2 `stage2` 是性能热点与 Plugin 价值之间的关键桥梁

Plan B 显示 `stage2` 位列全模型第二。Plan C 显示 `stage2/context` 比 `stage2/local` 更重，而 `context` 是 LiteMLA 所在路径。这使得 `stage2/context` 成为最有数据支撑的高区分度 Plugin 候选。

### 8.3 `head` 重要，但主要是标准算子链工程候选

`head/middle` 在 Plan C 中为 6.399 ms / iter，是明显热点。但它是 MBConv 风格模块，预计更容易被 TensorRT/cuDNN 优化。Head 中还包含 resize/add 行为，尤其 bicubic upsample 可能影响 ONNX/TensorRT 导出，这是 Phase 2 需要单独处理的部署风险。

### 8.4 LiteMLA 不是最大瓶颈，但仍是最佳 Plugin 叙事主线

LiteMLA 的价值不在于“当前 profile 最大耗时”，而在于：

- 它是论文核心非标准线性注意力模块。
- 它包含 reshape / ReLU(Q/K) / padding / 两次 MatMul / 归一化除法等自定义算子友好的结构。
- 它比 Conv / MBConv 链更难被 TensorRT 自动融合。
- 它更能展示 CUDA kernel 设计与 TensorRT Plugin 集成能力。

### 8.5 `aggregation + cat + relu_linear_att` 是最值得评估的 LiteMLA 组合边界

Plan D 显示 `aggregation` 与 `relu_linear_att` 都重，且中间存在 `cat`。组合 Plugin 有机会减少中间 tensor 落地、拼接、再读取带来的 memory traffic，也可能减少 kernel launch 数量。

这不等于已经证明一定能加速，但它是当前数据最支持的性能导向 LiteMLA Plugin 边界。

---

## 9. Phase 3 优化候选排序

### P1：stage2 LiteMLA Plugin 主线（Phase 3 已扩展到 stage2+stage3）

这是 Phase 3 默认的高区分度主线。

> Phase 3 回填：最终主交付线采用 P1a `relu_linear_att-only`，覆盖范围从 Phase 1 候选中的 stage2 扩展到 stage2+stage3 四个 LiteMLA context block；P1b/P1mix 已作为消融保留。

#### P1a：局部单段 Plugin

候选：

- `relu_linear_att-only`
- `aggregation-only`

推荐 MVP：`relu_linear_att-only`。

理由：

- 它最能体现“非标准线性注意力”的自定义 CUDA Plugin 能力。
- 它直接对应论文机制：`ReLU(Q/K)`、`(V·K^T)·Q`、末尾补 1 行归一化 trick。
- 边界相对清晰，适合作为第一版 TensorRT Plugin MVP。

注意：

`aggregation-only` 实测耗时与 `relu_linear_att-only` 接近，但它主要由卷积分支构成，可能被 TensorRT/cuDNN 标准优化覆盖。是否值得单独做 Plugin，需要等 Phase 2 TensorRT baseline 后再判断。

#### P1b：中段组合 Plugin

候选：

```text
aggregation + cat + relu_linear_att
```

这是 Plan D 最支持的性能导向 LiteMLA Plugin 边界。

理由：

- `aggregation` 与 `relu_linear_att` 共同构成 Plan D 大部分 attributed time。
- `cat` 位于两者之间，引入中间张量拼接与 memory traffic。
- 组合边界可能减少 kernel launch 和中间读写。

风险：

实现复杂度高于 `relu_linear_att-only`，因为它跨越卷积分支聚合与注意力计算。建议先做 MVP 或小型 prototype，再决定是否投入完整组合 Plugin。

#### P1c：整体 LiteMLA Plugin fallback / 上限方案

候选：

```text
qkv + aggregation + cat + relu_linear_att + proj
```

理由：

- 融合空间最大。
- 理论上可减少更多中间 tensor movement。

风险：

- 实现复杂度最高。
- 权重组织、TensorRT 接入、调试与数值对齐更难。
- FP16/FP32 策略更复杂，尤其 `relu_linear_att` 原始路径禁用 autocast。

因此，整体 LiteMLA 应作为 fallback / stretch goal，而不是第一版实现目标。

### P2：标准算子链工程优化候选

候选：

- `stage0` early MBConv / Conv chains。
- `head/middle`。
- `stage2/local` MBConv。

理由：

- 它们是重要 PyTorch hotspot。
- 端到端收益潜力可能大于 LiteMLA。
- 但它们主要由标准 Conv / MBConv 链构成，更可能被 TensorRT/cuDNN 自动优化。

决策规则：

Phase 2 TensorRT baseline 前，不建议直接为 P2 手写 Plugin。应先看 TensorRT 后这些区域是否仍是残余瓶颈。

### P3：低优先级探索项

候选：

- 更大 EfficientViT 变体上的多尺度 QKV aggregation。
- Head resize 替代与 bicubic/bilinear 精度折中。
- INT8 / mixed precision 探索。

这些方向可以增强项目系统性，但不应挤占 P1 Plugin 主线。

---

## 10. 粗略加速空间估算

下表不是最终性能承诺，只是基于 PyTorch attributed kernel time 的规划级估算。

| 候选 | 当前 attributed ms / iter | 若快 25% | 若快 50% | 若快 75% |
|---|---:|---:|---:|---:|
| `relu_linear_att-only` | 3.607 | save 0.90 ms | save 1.80 ms | save 2.71 ms |
| `aggregation-only` | 3.680 | save 0.92 ms | save 1.84 ms | save 2.76 ms |
| `aggregation + cat + relu_linear_att` | 8.343 | save 2.09 ms | save 4.17 ms | save 6.26 ms |
| 整体 Plan D LiteMLA internal ranges | 9.471 | save 2.37 ms | save 4.74 ms | save 7.10 ms |
| `stage0` region | 24.528 | save 6.13 ms | save 12.26 ms | save 18.40 ms |

解释：

- `stage0` 的 PyTorch 端到端收益上限最高，但 Plugin 展示价值低于 LiteMLA。
- `relu_linear_att-only` 直接收益较小，但展示价值最高，适合作为 MVP。
- `aggregation + cat + relu_linear_att` 是 LiteMLA 相关性与可观收益之间更好的折中。

---

## 11. 风险与解释边界

1. **PyTorch hotspot 不等于 TensorRT hotspot**

   TensorRT 可能较好融合 Conv/BN/activation 链。Phase 2 必须重新 profile TensorRT engine，再决定是否投入 P2。

2. **Phase 1 不测 mIoU**

   固定输入图只用于 latency/profiling。精度对齐属于 Phase 2 和 Phase 3。

3. **NVTX duration 不是组件耗时**

   定量归因使用 sqlite correlation 统计，不使用 NVTX range 的 raw duration。

4. **CPU enqueue / OS 调度不是本报告的主证据**

   当前 Nsight 运行没有启用管理员权限下的 CPU sampling、CPU context switch 与 WDDM trace，因此本报告不声称完全排除 Python CPU enqueue、多线程调度或 Windows WDDM 对 CUDA launch cadence 的轻微影响。现有 timeline 与 latency 分布未显示这些因素构成主导瓶颈，但若 Phase 2/3 出现明显 GPU 空洞或 launch 间隙，应使用管理员权限重新采集 CPU/WDDM trace。

5. **dataloader / preprocessing 不属于当前 latency 口径**

   Phase 1 使用固定单张输入图，图像读取、预处理、hash、MACs 等步骤均在 warmup/measure 之外完成。当前 latency 只覆盖 `model(x)` 推理本体，因此可以排除 dataloader/preprocessing 拖慢本报告中的推理 latency。若后续评估真实视频流或数据集 pipeline，需要另测 end-to-end pipeline latency。

6. **Plan D 只覆盖 stage2**

   Plan D 只分析 Plan C 选出的两个 `stage2/context` LiteMLA，不声称该比例适用于所有模型变体。

7. **FP16 策略未定**

   上游 `relu_linear_att` 禁用 autocast，说明这段对数值稳定性敏感。Phase 3 Plugin 需要明确内部保留 FP32，还是设计稳定 FP16 路径。

8. **Bicubic upsample 影响 Phase 2**

   SegHead 使用 bicubic upsample，TensorRT 可能不原生支持。该问题应在 ONNX/TensorRT 导出阶段单独处理。

---

## 12. 下一步

1. 启动 Phase 2 TensorRT baseline。
2. ONNX 导出时固定 `1024x2048` 输入，避免 LiteMLA shape-adaptive 分支歧义。
3. 构建 TensorRT FP32 / FP16 engine 并重新 profile。
4. 观察 P2 标准算子链热点在 TensorRT 后是否仍然存在。
5. Phase 3 先以 `relu_linear_att-only` 做 MVP，同时评估 `aggregation + cat + relu_linear_att` 是否适合作为主性能边界。

---

## 附录 A：源数据与证据

Metrics:

- [baseline_b0_cityscapes_1024x2048_levelA_latency_formal_v1.json](./results/metrics/baseline_b0_cityscapes_1024x2048_levelA_latency_formal_v1.json)
- [baseline_b0_cityscapes_1024x2048_levelB_latency_nsys.json](./results/metrics/baseline_b0_cityscapes_1024x2048_levelB_latency_nsys.json)
- [baseline_b0_cityscapes_1024x2048_levelC_latency_nsys.json](./results/metrics/baseline_b0_cityscapes_1024x2048_levelC_latency_nsys.json)
- [baseline_b0_cityscapes_1024x2048_levelD_latency_nsys.json](./results/metrics/baseline_b0_cityscapes_1024x2048_levelD_latency_nsys.json)
- [planB_nsys_attribution_summary.md](./results/metrics/planB_nsys_attribution_summary.md)
- [planC_nsys_attribution_summary.md](./results/metrics/planC_nsys_attribution_summary.md)
- [planD_nsys_attribution_summary.md](./results/metrics/planD_nsys_attribution_summary.md)

Figures:

- [planB_timeline_overview.png](./results/figures/planB_timeline_overview.png)
- [planB_single_forward_nvtx.png](./results/figures/planB_single_forward_nvtx.png)
- [planC_timeline_overview.png](./results/figures/planC_timeline_overview.png)
- [planC_stage0_components.png](./results/figures/planC_stage0_components.png)
- [planC_stage2_components.png](./results/figures/planC_stage2_components.png)
- [planC_head_components.png](./results/figures/planC_head_components.png)
- [planD_timeline_overview.png](./results/figures/planD_timeline_overview.png)
- [planD_litemla_aggregation_components.png](./results/figures/planD_litemla_aggregation_components.png)
- [planD_litemla_relu_linear_att_components.png](./results/figures/planD_litemla_relu_linear_att_components.png)

Design and architecture notes:

- [architecture_analysis.md](./architecture_analysis.md)
- [baseline_inference_design.md](./design_notes/baseline_inference_design.md)
