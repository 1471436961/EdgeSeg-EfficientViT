# EfficientViT-Seg-B0 架构精读报告

> **目的**：在动笔写 `baseline_inference.py` 与 Nsight 剖析脚本之前，先把模型结构、关键算子序列、张量形状 **完全摸清**。本报告直接决定：
>
> 1. NVTX range 应该插在**哪些层级 / 哪些算子序列**上；
> 2. 阶段三 TensorRT Plugin 的**候选融合目标**是什么；
> 3. 哪些"想当然"的传统 Transformer 假设在 EfficientViT 上**根本不成立**。
>
> **源码版本**：`mit-han-lab/efficientvit @ master`（本地 commit `de7d773`）。
> **撰写日期**：2026-05-26。
> **Phase 1 实测回填**：本文最初生成于源码精读阶段；2026-06-06 根据 Plan B/C/D Nsight attribution 结果做口径修订。静态结构事实以本文为准，性能热点与 Phase 3 Plugin 候选排序以 [`bottleneck_analysis_report.md`](bottleneck_analysis_report.md) 为准。

---

## 0. 关键结论先行（TL;DR）

| # | 结论 | 对后续阶段的影响 |
|---|------|------|
| 1 | **EfficientViT 不是传统 Transformer**：没有 LayerNorm、没有 Softmax、没有 Patch Embedding、没有位置编码。 | V3.0 文档中"LayerNorm+残差""MatMul+Softmax+Scale"的融合假设 **需要修正**。 |
| 2 | **核心创新 = LiteMLA（轻量多尺度线性注意力）**：`ReLU(Q), ReLU(K), V` 后做 `(V·K^T)·Q` 的线性顺序乘法，复杂度 O(N·d²) 而非 O(N²·d)。 | 真正的融合机会是 `Conv1x1(QKV) → 多尺度 DWConv → ReLU → MatMul → MatMul → 归一化除法` 这一序列。 |
| 3 | **归一化全部用 BN2d**（而非 LN）。 | TRT 部署时 **BN 直接折叠进 Conv**，几乎零开销；**不必单独融合 LN**。 |
| 4 | **B0 注意力只在语义 stage3/stage4 出现**：两阶段各 2 个 `EfficientViTBlock`，共 4 个 LiteMLA 实例。 | Nsight attribution 已验证：注意力/context 不是全模型最大瓶颈；最大热点在早期高分辨率代码/NVTX `stage0`，其次是代码/NVTX `stage2`，`head` 也有明确耗时。Plan D 第一版只细拆语义 stage3（代码/NVTX `stage2`）中的 2 个 LiteMLA。 |
| 5 | **B0 在 stage4 仅 128 通道，dim=16**，多尺度只有一个 5×5 DW 分支。 | LiteMLA Plugin 的定位应是**高区分度非标准算子展示**，而不是最大端到端收益点；收益预期要保守估算。 |
| 6 | LiteMLA 内部含一个 **形状自适应分支**：`H*W > dim` 走线性注意力，否则走二次注意力。 | TRT 部署时必须**冻结输入分辨率**，否则 Plugin 行为无法静态确定。 |

---

## 1. 顶层结构

### 1.1 `EfficientViTSeg`（`efficientvit/models/efficientvit/seg.py:107`）

```
input (B, 3, H, W)
   │
   ▼
backbone (EfficientViTBackbone)  ──→ feed_dict = {
   │                                    "input_stem": (B, 8,   H/2,  W/2),
   │                                    "stage1":     (B, 16,  H/4,  W/4),
   │                                    "stage2":     (B, 32,  H/8,  W/8),    ← 进 head
   │                                    "stage3":     (B, 64,  H/16, W/16),   ← 进 head
   │                                    "stage4":     (B, 128, H/32, W/32),   ← 进 head
   │                                    "stage_final":(B, 128, H/32, W/32),
   │                                  }
   ▼
head (SegHead, DAG 结构)         ──→ "segout": (B, 19, H/8, W/8)
   │
   ▼
最终通常由外部 dataloader/eval 把 segout 双线性 upsample 回 (H, W)
```

> ⚠️ 注意 `EfficientViTSeg.forward` 只返回 `feed_dict["segout"]`，**输出空间分辨率是 H/8**（不是原图）。对 Cityscapes 1024×2048 输入，segout 是 128×256。Nsight 测延迟时必须明确这一点，否则 mIoU 对比会错位。

### 1.2 `EfficientViTBackbone`（`backbone.py:33`）

5 个阶段（input_stem + stage1~4），每阶段下采样 2 倍：

| 阶段 | B0 通道 | B0 深度 | 主算子 | 备注 |
|------|---------|---------|--------|------|
| `input_stem` | 8 | 1 个 DSConv + 残差 | `DSConv` | stride=2 |
| `stage1` | 16 | 2 | `MBConv (expand=4)` + 残差 | stride=2 |
| `stage2` | 32 | 2 | `MBConv (expand=4)` + 残差 | stride=2 |
| `stage3` | 64 | 1 `MBConv` (下采样) + 2 `EfficientViTBlock` | **MBConv + LiteMLA** | stride=2 |
| `stage4` | 128 | 1 `MBConv` (下采样) + 2 `EfficientViTBlock` | **MBConv + LiteMLA** | stride=2 |

注意力**只**在语义 stage3 / stage4 出现。前 3 个语义阶段是纯 CNN（MobileNetV3 风格）。

> **阶段命名口径**：本文结构表里的 `stage1~4` 是模型语义阶段；Phase 1 NVTX / Nsight 中的 `stage0~3` 是 `backbone.stages` 的 `ModuleList` 索引。映射为：语义 `stage1` = 代码/NVTX `stage0`，语义 `stage2` = 代码/NVTX `stage1`，语义 `stage3` = 代码/NVTX `stage2`，语义 `stage4` = 代码/NVTX `stage3`。

### 1.3 `SegHead`（`seg.py:30`）— DAG 结构

B0 head 配置（`seg.py:120-142`）：

```
inputs:
  stage4 (128ch, stride=32) ── 1×1 Conv→32ch ── Upsample×4 ──┐
  stage3 ( 64ch, stride=16) ── 1×1 Conv→32ch ── Upsample×2 ──┤── add ──→ (32ch, stride=8)
  stage2 ( 32ch, stride= 8) ── 1×1 Conv→32ch ─────────────────┘

middle:  head_depth=1 个 MBConv(expand=4) + 残差        (32ch, stride=8)

outputs (segout):
  1×1 Conv → 128ch (final_expand=4) → 1×1 Conv → 19ch   (19ch, stride=8)
```

**关键观察**：
- Head 里 **3 路上采样到同一分辨率再相加**（FPN 风格，但加法而非 concat）。
- Upsample 默认 **bicubic**（`ops.py:84`），**TRT 不直接支持 bicubic，部署时大概率要降级为 bilinear**——这是一个潜在的精度对齐风险点。
- `UpSampleLayer.forward` 显式 `@torch.autocast(enabled=False)`，**强制 FP32 跑插值**。这意味着 FP16/AMP 或 TRT FP16 部署时可能出现 FP32 fallback / dtype cast 开销；Phase 1 当前 FP32 profile 不把这里的 cast 当作已验证瓶颈。

---

## 2. 核心模块逐个拆解

### 2.1 `DSConv` —— Depthwise Separable Conv（`ops.py:270`）

```
x ─→ DWConv(k=3, groups=C, BN, ReLU6) ─→ PWConv(1×1, BN) ─→ y
```
- 两个 ConvLayer 串联，BN 紧跟 Conv。
- **TRT 友好**：BN 必然能被折叠，最终就是两个 Conv kernel。

### 2.2 `MBConv` —— MobileNetV2 倒残差（`ops.py:312`）

```
x ─→ PWConv(1×1, expand→6×, BN, ReLU6)
   ─→ DWConv(3×3, BN, ReLU6)
   ─→ PWConv(1×1, project, BN)
   ─→ y
（外层套 ResidualBlock 实现 skip）
```
- **B0 backbone 里 expand=4**，`EfficientViTBlock.local_module` 里 expand=4，head middle 里 expand=4。
- **算子序列固定为 `1×1 Conv → 3×3 DW Conv → 1×1 Conv`**。这是 stage1/stage2 的主算力消耗点。
- 在 EfficientViTBlock 内的 MBConv，`use_bias=(True, True, False), norm=(None, None, "bn2d")`：**前两层没有 BN！只在最后一层做 BN**。这是一个反直觉的细节，专门为了"减少归一化次数"。

### 2.3 `LiteMLA` —— **本项目的灵魂**（`ops.py:518`）

#### 2.3.1 模块定义（B0 stage3 实例：`in_channels=64, dim=16, heads_ratio=1.0, scales=(5,)`）

```python
heads      = 64 // 16 * 1.0 = 4
total_dim  = 4 * 16 = 64

qkv:    Conv1x1, 64 → 3*64=192            (BN=None, act=None)
aggreg: ModuleList([
    Sequential(
        Conv5x5 DW (groups=192),          # depthwise 多尺度聚合
        Conv1x1  (groups=3*heads=12),     # head-wise pointwise
    ),
])  # B0 scales=(5,) 只有一个分支
kernel_func: ReLU
proj:   Conv1x1, 64*(1+1)=128 → 64        (BN=BN2d, act=None)
```

> stage4 实例：`in_channels=128, dim=16, heads=8, total_dim=128`，qkv 输出 384ch，proj 输入 256ch。

#### 2.3.2 forward 数据流（**核心算子序列**）

```
x (B, C, H, W)
   │
   │ ① qkv = Conv1x1(x)             →  (B, 3*total_dim, H, W)
   │
   │ ② 多尺度聚合：
   │    multi_scale_qkv = [qkv]
   │    for each scale in scales:        # B0 只 1 个
   │       msqkv = Conv5x5_DW(qkv)       # (B, 3*total_dim, H, W)
   │       msqkv = Conv1x1_grouped(msqkv)# (B, 3*total_dim, H, W), groups=3*heads
   │       multi_scale_qkv.append(msqkv)
   │    qkv = concat(multi_scale_qkv, dim=1)   # (B, 3*total_dim*(1+S), H, W)
   │
   │ ③ reshape →  (B, heads*(1+S), 3*dim, H*W)
   │    split   →  Q, K, V  各 (B, heads*(1+S), dim, H*W)
   │
   │ ④ Q' = ReLU(Q),   K' = ReLU(K)        ← 注意：没有 softmax！
   │
   │ ⑤ V_pad = pad(V, 末尾补 1 行)         shape (B, heads*(1+S), dim+1, H*W)
   │    K_T  = K'.transpose(-1,-2)         shape (B, heads*(1+S), H*W, dim)
   │
   │ ⑥ 线性注意力（H*W > dim 时走这条，2GB MX250 + 1024×2048 输入下必走此路）:
   │    VK  = V_pad @ K_T                  shape (B, heads*(1+S), dim+1, dim)
   │    out = VK    @ Q'                   shape (B, heads*(1+S), dim+1, H*W)
   │    out = out[:, :, :-1] / (out[:, :, -1:] + eps)   ← 归一化技巧（替代 softmax）
   │
   │ ⑦ reshape →  (B, total_dim*(1+S), H, W)
   │
   │ ⑧ y = Conv1x1(out) + BN           ← proj
```

#### 2.3.3 与传统 Transformer 注意力的对照表

| 维度 | 传统 ViT MHSA | EfficientViT LiteMLA |
|------|---------------|----------------------|
| 输入格式 | (B, N, C) tokens | (B, C, H, W) 特征图 |
| QKV 生成 | `nn.Linear` | `nn.Conv2d 1×1` |
| 多尺度 | 无 | **DWConv 5×5 + grouped 1×1** 额外分支 |
| Q/K 激活 | Softmax(Q·K^T/√d) | **ReLU(Q), ReLU(K)** |
| 乘法顺序 | `(Q·K^T)·V`，O(N²·d) | **`(V·K^T)·Q`，O(N·d²)** |
| 归一化 | softmax 自然归一 | **手工 `out / sum(K) trick`**：给 V 末尾补 1 行，乘出的最后一行恰好是 sum(K')，最后做逐元素除法 |
| 归一化层 | LayerNorm × 2 | **完全没有** LayerNorm，只在 proj 末尾接 BN2d |
| 位置编码 | 显式 PE | **无**（隐式由 5×5 DW 卷积提供局部位置感） |
| FFN | 后接 MLP | 后接 **MBConv**（卷积 FFN） |

> **这就是为什么 V3.0 文档里"LayerNorm+残差"的融合目标在 EfficientViT 上不存在。** 真正应该融合的是 ④⑤⑥ 这一段。

#### 2.3.4 FP16/autocast 行为（⚠️ 部署关键）

`relu_linear_att` 和 `relu_quadratic_att` 都标了 `@torch.autocast(enabled=False)`，并在内部把 FP16 强转 FP32 计算：

- 在 FP16/AMP 推理时，整个 LiteMLA 注意力计算**实际上是 FP32 的**；
- 这是出于"线性注意力数值稳定性差，eps trick 容易出 NaN"的考虑；
- **对 TRT FP16 部署**：要么把这部分手动用稳定的算法重写成 FP16，要么 Plugin 内部继续走 FP32 内核——这是阶段三 Plugin 设计的一个核心折中点；
- 对 Nsight 时间线：在 FP16/AMP 或 TRT FP16 profile 中，应重点检查注意力区段是否出现 `dtype cast` / FP32 fallback；Phase 1 FP32 profile 不把 cast 作为已验证开销。

### 2.4 `EfficientViTBlock`（`ops.py:671`）

```
x ─→ Residual( LiteMLA )        # context module
   ─→ Residual( MBConv expand=4 )   # local module (卷积 FFN)
   ─→ y
```
**等价于一个 Transformer Block，但把 attention 换成 LiteMLA，把 MLP 换成 MBConv。**

---

## 3. B0 完整流图（Cityscapes, 输入 1024×2048）

```
input (1, 3, 1024, 2048)
  │
  │ input_stem:  Conv3x3/2 + DSConv残差        →  (1,   8,  512, 1024)
  │
  │ stage1:      2× MBConv(expand=4) + 残差   →  (1,  16,  256,  512)   首块 stride=2
  │
  │ stage2:      2× MBConv(expand=4) + 残差   →  (1,  32,  128,  256)   ← 进 head
  │
  │ stage3:
  │   MBConv (fewer_norm, stride=2)             →  (1,  64,   64,  128)
  │   EfficientViTBlock × 2:
  │       LiteMLA(C=64,  dim=16, heads=4) + 残差
  │       MBConv(expand=4)                + 残差   →  (1,  64,   64,  128)   ← 进 head
  │
  │ stage4:
  │   MBConv (fewer_norm, stride=2)             →  (1, 128,   32,   64)
  │   EfficientViTBlock × 2:
  │       LiteMLA(C=128, dim=16, heads=8) + 残差
  │       MBConv(expand=4)                + 残差   →  (1, 128,   32,   64)   ← 进 head
  │
  ▼
SegHead (DAG, head_width=32, head_stride=8):
  in[stage4]:  Conv1x1 →32ch  + Upsample×4 (bicubic)  →  (1, 32, 128, 256)
  in[stage3]:  Conv1x1 →32ch  + Upsample×2 (bicubic)  →  (1, 32, 128, 256)
  in[stage2]:  Conv1x1 →32ch                            →  (1, 32, 128, 256)
  merge (add) →                                            (1, 32, 128, 256)
  middle: 1× ResBlock(MBConv expand=4)                   →  (1, 32, 128, 256)
  outputs:
     Conv1x1 →128ch (final_expand=4) + hswish + BN     →  (1, 128, 128, 256)
     Conv1x1 →19ch (logits)                              →  (1,  19, 128, 256)
```

**算力直觉（粗估）**：
- 早期高分辨率 MBConv / Conv（语义 stage1/2；代码/NVTX `stage0/1`）是 backbone 里 **绝对算力大头**；Phase 1 Nsight 已验证代码/NVTX `stage0` 是最大 GPU kernel 热点。
- 语义 stage3/4（代码/NVTX `stage2/3`）的 LiteMLA 特征图较小（64×128 / 32×64），单次理论算力不是全模型最大项。但 LiteMLA 由 qkv、aggregation、cat、linear attention、proj 等多个子路径组成，kernel 更碎，且在 FP16/TensorRT 部署时还涉及 autocast-disabled 的数值策略。Phase 1 Plan D 已确认代码/NVTX `stage2` 的 LiteMLA 内部主要耗时集中在 aggregation 与 relu_linear_att；至于这些耗时更偏 launch-bound、memory-bound 还是 compute-bound，需要 Phase 3 前用 Nsight Compute 或 microbenchmark 进一步确认。
- Seg Head 的两次 bicubic upsample + add + final_expand 是一个常被忽视的耗时点（H/8 上的 1×1 Conv 输入是 128×256 = 32768 像素）。

---

## 4. 候选融合目标（喂给阶段三 Plugin / 工程优化）

> **Phase 1 Nsight attribution 后修正**：候选目标不能只按论文模块排序，也不能只按 PyTorch profile 中的最大耗时排序。当前 B0/MX250 profile 下，代码/NVTX `stage0` 是最大 GPU kernel 热点，代码/NVTX `stage2` 第二，`head/middle` 也很明显；但 `stage0/block*/main`、`head/middle`、`stage2/block*/local` 主要是 MBConv/Conv 系列，属于 TensorRT/cuDNN 可能已经较好处理的标准算子区域。LiteMLA 不是最大端到端瓶颈，但更符合自定义 Plugin 的高区分度主线。

### P1：语义 stage3 / 代码-NVTX stage2 的 LiteMLA Plugin 主线

Plan D 第一版只细拆代码/NVTX `stage2` 中的两个 LiteMLA，因为 Plan B/C 显示这里的 context 模块在后半 backbone 中更值得优先解释。候选边界分三层：

| 优先级 | 候选边界 | 角色 | 主要理由 |
|---|---|---|---|
| P1a | `aggregation-only` / `relu_linear_att-only` | MVP / 单段验证 | 边界最小，便于先验证 Plugin 接入、数值对齐和 Nsight 归因；其中 `relu_linear_att-only` 最能体现非标准线性注意力。 |
| P1b | `aggregation + cat + relu_linear_att` | 主性能评估方向 | Plan D 显示 `aggregation` 与 `relu_linear_att` 是 LiteMLA 内部两大主耗时，中间 `cat` 连接两者；组合边界有机会减少中间 tensor 写回、读取和 kernel launch。 |
| P1c | 整体 LiteMLA | fallback / 上限方案 | 融合空间最大，但需要覆盖 qkv、aggregation、cat、linear attention、proj 和残差前后语义，数值验证与维护风险最高，不建议作为第一版目标。 |

### P2：标准算子链工程优化候选

这些区域可能贡献更大的 PyTorch 端到端耗时，但不应直接等同于“最适合写 Plugin”：

- `stage0/block*/main`：早期高分辨率 MBConv / Conv，是当前最大热点；但它主要是标准卷积链，需等 Phase 2 TensorRT baseline 判断是否仍有明显残余瓶颈。
- `head/middle`：MBConv，在 Plan C 中耗时明显；同样属于标准卷积链，适合作为 Phase 2 后的工程优化候选。
- `stage2/block*/local`：MBConv local module，与 context 相邻但性质不同；不能把 local 的耗时归因到 LiteMLA。
- `head` 的 resize/add 路径：bicubic 在 TensorRT 中有部署风险，可能需要降级 bilinear 或补一个工程型 plugin；这更多是精度/部署兼容问题，而不是 LiteMLA 主线。

### P3：低优先级探索项

- `Conv1x1(qkv) → DWConv 5×5 → grouped Conv1x1 → cat` 的多尺度聚合前段：B0 scales 只有一个分支，收益不宜高估；但未来扩到更大模型时价值会上升。
- INT8 / mixed precision：需要等待 Phase 2/3 的 TensorRT baseline 与数值验证，不能由 Phase 1 FP32 profile 直接决定。

### 不推荐作为 Phase 3 Plugin 主线

- **LayerNorm+残差**：EfficientViT-Seg-B0 中不存在这个序列。
- **MatMul+Softmax+Scale**：LiteMLA 没有 softmax，传统 ViT attention 融合假设不适用。
- **泛化 MBConv / DSConv Plugin**：标准卷积链大概率已有 TensorRT/cuDNN 优化路径，除非 Phase 2 证明它们在 TensorRT 下仍是未被优化的残余热点。

---

## 5. 对 V3.0 战略文档的修订建议

| V3.0 原文 | 建议修订为 |
|-----------|------------|
| "重点关注 MatMul+Softmax+Scale" | "重点关注 **LiteMLA 线性注意力核**：`ReLU(Q/K) → V·K^T → VK·Q → 末尾归一化除法` 整段融合" |
| "LayerNorm+残差 等算子序列" | "**EfficientViT 全程使用 BN2d，无 LN**。BN 可由 TRT 自动折叠，无需 Plugin。改为关注 **`Conv1x1 + bicubic_up + add`**（Seg Head 多输入融合）" |
| "找出 N 个值得融合的算子序列" | 源码精读阶段先识别 LiteMLA、SegHead、早期 MBConv/Conv 等候选；Phase 1 Nsight attribution 后修订为 P1 LiteMLA 分层 Plugin 主线 + P2 标准算子链工程优化候选 + P3 低优先级探索项 |

> 本修订方向已同步到项目根目录的 [`PROJECT_STRATEGY.md`](../PROJECT_STRATEGY.md)，作为 Phase 2/3 的后续战略依据。

---

## 6. 阶段一 NVTX 标注建议（直接对应决策 2）

详见 `phase1/README.md` 的"决策 2"小节。本报告负责回答"模型有哪些天然的层级断点"，README 负责落地为代码里的 NVTX range。
