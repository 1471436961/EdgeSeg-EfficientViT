# EfficientViT-Seg-B0 架构精读报告

> **目的**：在动笔写 `baseline_inference.py` 与 Nsight 剖析脚本之前，先把模型结构、关键算子序列、张量形状 **完全摸清**。本报告直接决定：
>
> 1. NVTX range 应该插在**哪些层级 / 哪些算子序列**上（决策 3）；
> 2. 阶段三 TensorRT Plugin 的**候选融合目标**是什么（V3.0 战略核心）；
> 3. 哪些"想当然"的传统 Transformer 假设在 EfficientViT 上**根本不成立**（避免阶段二/三踩坑）。
>
> **源码版本**：`mit-han-lab/efficientvit @ master`（本地 commit `de7d773`）。
> **撰写日期**：2026-05-26。

---

## 0. 关键结论先行（TL;DR）

| # | 结论 | 对后续阶段的影响 |
|---|------|------|
| 1 | **EfficientViT 不是传统 Transformer**：没有 LayerNorm、没有 Softmax、没有 Patch Embedding、没有位置编码。 | V3.0 文档中"LayerNorm+残差""MatMul+Softmax+Scale"的融合假设 **需要修正**。 |
| 2 | **核心创新 = LiteMLA（轻量多尺度线性注意力）**：`ReLU(Q), ReLU(K), V` 后做 `(V·K^T)·Q` 的线性顺序乘法，复杂度 O(N·d²) 而非 O(N²·d)。 | 真正的融合机会是 `Conv1x1(QKV) → 多尺度 DWConv → ReLU → MatMul → MatMul → 归一化除法` 这一序列。 |
| 3 | **归一化全部用 BN2d**（而非 LN）。 | TRT 部署时 **BN 直接折叠进 Conv**，几乎零开销；**不必单独融合 LN**。 |
| 4 | **B0 注意力只在 stage3/stage4 出现**，且只有 2 个 `EfficientViTBlock`。 | Nsight attribution 已验证：注意力/context 不是全模型最大瓶颈；最大热点在早期高分辨率 `stage0`，其次是 `stage2`，`head` 也有明确耗时。 |
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

注意力**只**在 stage3 / stage4 出现。前 3 个阶段是纯 CNN（MobileNetV3 风格）。

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
- `UpSampleLayer.forward` 显式 `@torch.autocast(enabled=False)`，**强制 FP32 跑插值**。这意味着 FP16 推理时这里会有 `cast→up→cast` 的额外开销，是 Nsight 上能看到的一个"小尖刺"。

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
- 对 Nsight 时间线：你会在注意力区段看到明显的 `dtype cast`，可以作为优化标记点。

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
- 早期高分辨率 MBConv / Conv（语义 stage1/2；代码中对应 `backbone.stages.0/1`）是 backbone 里 **绝对算力大头**；Phase 1 Nsight 已验证 `backbone.stages.0` 是最大 GPU kernel 热点。
- stage3/4 的 LiteMLA 特征图小（64×128 / 32×64），单次算力不大，但 **算子种类多、kernel launch 多、cast 多**，更容易出现 launch overhead 而非计算 bound。
- Seg Head 的两次 bicubic upsample + add + final_expand 是一个常被忽视的耗时点（H/8 上的 1×1 Conv 输入是 128×256 = 32768 像素）。

---

## 4. 候选融合目标（喂给阶段三 Plugin / 工程优化）

> **Phase 1 Nsight attribution 后修正**：候选目标不能只按论文模块排序。当前 B0/MX250 profile 下，`stage0` 是最大 GPU kernel 热点，`stage2` 第二，`head/middle` 是明显组件热点；LiteMLA/context 值得做，但不是最大瓶颈。因此本节按"求职展示价值"与"端到端收益潜力"双维度排序。

### 🥇 优先级 1：**LiteMLA 注意力核** —— 高区分度 Plugin 主交付物
**融合范围**：从 `qkv 与多尺度聚合的 concat 之后`，到 `proj 之前`。
即把 ③ reshape → ④ ReLU → ⑤ pad+transpose → ⑥ 两次 MatMul → 归一化除法 → ⑦ reshape 这一整段写成一个 CUDA kernel（或几个紧凑 kernel）。

**理论收益**：
- 消除 reshape / transpose / pad 这些 view/layout 算子（PyTorch 里它们看似免费，TRT 里却常常 materialize 为真实 kernel）；
- 让 `V·K^T` 和 `VK·Q` 共享 shared memory，省一次 HBM 来回；
- 把"末尾 1 行的归一化除法"做成 fused epilogue，消除一次完整 traversal。
- 预期注意力/context 段加速 **1.5×~2.5×**（待实测验证），但端到端收益需按该段真实占比保守估算。

**简历卖点**：直接对标论文中的 LiteMLA 模块，明确"为线性注意力写了一个 TRT Plugin"，比"我做了 MatMul+Softmax 融合"具体得多。

### 🥈 优先级 2：**Seg Head / head middle 多输入融合**
**融合范围**：head 的 3 路 1×1 Conv 输出 + 2 路 bicubic upsample + add。
- 真正的痛点：**bicubic upsample 在 TRT 不原生支持**，要么走 plugin，要么降级 bilinear。
- 如果走 plugin，可以顺手把 add 也融进去：`Conv1x1 + bicubic_up + add` 三件套一锅炒。
- 风险：bicubic→bilinear 降级会带来 mIoU 损失，需要阶段二实测。
- Phase 1 Plan C 显示 `head/middle` 是明显组件热点，端到端收益和工程必要性都比预想更强。

### 🥉 优先级 3：**stage0 early MBConv / Conv 堆叠优化**
**融合范围**：`stage0/block0/main`、`stage0/block1/main` 中的早期高分辨率 Conv/BN/activation/Residual 序列。
- Phase 1 Plan B/C 显示它是当前最大 GPU kernel 热点，端到端收益潜力最高。
- 风险：这些是相对标准的卷积类结构，TensorRT/cuDNN 可能已经能较好优化；手写 Plugin 的差异化不如 LiteMLA，且维护成本更高。
- 更适合作为 Phase 2 TRT baseline 后的工程优化候选，而不是一开始就替代 LiteMLA 主线。

### 备选研究项：**多尺度 QKV 聚合段**
**融合范围**：`Conv1x1(qkv) → DWConv 5×5 → Conv1x1 grouped → concat`。
- 这里有一个 **concat 操作**，TRT 里 concat 经常拖累——可以重写为"直接写入连续缓冲区，省掉 concat"。
- B0 scales 只 1 个，融合收益相对有限；但对 B1/B2 如果未来用到（scales 可能更多），价值更高。

### ⚠️ 不推荐融合
- **MBConv / DSConv**：TRT 自己有非常成熟的 `Conv-BN-Act` 融合，自己写 plugin 反而更慢。
- **LayerNorm+残差**：**不存在**这个序列，原 V3.0 文档需修订。
- **Softmax**：**不存在**，整个模型 0 个 softmax。

---

## 5. 对 V3.0 战略文档的修订建议

| V3.0 原文 | 建议修订为 |
|-----------|------------|
| "重点关注 MatMul+Softmax+Scale" | "重点关注 **LiteMLA 线性注意力核**：`ReLU(Q/K) → V·K^T → VK·Q → 末尾归一化除法` 整段融合" |
| "LayerNorm+残差 等算子序列" | "**EfficientViT 全程使用 BN2d，无 LN**。BN 可由 TRT 自动折叠，无需 Plugin。改为关注 **`Conv1x1 + bicubic_up + add`**（Seg Head 多输入融合）" |
| "找出 N 个值得融合的算子序列" | 在 architecture_analysis.md 中已锁定 **3 个候选**，阶段一 Nsight 主要验证它们各自的耗时占比和加速空间 |

> 本修订方向已同步到仓库外的 `PROJECT_STRATEGY.md`，作为 Phase 2/3 的后续战略依据。

---

## 6. 阶段一 NVTX 标注建议（直接对应决策 3）

详见 `phase1/README.md` 的"决策 3"小节。本报告负责回答"模型有哪些天然的层级断点"，README 负责落地为代码里的 NVTX range。
