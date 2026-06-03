# `phase1/scripts/` — 脚本使用速查

> **配套设计文档**：[`../design_notes/baseline_inference_design.md`](../design_notes/baseline_inference_design.md)
> **契约**：所有脚本必须先在 [`PROJECT_CONVENTIONS.md`](../../../PROJECT_CONVENTIONS.md) §1 三段式下与用户确认设计，才能落盘。
> **运行环境**：`conda activate efficientvit` (Windows / PowerShell)

---

## 脚本清单

| 脚本 | 状态 | 一句话定位 | 设计文档 |
|------|------|----------|---------|
| `baseline_inference.py` | ✅ v1.0 | 单卡 batch=1 latency 基准 + Nsight NVTX 注入 | [baseline_inference_design.md](../design_notes/baseline_inference_design.md) |
| `compare_baselines.py` | ⏳ TODO | 对多份 JSON 做表格化对比 | — |
| `evaluate.py` | ⏳ TODO（可选） | Cityscapes mIoU 评估 | — |

---

## `baseline_inference.py` 速查

### 必读

- **必须** `--weights` 或 `--allow-random-weights` 二选一；前者用于正式 baseline，后者仅用于 smoke test。
- **默认主口径**：`--measurement-mode latency`（逐次 CUDA Event，报 p50/p95/p99）。
- **默认 NVTX**：`A`（无 NVTX，干净 latency）。
- **默认输入**：1024×2048 + 固定 seed 的 dummy 张量。正式报告请用 `--input-image`。
- **输出位置**：默认 `phase1/results/metrics/baseline_<model>_<dataset>_<HxW>_level<X>_<mode>_<timestamp>.json`。

### CLI 一览（重要参数）

```
--model         {b0}                       # Phase 1 只支持 b0
--dataset       {cityscapes}
--weights       PATH                       # 必填（除非 --allow-random-weights）
--allow-random-weights                     # smoke test only
--resolution    H W                        # 默认 1024 2048
--input-image   PATH                       # 可选；省略则用 dummy
--device        cuda
--seed          INT                        # 默认 2026

--nvtx-level    {A,B,C}                    # A=无, B=stage-level, C=hotspot component-level
--profile-macs                             # 可选 torchprofile MACs

--warmup        20
--measure       100
--measurement-mode  {latency, throughput}  # 默认 latency
--cudnn-benchmark   {on, off}              # 默认 on

--out           PATH                       # 输出 JSON 路径；省略则自动派生
--dry-run                                  # 解析 + 构建 + 不计时，调试用
```

### 场景 1：正式 baseline（推荐）

```powershell
# 干净 latency（Plan A）
python phase1/scripts/baseline_inference.py `
    --weights phase1/weights/b0.pt `
    --input-image phase1/data/cityscapes_sample.png `
    --resolution 1024 2048 `
    --nvtx-level A `
    --measurement-mode latency `
    --warmup 20 --measure 100
```

控制台尾部会输出：
```
[OK] saved -> E:\...\phase1\results\metrics\baseline_b0_cityscapes_1024x2048_levelA_latency_<ts>.json
  latency (ms):  mean=XXX  p50=XXX  p95=XXX  p99=XXX
```

### 场景 2：Nsight Plan B（stage 级归因）

```powershell
nsys profile `
    -o phase1/results/nsight/levelB `
    --trace=cuda,nvtx `
    --force-overwrite=true `
    python phase1/scripts/baseline_inference.py `
        --weights phase1/weights/b0.pt `
        --resolution 1024 2048 `
        --nvtx-level B `
        --warmup 20 --measure 100
```

Nsight UI 中应能看到 `stem / stage0 / stage1 / stage2 / stage3 / head` 六个 range。

### 场景 3：Nsight Plan C（热点组件级归因）

```powershell
nsys profile `
    -o phase1/results/nsight/levelC `
    --trace=cuda,nvtx `
    --force-overwrite=true `
    python phase1/scripts/baseline_inference.py `
        --weights phase1/weights/b0.pt `
        --resolution 1024 2048 `
        --nvtx-level C `
        --warmup 20 --measure 100
```

**注意**：
- Plan C 使用 forward hooks 展开 `backbone.stages.2`、`backbone.stages.3` 和 `head` 内部组件，不改写 forward 数值路径，因此不需要 sanity check。
- EfficientViTBlock 的真实执行顺序是 `context_module -> local_module`；Plan C range 命名也按这个顺序。
- `head` 的 merge add 是 `DAGBlock.forward()` 内部函数调用，不是独立 module，当前 hook-only 方案不单独计入一个 range。
- LiteMLA 内部 `qkv / aggregation / attention matmul / proj` 子算子级 profiling 需要另写专门脚本。

> Windows Nsight Systems 2026.2.1 实测：`osrt` 不是合法 trace 值；`wddm` 需要管理员权限，普通终端会被禁用。Phase 1 建议统一使用 `--trace=cuda,nvtx`。

### 场景 4：Smoke test（无权重，快速跑通流程）

```powershell
python phase1/scripts/baseline_inference.py `
    --allow-random-weights `
    --resolution 512 1024 `
    --warmup 2 --measure 5 `
    --nvtx-level A
```

输出 JSON 中 `is_smoke_test=true`、`weights_status="random"`，**不应**作为正式数据使用。

### 场景 5：Dry-run（只验证 CLI / 模型构建）

```powershell
python phase1/scripts/baseline_inference.py `
    --allow-random-weights --dry-run
```

只构建模型、解析参数、打印预期输出路径，不进入计时。CI / lint 用。

---

## 输出 JSON 解读速查

| 字段 | 含义 | 备注 |
|------|------|------|
| `status` | `ok` | 不应出现 `missing` |
| `script_version` | `baseline_inference.py@<sha>[-dirty]` | 若 git 不可用，会是 `git_unavailable` |
| `is_smoke_test` | true 仅当 `--allow-random-weights` | 正式报告必须为 false |
| `weights.weights_sha256` | 权重文件 hash | 用于回溯"我用的是哪份权重" |
| `input.input_sha256` | 输入张量/图像 hash | 复现性锚点 |
| `timing.mode` | `latency` / `throughput` | 比较时必须同模式 |
| `timing.ms.p50/p95/p99` | 单帧延迟分位数（ms） | 主报告口径 |
| `nvtx.level` | A/B/C | 与 nsys-rep 文件名对齐 |
| `nvtx.component_ranges` | Plan C 组件级 range 列表 | 例如 `stage2/block1/context` |
| `sanity_check.performed` | 当前 A/B/C 均为 `false` | Plan C 是 hook-only，不改数值路径 |
| `memory.max_memory_allocated_mb` | peak GPU memory | MX250 上盯紧 2GB 上限 |

完整 schema 见 [`baseline_inference_design.md` §4](../design_notes/baseline_inference_design.md#4-json-schemav10)。

---

## 常见问题

**Q：第一次跑报 `ModuleNotFoundError: efficientvit`？**
A：脚本会自动把 repo root 加进 `sys.path`，但前提是脚本必须放在 `phase1/scripts/` 下、repo 结构未被破坏。检查 `git rev-parse --show-toplevel` 输出是否为 `E:/EdgeSeg-EfficientViT/EdgeSeg-EfficientViT`。

**Q：Nsight 没看到 NVTX range？**
A：确认 `--trace` 包含 `nvtx`；确认 `--nvtx-level` 不是 `A`；确认 nsys 用的是 PATH 中的 2026.2.1 版本（`nsys --version`）。

**Q：latency 抖动很大（std > mean × 20%）？**
A：①确认 `--cudnn-benchmark on`；②增大 `--warmup`（如 50）；③关闭桌面录屏 / Chrome 硬件加速；④检查 `nvidia-smi` 是否有其他进程占用 MX250。

**Q：Plan C 为什么不再做 sanity check？**
A：当前 Plan C 只注册 forward hooks，不 monkey-patch 模块，也不改写任何计算路径；hooks 只做 NVTX push/pop，所以没有 patched-vs-original 数值对照需求。

**Q：MX250 OOM？**
A：①关闭 `--profile-macs`；②降到 `--resolution 512 1024`；③确保没在用 batch>1；④`nvidia-smi` 关掉其他 CUDA 进程。

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-05-28 | 初版，对应 `baseline_inference.py` v1.0 |
