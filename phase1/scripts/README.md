# `phase1/scripts/` — 脚本使用速查

> **配套设计文档**：[`../design_notes/baseline_inference_design.md`](../design_notes/baseline_inference_design.md)
> **契约**：所有脚本必须先在 [`PROJECT_CONVENTIONS.md`](../../PROJECT_CONVENTIONS.md) §1 三段式下与用户确认设计，才能落盘。
> **运行环境**：`conda activate efficientvit` (Windows / PowerShell)

---

## 脚本清单

| 脚本 | 状态 | 一句话定位 | 设计文档 |
|------|------|----------|---------|
| `baseline_inference.py` | ✅ implemented | 单卡 batch=1 latency 基准 + Nsight NVTX 注入 | [baseline_inference_design.md](../design_notes/baseline_inference_design.md) |
| `analyze_nsys_attribution.py` | ✅ implemented | 从 Nsight SQLite 中按 NVTX range 归因 CUDA kernel 耗时，输出 Markdown/JSON 汇总表 | 脚本 docstring |
| `compare_baselines.py` | ⏸ Deferred（未实现） | 可选：对多份 baseline JSON 做表格化对比 | — |
| `evaluate.py` | ⏸ Out of Phase 1 mainline（未实现） | 可选：Cityscapes mIoU / 精度评估；不属于 Phase 1 baseline 主线 | — |

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

--nvtx-level    {A,B,C,D}                  # A=无, B=stage-level, C=stage0/stage2/head, D=stage2 LiteMLA internal
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
    --weights phase1/weights/efficientvit_seg_b0_cityscapes.pt `
    --input-image phase1/data/city_asset_cityscapes_like.png `
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
        --weights phase1/weights/efficientvit_seg_b0_cityscapes.pt `
        --input-image phase1/data/city_asset_cityscapes_like.png `
        --resolution 1024 2048 `
        --nvtx-level B `
        --warmup 20 --measure 100
```

Nsight UI 中应能看到 `stem / stage0 / stage1 / stage2 / stage3 / head` 六个 range。

### 场景 3：Nsight Plan C（stage0/stage2/head 热点组件级归因）

```powershell
nsys profile `
    -o phase1/results/nsight/levelC `
    --trace=cuda,nvtx `
    --force-overwrite=true `
    python phase1/scripts/baseline_inference.py `
        --weights phase1/weights/efficientvit_seg_b0_cityscapes.pt `
        --input-image phase1/data/city_asset_cityscapes_like.png `
        --resolution 1024 2048 `
        --nvtx-level C `
        --warmup 20 --measure 100
```

**注意**：
- Plan C 使用 forward hooks 展开 `backbone.stages.0`、`backbone.stages.2` 和 `head` 内部组件，不改写 forward 数值路径，因此不需要 sanity check。
- `stage0` 展开为 `block0/main`、`block1/main`；`stage2` 中 EfficientViTBlock 的真实执行顺序是 `context_module -> local_module`，Plan C range 命名也按这个顺序。
- `head` 的 merge add 是 `DAGBlock.forward()` 内部函数调用，不是独立 module，当前 hook-only 方案不单独计入一个 range。
- 组件占比分析不要直接使用 NVTX range duration；请从 Nsight sqlite 中用 CUDA runtime/kernel `correlationId` 将 kernel duration 归因到 NVTX range。

### 场景 3.5：Nsight Plan D（stage2 LiteMLA 内部子路径归因）

```powershell
nsys profile `
    -o phase1/results/nsight/levelD `
    --trace=cuda,nvtx `
    --force-overwrite=true `
    python phase1/scripts/baseline_inference.py `
        --weights phase1/weights/efficientvit_seg_b0_cityscapes.pt `
        --input-image phase1/data/city_asset_cityscapes_like.png `
        --resolution 1024 2048 `
        --nvtx-level D `
        --warmup 20 --measure 100
```

Plan D 使用实例级 `LiteMLA.forward` patch，只针对 `stage2/context` 的两个 LiteMLA 拆第一层内部子路径：

- `stage2/block1/litemla/qkv`
- `stage2/block1/litemla/aggregation`
- `stage2/block1/litemla/cat`
- `stage2/block1/litemla/relu_linear_att`
- `stage2/block1/litemla/proj`
- `stage2/block2/litemla/...`

注意：
- `relu_linear_att()` 本身保持黑盒调用，因此上游 `@torch.autocast(device_type="cuda", enabled=False)` 与 dtype 逻辑仍然生效。
- Plan D 会在 profiling 前做 patched-vs-original `torch.allclose(atol=1e-5, rtol=1e-5)` sanity check；sanity 阶段不会发出正式 Plan D NVTX range，避免污染 attribution 的 warmup/measure 切片。
- Plan D 用于确定 Phase 3 stage2 LiteMLA Plugin 的具体可融合子路径，不替代 Plan B 的全模型归因，也不替代 Plan C 的热点组件归因。

> Windows Nsight Systems 2026.2.1 实测：`osrt` 不是合法 trace 值；`wddm` 需要管理员权限，普通终端会被禁用。Phase 1 建议统一使用 `--trace=cuda,nvtx`。在自动化/AI 执行上下文中，优先使用全路径 `D:\software\nsight_systems\target-windows-x64\nsys.exe`；普通沙箱可能导致 nsys 固定 75 秒超时，需在非沙箱权限下运行。

已归档截图位于 `phase1/results/figures/`：

- Plan B：`planB_timeline_overview.png`、`planB_single_forward_nvtx.png`
- Plan C：`planC_timeline_overview.png`、`planC_stage0_components.png`、`planC_stage2_components.png`、`planC_head_components.png`
- Plan D：`planD_timeline_overview.png`、`planD_litemla_aggregation_components.png`、`planD_litemla_relu_linear_att_components.png`

读图口径：组件/阶段名称与边界看 `Threads -> NVTX`；GPU kernel 对应关系看 `CUDA HW -> Kernels`；`CUDA HW -> NVTX` 的宽度只作趋势参考。报告中的定量耗时仍以 `results/metrics/*_nsys_attribution_summary.md` 为准。

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
| `nvtx.level` | A/B/C/D | 与 nsys-rep 文件名对齐 |
| `nvtx.component_ranges` | Plan C/D 组件级 range 列表 | 例如 `stage0/block0/main`、`stage2/block1/context`、`stage2/block1/litemla/relu_linear_att` |
| `sanity_check.performed` | Plan D 为 `true`，A/B/C 为 `false` | Plan D patch LiteMLA.forward，需做数值等价检查 |
| `memory.max_memory_allocated_mb` | peak GPU memory | MX250 上盯紧 2GB 上限 |
| `env.env_patches` | 当前 run 使用的 import 兼容补丁 | Windows/MX250 上通常为 `["triton_stub", "wandb_stub"]` |
| `env.wandb_stubbed` | 是否屏蔽上游训练日志库 `wandb` | 仅用于避免推理脚本退出时出现 wandb 临时目录清理警告 |

完整 schema 见 [`baseline_inference_design.md` §4](../design_notes/baseline_inference_design.md#4-json-schemav10)。

---

## 常见问题

**Q：第一次跑报 `ModuleNotFoundError: efficientvit`？**
A：脚本会自动把 repo root 加进 `sys.path`，但前提是脚本必须放在 `phase1/scripts/` 下、repo 结构未被破坏。检查 `git rev-parse --show-toplevel` 输出是否为 `E:/EdgeSeg-EfficientViT/EdgeSeg-EfficientViT`。

**Q：Nsight 没看到 NVTX range？**
A：确认 `--trace` 包含 `nvtx`；确认 `--nvtx-level` 不是 `A`；确认 nsys 用的是 PATH 中的 2026.2.1 版本（`nsys --version`）。

**Q：latency 抖动很大（std > mean × 20%）？**
A：①确认 `--cudnn-benchmark on`；②增大 `--warmup`（如 50）；③关闭桌面录屏 / Chrome 硬件加速；④检查 `nvidia-smi` 是否有其他进程占用 MX250。


**Q：MX250 OOM？**
A：①关闭 `--profile-macs`；②降到 `--resolution 512 1024`；③确保没在用 batch>1；④`nvidia-smi` 关掉其他 CUDA 进程。

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-05-28 | 初版，对应 `baseline_inference.py` |
| 2026-06-05 | 文档示例同步为 Phase 1 正式权重与固定样图路径；脚本清单不再写死实现版本号 |
