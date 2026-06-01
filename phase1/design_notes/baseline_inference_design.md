# `baseline_inference.py` — 设计文档

> **关联代码**：[`phase1/scripts/baseline_inference.py`](../scripts/baseline_inference.py)
> **关联契约**：[`PROJECT_CONVENTIONS.md`](../../../PROJECT_CONVENTIONS.md) §1（三段式）§2（文档分层）
> **关联战略**：[`Floatboat.md`](../../../Floatboat.md) V3.1 · Phase 1
> **关联架构分析**：[`phase1/architecture_analysis.md`](../architecture_analysis.md)
> **创建日期**：2026-05-28
> **状态**：v1.0（初版落盘，对应脚本第一次 commit）

---

## 0. 一句话定位

> **`baseline_inference.py` 是 Phase 1 的核心可执行物**：在 MX250 上、固定输入、固定权重、固定测时口径下，输出一份**可复现、可被 Nsight 归因、机器可读**的 EfficientViT-Seg-B0 推理延迟基准（JSON），为 Phase 2/3 的所有优化提供**参照线**。

---

## 1. 目标与非目标

### 1.1 目标（必须达成）

1. **可复现**：给定 `--weights` + `--input-image`（或固定 dummy seed），任意主机重跑应得到统计上等价的 latency 分布（mean ±3σ 之内）。
2. **可归因**：通过 `--nvtx-level B/C` 让 Nsight Systems 能把 timeline 上的 CUDA kernel 归属到我们关心的结构（stem / stage0..3 / head；或 LiteMLA 内部）。
3. **机器可读**：单次运行产出一份 JSON，包含**测时 + 环境 + 权重 + 输入 + sanity + 可选 MACs**，无需 grep 控制台。
4. **零侵入**：不修改 `efficientvit/` 源码；NVTX 通过 hook + 实例级 monkey-patch 注入，运行结束后还原。
5. **契约对齐**：严格满足用户在三段式 §2 中提出的 7 条实现约束（见 §6）。

### 1.2 非目标（明确不做）

- ❌ **不**测 mIoU / 不做 Cityscapes 全集评估 → 留给 Phase 1 后续 `evaluate.py`（如果决定写）。
- ❌ **不**做多 batch / 动态分辨率 sweep → 输入分辨率与 batch=1 固定。
- ❌ **不**做训练 / 微调 → 纯 inference。
- ❌ **不**承担 Plugin 实现 → Plan-C 只产 NVTX 归因数据，**不**等价于 fused kernel。
- ❌ **不**自动比较多次 run → 多 run 对比留给 Phase 1 后续的 `compare_baselines.py`。

---

## 2. 总体形状

```
                                ┌─────────────────────────┐
   CLI args ──► validate_args ──┤   build_model / weights │──► model (eval, cuda)
                                ├─────────────────────────┤
                                │   build_input_tensor    │──► x  (1,3,H,W)
                                ├─────────────────────────┤
                                │   sha256 hashing        │  *outside* timing
                                ├─────────────────────────┤
                                │   (optional) MACs       │  *outside* timing
                                ├─────────────────────────┤
                                │   NVTX inject:          │
                                │     B = hooks           │
                                │     C = sanity + patch  │  *patch only if sanity OK*
                                ├─────────────────────────┤
                                │   warmup (no record)    │
                                │   measure (CUDA Events) │  <-- only this is "timing"
                                ├─────────────────────────┤
                                │   remove hooks / patch  │  finally:
                                ├─────────────────────────┤
                                │   assemble + save JSON  │
                                └─────────────────────────┘
```

四个关键边界：

- **`x` 在 GPU 上**之后，所有计时之前，先完成 hashing/MACs/sanity（约束 #6）。
- **NVTX 注入**只发生在 warmup 之前；运行结束在 `finally` 还原（约束 #1/#2）。
- **CUDA sync** 只出现在 warmup/measure 边界、CUDA Event 读取，**绝不**出现在 NVTX range 内部。
- **JSON 落盘**永远是 main 的最后一步，确保即便 sanity 失败也能留下取证报告。

---

## 3. 关键设计决策（含取舍）

### 3.1 测时口径：双模式，latency 为默认主口径

| 模式 | 实现 | 报告 | 适用场景 |
|------|------|------|---------|
| `--measurement-mode latency`（默认） | 每次 iter `start.record() / model(x) / end.record() / end.sync()`，逐次读 `start.elapsed_time(end)` | p50 / p95 / p99 / mean / std / min / max | **单请求 latency**（机器人单帧、端侧推理），**Phase 1 主口径** |
| `--measurement-mode throughput` | 整批 enqueue 100 次 → 单次 `torch.cuda.synchronize()` → 单个 Event pair | total_ms / avg_ms / fps | 稳态 GPU 吞吐参考；**辅助口径** |

**取舍**：
- ✅ **逐次 CUDA Event 不引入 host-side sync 进入计时区间**：`end.synchronize()` 只是等 event 完成，等待时间**不**计入 `start.elapsed_time(end)`，所以逐次测量是合法的、且数学上能正确反映单请求 latency。
- ❌ **拒绝把 throughput 当主口径**：批量 enqueue 后单 sync 测的是 steady-state 平均，对 batch=1 单请求是误导性的（掩盖了 kernel launch 抖动 / 首尾尾流效应）。
- 🟡 **JSON 同时记录 `measurement_mode` 字段**，未来对比时不会混淆口径。

> ⚠️ 这条决策来自用户在 v2 评审中的第 1 条修正。原始草案错误地把 throughput 当主口径。

### 3.2 NVTX 分层：Plan A / B / C 三档

| 档位 | 注入方式 | range 粒度 | 用途 | 风险 |
|------|---------|----------|-----|------|
| A（默认） | 无 | 0 | **干净 latency 基准**，与 Phase 2/3 对比的 anchor | 0 |
| B | `register_forward_pre_hook` + `register_forward_hook` 在 `stem / stage0..3 / head` 上 | 6 ranges | **整体瓶颈定位**（stage 级） | hook 极轻量，几乎可忽略 |
| C | 实例级 `MethodType` monkey-patch 每个 `LiteMLA.forward` | ~8 ranges（B0 在 stage3/4 各 2 个 LiteMLA × 2 个 stage 边界） | **Plugin 设计输入**（验证 LiteMLA 是否真的是瓶颈，量化它占比） | 改写了 forward 调用路径，需要 sanity check 兜底 |

> 编号说明：Plan B 的 `stage0..3` 是 `backbone.stages` 的 `ModuleList` 索引；架构分析中的 `stage1..4` 是语义阶段编号，两者一一对应。

**取舍 1：为何 B 与 C 分开跑，而不合并？**
- 合并 = 同一份 timeline 上既有 stage 级、又有 LiteMLA 内部级 → Nsight UI 看着乱，且 Plan C 的 patch 会改变 `LiteMLA.forward` 的入口栈帧，Plan B 的 hook 可能与之打架。
- 分开 = 两次 nsys run，分别 `levelB.nsys-rep` / `levelC.nsys-rep`，分析时按需切换。
- 代价：多跑一次。但 MX250 上单次 nsys + 100 iter ≈ 30 秒，可接受。

**取舍 2：为何 Plan C 是实例级（`types.MethodType`）而不是类级？**

| 维度 | 类级 `LiteMLA.forward = wrapper` | 实例级 `m.forward = MethodType(wrapper, m)` |
|------|----------------------------------|--------------------------------------------|
| 副作用范围 | **整个 Python 进程**所有 LiteMLA 实例 | 仅本次脚本构建的 model 内的实例 |
| 还原难度 | 必须显式 `LiteMLA.forward = original` | `del m.forward` 即可自动回落到类方法 |
| 重入风险 | 多次调用会层层包裹 → 需要 sentinel | 每个实例独立检查 `_edgeseg_nvtx_patched` 即可 |
| 跨实例隔离 | ❌ 同 Notebook 里 import 多个 model 会互相污染 | ✅ 完全隔离 |
| 实现复杂度 | 略低（一行赋值） | 略高（要遍历 + setattr） |

→ **采纳实例级**。代价是多 ~10 行代码，换来更安全的副作用边界（约束 #2）。

**取舍 3：Plan C 内部只包一层 `LiteMLA` range，而不进入更细粒度？**
- 用户原始 v2 草案曾考虑在 `qkv_proj / multiscale / relu_lin_attn / proj` 上各打一个 range。
- 否决理由：
  1. **会破坏 `@torch.autocast(enabled=False)` 的语义边界**——这些子模块都在 autocast disable 区间内，子粒度 range 没问题，但**子粒度 sanity 极难做**（要 hook 4 个 sub-attribute 而非一个 forward）。
  2. **Plugin 的边界本来就是整个 LiteMLA**——Phase 3 要替换的就是这一整块（见 [`architecture_analysis.md`](../architecture_analysis.md) §结论 5），子粒度 range 对 Plugin 边界界定没增量信息。
  3. Nsight timeline 上同一 LiteMLA range 内的 CUDA kernel 序列已足够暴露内部行为。
- → 现版只包一层 `LiteMLA` range。若后续 Plugin 设计需要更细，再单独写 `litemla_internal_profile.py`，不污染本脚本。

### 3.3 Plan-C sanity check：逐模块 + hook 先于 patch

完整流程（实现在 `run_sanity_check()`）：

1. 用 `register_forward_hook` 给每个 `LiteMLA` 注册一个 capture hook，记录 `inp[0].detach().clone()` 与 `out.detach().clone()`。
2. 跑一次 `model(x)`（**原始 forward**，因为此时尚未 patch）。
3. **移除所有 hook**（约束 #4：hook 与 patch 不并存）。
4. 应用 Plan-C monkey-patch。
5. 对每个 LiteMLA `m`，**单独**调用 `m(cap['inp'])`（此时调用的是 patched forward）。
6. 与 step 2 缓存的 `cap['out']` 比较：
   - 记录 `max_abs_diff` 与 `mean_abs_diff`。
   - 用 `torch.allclose(atol=1e-5, rtol=1e-5)` 作 pass/fail 判定。
7. 任一模块失败 → 写一份带 `status=sanity_failed` 的 JSON 取证，`exit code 3`，**不进入 warmup/measure**。
8. 全通过 → 保留 patch，进入 warmup/measure。

**关键设计点**：
- ✅ **hook 在 patch 之前**：保证 step 2 跑的是货真价实的 original forward，不会出现"patched vs patched"的伪检查（约束 #1 的精神）。
- ✅ **逐模块比较而非端到端**：如果某个 LiteMLA 的 patched forward 引入了误差，逐模块能直接定位是 `backbone.stages.3.0.context_module.main`（举例），端到端比较只能看到 segout 偏差，无从定位。
- ✅ **FP32 阈值 1e-5**：Q1 选定。**未来加 FP16/AMP 时必须放宽**（典型 atol=1e-3），且 design note 这里要更新。
- ✅ **失败时仍写 JSON**：方便复现 / 提交 issue / 后续回归对比。这是少数允许 `status != "ok"` 的 JSON 之一（与约束 #7 不冲突——#7 针对的是 weights 缺失，不是数值回归）。

### 3.4 权重：强制 + hash + smoke-test 例外

| `--weights` | `--allow-random-weights` | 行为 |
|-------------|--------------------------|------|
| 给了 | 没给 | 加载，`weights_status="loaded"`，hash 写入 JSON |
| 给了 | 给了 | `parser.error()` 退出（互斥） |
| 没给 | 没给 | **直接退出**（约束 #7），不写 JSON |
| 没给 | 给了 | 随机初始化，`weights_status="random"`，`is_smoke_test=true` |

**取舍**：
- 旧草案曾允许 `weights_status="missing"` 进入正式 JSON。被否决：**正式 JSON 不应携带"我其实没加载权重"的语义**，否则下游 dashboard / Phase 2 比较脚本会无脑取数。
- smoke test 路径独立标 `is_smoke_test`，让所有消费者（人、脚本、未来 dashboard）一眼区分。
- 权重 SHA256 写入 JSON → 任何"我用的就是 b0 cityscapes"的争议都能用 hash 一句话解决。

### 3.5 输入：dummy（默认）+ image（推荐正式）+ hash

- **dummy**：`torch.randn(1, 3, H, W, generator=Generator(seed=2026))`。固定 seed，可复现。**适用于 smoke test 与初测**。
- **image**：`--input-image path.jpg`，PIL load → resize → CHW/255。**正式报告推荐**。
- 两种都写 sha256 入 JSON（`input_sha256`），dummy 的 hash 来自张量字节，image 的 hash 来自文件字节。
- JSON 字段 `input_status` ∈ {`dummy`, `image`}。

> ⚠️ Phase 1 不做 ImageNet 归一化（mean/std），因为本脚本不测 mIoU。延迟与归一化无关。**Phase 1 若决定加 `evaluate.py`，归一化要补回**。

### 3.6 `inference_mode` 替代 `no_grad`

约束 #5 要求。`torch.inference_mode()` 比 `no_grad()` 多关闭了一些 autograd 内部簿记（view tracking、version counter），对纯推理 benchmark 更干净。所有 warmup / measure / sanity / MACs 全部包在 `inference_mode` 内。

**注意**：`inference_mode` 下产生的 tensor 不能在退出 context 后被 autograd 操作。因为本脚本是 end-to-end 推理，无后续梯度需求，安全。

### 3.7 `cudnn.benchmark=on` 但 `deterministic=off`

- 固定 input shape → benchmark 能稳定挑到最优 algo → latency 更低更稳。
- deterministic=off 是 benchmark 的对偶要求，**不**追求 bit-exact 复现，追求统计意义复现。
- JSON 同时记录两个字段，将来若要做 deterministic 对比，直接看 JSON 即可知道当前 run 的口径。

### 3.8 `script_version` 的稳健 git 解析

实现路径（`resolve_script_version()`）：

```text
1. git rev-parse --show-toplevel   (from script_dir)
   失败 -> "baseline_inference.py@git_unavailable"
2. git rev-parse --short=7 HEAD    (from repo_root)
   失败 -> "baseline_inference.py@uncommitted"
3. relative = __file__ relative to repo_root
4. git diff --quiet HEAD -- <relative>   (from repo_root)
   rc=0 -> "@<sha>"
   rc=1 -> "@<sha>-dirty"
   other -> "@<sha>-unknown"
```

**取舍**：
- 用 `--show-toplevel` 拿到 repo root，然后用相对路径喂给 `git diff`，避免 Windows 绝对路径 `E:\...` 在 git pathspec 下的兼容性坑（约束 #3）。
- 任一步失败都 **fallback 到字符串**，绝不抛异常让脚本崩溃。
- 5 秒 timeout 防止 git lockup 导致 benchmark 卡住。

### 3.9 MACs：可选、隔离、绝不影响计时

- `--profile-macs` 才启用，默认关闭（约束 #6 + 用户对 thop 的否决）。
- 用已在 `requirements.txt` 中的 `torchprofile`（不引入 `thop`）。
- **跑在 warmup 之前**，结果存入 JSON 后再开始 warmup。
- 失败 best-effort（`status=error`），不抛、不退出。
- ⚠️ Plan C patched 状态下跑 MACs 可能数值不准（patched forward 多一层 Python 包装）→ 已在脚本顺序上把 MACs 放在 NVTX 注入**之前**。

---

## 4. JSON Schema（v1.0）

```json5
{
  "json_schema_version": "1.0",
  "status": "ok | sanity_failed",
  "script_version": "baseline_inference.py@<sha>[-dirty] | uncommitted | git_unavailable",
  "run_timestamp": "2026-05-28T12:34:56+0800",
  "is_smoke_test": false,
  "args": { /* full argparse Namespace */ },
  "model":   { "name": "b0", "dataset": "cityscapes" },
  "weights": {
    "weights_status": "loaded | random",
    "weights_path":   "/abs/path/b0.pt | null",
    "weights_sha256": "<hex> | null",
    "weights_load_msg": "..."
  },
  "input": {
    "input_status": "image | dummy",
    "input_path":   "/abs/path | null",
    "input_sha256": "<hex>",
    "resolution": [1024, 2048],
    "batch_size": 1,
    "dtype": "fp32"
  },
  "env": {
    "device_name": "NVIDIA GeForce MX250",
    "device_capability": "sm_61",
    "torch_version": "2.4.1+cu124",
    "cuda_version":  "12.4",
    "cudnn_version": 90100,
    "python_version": "3.10.x",
    "platform": "...",
    "hostname": "..."
  },
  "cudnn": { "benchmark": true, "deterministic": false },
  "nvtx": {
    "level": "A | B | C",
    "applied": true,
    "hook_count": 14,                    // Plan B only
    "patched_modules": ["backbone.stages.3...", ...]  // Plan C only
  },
  "sanity_check": {
    "performed": true,
    "passed": true,
    "atol": 1e-5, "rtol": 1e-5,
    "per_module": [
      { "name": "...", "ok": true, "max_abs_diff": 0.0, "mean_abs_diff": 0.0 }
    ],
    "notes": "checked N LiteMLA modules"
  },
  "macs": {
    "status": "skipped | ok | error | unavailable",
    "macs": 4400000000,           // null unless status=ok
    "tool": "torchprofile",
    "error": null
  },
  "timing": {
    "mode": "latency",
    "ms": { "mean": ..., "std": ..., "p50": ..., "p95": ..., "p99": ..., "min": ..., "max": ... },
    "samples": [/* len = --measure */]
  },
  "memory": {
    "max_memory_allocated_mb": ...,
    "max_memory_reserved_mb":  ...
  }
}
```

**版本演进规则**：新增字段不升 schema version；删除/改义字段 → `1.1`。

---

## 5. 函数级地图

| 函数 | 职责 | 是否在 timing 内 |
|------|------|----------------|
| `build_arg_parser` / `validate_args` | CLI + 契约验证 | ❌ |
| `resolve_script_version` | git sha + dirty 检测 | ❌ |
| `sha256_of_file` / `sha256_of_tensor` | hashing | ❌ |
| `collect_env_meta` | torch/cuda/python 版本 | ❌ |
| `build_model` | 构造 + 加载权重 + hash | ❌ |
| `build_input_tensor` | dummy/image + hash | ❌ |
| `_find_seg_components` | 定位 stem/stages/head | ❌ |
| `apply_plan_b_hooks` / `remove_hooks` | Plan B 注入/还原 | ❌ |
| `_find_litemla_modules` | 遍历找 LiteMLA 实例 | ❌ |
| `_make_patched_litemla_forward` | 构造 patched closure | ❌ |
| `apply_plan_c_monkey_patch` / `restore_plan_c_monkey_patch` | Plan C 注入/还原（实例级、幂等） | ❌ |
| `run_sanity_check` | 逐模块 patched vs original 对照 | ❌ |
| `maybe_profile_macs` | 可选 MACs | ❌ |
| **`measure_latency_per_iter`** | **主测时口径** | ✅ |
| `measure_throughput_batched` | 辅助测时 | ✅ |
| `collect_memory_stats` | peak memory | ❌（在 measure 之后） |
| `derive_default_out_path` / `save_json` | 输出 | ❌ |
| `_assemble_payload` | JSON 组装 | ❌ |
| `main` | orchestrator | — |

---

## 6. 用户 7 条实现约束 → 落实位置（约束追溯表）

| # | 约束摘要 | 落实位置（文件:符号） | 验证方式 |
|---|---------|-------------------|--------|
| 1 | 保存 `original_forward` 引用；幂等；可恢复 | `apply_plan_c_monkey_patch()` 中 `setattr(m, _PATCH_ORIG, ...)` + `_PATCH_FLAG` 检查 + `restore_plan_c_monkey_patch()` | smoke test: `--nvtx-level C` 跑两次同一进程，第二次应零增量 |
| 2 | 实例级 patch 优先 | `types.MethodType(...)` 绑定到实例；类的 `LiteMLA.forward` 全程**不动** | 检查脚本退出后 `LiteMLA.forward is <原始>` |
| 3 | git 路径处理稳健 | `resolve_script_version()`：先 `--show-toplevel` 再相对路径 | 在仓库外/无 git/未 commit 三种场景验证 fallback |
| 4 | sanity 只用 forward_hook，不用 pre_hook | `run_sanity_check()` 内只调用 `register_forward_hook` | code review |
| 5 | `no_grad` → `inference_mode` | warmup/measure/sanity/MACs 全部用 `with torch.inference_mode():` | `grep -n "no_grad" baseline_inference.py` → 0 命中 |
| 6 | hash / sanity / MACs 不混入 timing | main 函数严格顺序：build → hash(in build) → MACs → NVTX → warmup → measure | 函数级地图见 §5 |
| 7 | `weights_status="missing"` 不进入正式 JSON | `validate_args()` 在缺权重且无 smoke flag 时 `SystemExit` | smoke test |

---

## 7. 风险与已知坑

| 风险 | 触发条件 | 缓解 |
|------|--------|------|
| **MX250 2GB OOM** | 1024×2048 + MACs 同时开 | MACs 默认关闭；OOM 时降到 512×1024 |
| **`autocast(enabled=False)` 与 inference_mode 交互未测** | — | LiteMLA 内部 decorator 不依赖 grad，理论无冲突；运行时若报错回退 `no_grad` |
| **shape-adaptive 分支抖动**（`H*W > dim`） | 不同 input shape 切不同代码路径 | 固定分辨率，cudnn.benchmark=on 后稳态收敛即可 |
| **`measurement-mode=throughput` 与 NVTX 不太兼容** | enqueue 100 次后单 sync，Plan B/C 的 range 全部挤在一起 | 设计上：throughput 模式建议配 `--nvtx-level A` |
| **Plan-C sanity 在某些上下文偏移下假阳性** | `cudnn.benchmark` 在不同调用顺序下挑了不同 algo | sanity 在 patch 前后**同一进程**内连续跑，algo 一致；如出现假阳性，关 benchmark 重测 |
| **`torchprofile` 不支持的 op** | 自定义 op | best-effort，`status=error` 进 JSON，不阻塞 |

---

## 8. 使用速查

正式 baseline（Plan A，clean）：
```powershell
python phase1/scripts/baseline_inference.py `
  --weights phase1/weights/b0.pt `
  --resolution 1024 2048 `
  --nvtx-level A `
  --measurement-mode latency `
  --warmup 20 --measure 100
```

Nsight Plan B：
```powershell
nsys profile -o phase1/results/nsight/levelB `
  python phase1/scripts/baseline_inference.py `
  --weights phase1/weights/b0.pt --nvtx-level B
```

Nsight Plan C（含 sanity）：
```powershell
nsys profile -o phase1/results/nsight/levelC `
  python phase1/scripts/baseline_inference.py `
  --weights phase1/weights/b0.pt --nvtx-level C
```

Smoke test（无权重）：
```powershell
python phase1/scripts/baseline_inference.py `
  --allow-random-weights --warmup 2 --measure 5 --dry-run
```

详见 [`phase1/scripts/README.md`](../scripts/README.md)。

---

## 9. 后续演进 checklist

- [ ] 跑一次真实 Cityscapes b0 权重的 Plan A，确认 latency 落入合理区间（机器人场景 < 1s）
- [ ] 跑 Plan B，确认 7 个 stage range 在 Nsight UI 中可见
- [ ] 跑 Plan C，sanity 全部通过 + LiteMLA range 可见
- [ ] 写 `compare_baselines.py` 对多份 JSON 做表格化对比
- [ ] Phase 2 启动时，本设计文档要 cross-link 到 `phase2/design_notes/onnx_export_design.md`
- [ ] 若引入 FP16/AMP，**回到 §3.3 更新 sanity 阈值**

---

## 10. 变更日志

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0   | 2026-05-28 | 初版落盘，覆盖 7 条用户约束 + dual-run NVTX + JSON schema v1.0 |
| v1.0.1 | 2026-05-28 | 增加 §11 Triton stub 环境兼容层；§7 风险表新增一行；JSON `env` 块新增 `env_patches` / `triton_stubbed` 字段（仍在 schema 1.0 兼容范围内） |
| v1.0.2 | 2026-05-28 | **§11 大改**：stub 注入时机从"模块顶部立即"改为"`build_model()` 内延迟"；引入 `_FakeSymbol` 单例替代 `object`；新增 dunder 防御层（显式预填 `__file__/__path__/__loader__/__package__/__spec__`）；修正调用入口为真实函数名 `create_efficientvit_seg_model` + zoo key 格式 `efficientvit-seg-b0-cityscapes`；端到端 dry-run + smoke test 在 MX250 上验证通过 |

---

## 11. 环境兼容补丁 · Triton Stub

### 11.1 背景

上游 `efficientvit/models/nn/norm.py` 第 7 行无条件导入 `triton_rms_norm`：

```python
from efficientvit.models.nn.triton_rms_norm import TritonRMSNorm2dFunc
```

而 `triton_rms_norm.py` 顶层有：

```python
import triton
import triton.language as tl

@triton.jit
def _rms_norm_2d_fwd_fused(..., BLOCK_SIZE: tl.constexpr): ...
```

Windows 上没有官方 `triton` wheel（`triton-windows` 为社区 fork，且不支持 Pascal sm_61），该 import 链在到达 `efficientvit.seg_model_zoo.create_efficientvit_seg_model` 之前就会 `ModuleNotFoundError`。

**关键事实**：B0 推理路径 0 LayerNorm / 0 TritonRMSNorm（[architecture_analysis.md](../architecture_analysis.md) §结论 1），这个模块被 import 但从来不被调用。

### 11.2 方案选择

三选一：A 脚本内 stub 注入 / B 改上游加 try/except / C 装 triton-windows。采纳 **A**：

- 最小侵入：不动 `efficientvit/` 任何一行，保持 fork 与 upstream diff 干净
- 安全：B0 推理不会执行任何 Triton kernel，stub 仅负责 import 期调起
- 可透明：JSON 记录 `triton_stubbed=true`，下游人/机器可判定

### 11.3 stub 注入时机（**关键设计决策**）

**只能在 `import torch` 之后、`import efficientvit.*` 之前注入**——位置就在 `build_model()` 内、`from efficientvit.seg_model_zoo import ...` 这一行的前面，由一段 `global _TRITON_STUBBED; _TRITON_STUBBED = _install_triton_stub()` 完成。

**为什么不在脚本顶部立即注入？**

初版（v1.0.1）曾这样做。结果：PyTorch 在 `import torch` 期间就探测 `sys.modules["triton"]`，发现存在后**主动启用 triton 集成路径**——`torch.cuda._lazy_init` 调用 `_register_triton_kernels`、`torch._dynamo.utils` 读 `triton.language.dtype`、`torch._library.custom_op` 注册 `meta` 实现、`torch._prims.__init__` 注册 elementwise primitives……一系列原本只在"真有 triton"时才走的路径全被激活，最终在某个 `inspect.findsource → getsourcefile → filename.endswith(...)` 上崩。

**正确顺序**（端到端验证通过）：

```text
1. import torch           ← torch 看到无 triton → 选 fallback 路径
2. _install_triton_stub() ← sys.modules["triton"] 此时才有 stub
3. import efficientvit.*  ← upstream `import triton` 命中 stub
```

代码层面：脚本顶部只**定义** `_install_triton_stub()`、`_FakeSymbol` 类、`_FAKE_TRITON_SYMBOL` 单例，并占位 `_TRITON_STUBBED: bool = False`。真正调用发生在 `build_model()` 内（标注了 `# === Triton stub injection ===` 注释块）。

### 11.4 stub 需要覆盖的符号集

import 期求值的符号（stub 必需提供）：

| 符号 | 使用处 | stub 实现 |
|------|--------|----------|
| `triton` | `import triton` | `types.ModuleType("triton")` |
| `triton.jit` | `@triton.jit` 装饰器（两处） | `_jit(fn=None, **kwargs)` 返回 `_FakeTritonKernel` |
| `triton.language` | `import triton.language as tl` | `types.ModuleType("triton.language")` |
| `triton.language.constexpr` | `BLOCK_SIZE: tl.constexpr` 函数注解 | `object`（注解只需可解析，不需可调用） |

运行期才求值的符号（`triton.cdiv` / `triton.next_power_of_2` / `tl.program_id` / `tl.load` / `tl.store` / `tl.float32` / `tl.arange` / `tl.zeros` / `tl.where` / `tl.sum` / `tl.sqrt`）**不**预填；它们都在 kernel 函数体内，B0 路径不会 launch。任何 launch 尝试都会被 `_FakeTritonKernel.__getitem__` 拦下来报错。

### 11.5 PEP 562 兜底层 + `_FakeSymbol` 单例

PyTorch / EfficientViT 在 import 期还会零散读取 `triton.language.dtype`、`triton.cdiv` 等未预填的符号。为避免每次新增一个枚举一个，stub 在模块上装一个 PEP 562 `__getattr__`：

```python
def _stub_getattr(name: str):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)        # 见 §11.6
    return _FAKE_TRITON_SYMBOL
```

**为什么返回 `_FakeSymbol` 单例，不返回 `object` 类型？**

初版（v1.0.1）返回 `object`（类型本身）。结果：当某个属性被当作字符串使用时（典型场景：`inspect.getsourcefile` 里 `filename.endswith(s)`），`object.endswith` 会 `AttributeError`，进而触发更深的崩溃，**错误位置漂移到 CPython std-lib 内部**，调试痛苦。

`_FakeSymbol` 设计目标：

- `__getattr__(self, name) -> self`：让链式属性访问 `triton.language.foo.bar.baz` 一路返回自己，import 期不抛错。
- `__call__(*args, **kwargs) -> raise RuntimeError(...)`：被当函数调时**响亮崩**，错误文案点明 "stub 只做 import"。
- `__getitem__(item) -> raise RuntimeError(...)`：被当 kernel launch 时同样崩。
- `__repr__ -> "<FakeTritonSymbol import-only>"`：debug print 一眼可识别。

### 11.6 dunder 防御层（最容易被坑的一层）

PEP 562 `__getattr__` 只在普通查找失败时才被调用。但 `types.ModuleType` 默认**没有** `__file__` / `__path__` / `__loader__` 等模块元数据 dunder——这些读取会 fallback 到 PEP 562。一旦 `_stub_getattr("__file__")` 返回 `_FAKE_TRITON_SYMBOL`：

```python
# CPython inspect.py:776
if getattr(object, '__file__', None):   # _FAKE_TRITON_SYMBOL is truthy
    return object.__file__              # returns _FAKE_TRITON_SYMBOL
# 上层:
filename.endswith(s)                    # _FAKE_TRITON_SYMBOL.endswith → self
                                        # 然后被当函数调 → __call__ → RuntimeError
```

错误位置漂移到 `torchvision._meta_registrations` 的 `register_fake` 内部，stack 很难读。

**修复方案**（双层防御）：

1. **显式预填 dunder**（在 `_stub_getattr` 安装之前）：
   ```python
   stub.__file__ = None
   stub.__path__ = []
   stub.__loader__ = None
   stub.__package__ = "triton"
   stub.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
   ```
   有了真实属性后，`getattr(stub, "__file__", None)` 不再 fallback 到 `__getattr__`，inspect 看到 `__file__ is None` → 进 `raise TypeError('built-in module')` 分支 → 被 `getmodule` 的 try/except 吞掉 → 流程顺利。

2. **`_stub_getattr` 对未预填的 dunder 抛 `AttributeError`**：任何 dunder（`name.startswith("__") and name.endswith("__")`）只要没有显式预填，就清干净地抛 `AttributeError(name)`，让 std-lib 走它的"模块没这个能力"分支。**禁止**让 dunder fallback 到 `_FAKE_TRITON_SYMBOL`，否则就是上面那个漂移崩。

### 11.7 "响亮崩"设计

为什么 `triton.jit` 不能简单地 `return fn`（透传）？

- 透传 = 被 `@triton.jit` 装饰的函数变成普通 Python 函数，你可以 `_rms_norm_2d_fwd_fused(...)` 裸调。但 triton kernel 是用 `_rms_norm_2d_fwd_fused[(grid,)](...)` 调用的（需要 `__getitem__`），透传函数没有 `__getitem__` 会 `TypeError: 'function' object is not subscriptable`——错误位置隐藏于调用点，难调。
- `_FakeTritonKernel` 方案。提供 `__getitem__(grid)` 返回 `_launch`，`_launch(...)` 抛 RuntimeError。错误位置准确、文案明确指出 "B0 不应调用"。

错误文案原文：

```text
Triton is unavailable in this Windows environment.
This stub exists only to import EfficientViT modules;
TritonRMSNorm must not be executed for EfficientViT-Seg-B0
(see phase1/architecture_analysis.md conclusion #1: B0 has
ZERO LayerNorm / TritonRMSNorm). If you see this error,
you are running a NON-B0 path on a Triton-less machine.
```

### 11.8 可观测性

JSON `env` 块新增两个字段：

```json5
{
  "env": {
    "...": "...",
    "env_patches": ["triton_stub"],    // 未来可能增加其他补丁
    "triton_stubbed": true              // 布尔快查
  }
}
```

如果未来某台机器（如 Jetson Orin Linux）装了真的 triton，`_install_triton_stub()` 会检测到并 `return False`，`triton_stubbed=false`。两份 JSON 可以互相对比。

### 11.9 上游入口函数与 zoo key 格式

实际验证后确认上游真实入口是 `create_efficientvit_seg_model`（**不是** v1.0.1 文档里曾写的 `create_seg_model`），签名为：

```python
create_efficientvit_seg_model(name: str, pretrained: bool = True,
                              weight_url: Optional[str] = None, **kwargs)
```

且 `name` 必须是 zoo 注册表里的 key，格式为：

```
efficientvit-seg-{variant}-{dataset}
```

例如 `"efficientvit-seg-b0-cityscapes"`。脚本在 `build_model()` 里用 `f"efficientvit-seg-{args.model}-{args.dataset}"` 拼出 key，并写入 JSON 的 `weights.zoo_key` 字段以便回溯。

`--allow-random-weights` 映射为 `pretrained=False`（跳过 `load_state_dict_from_file`）；`--weights <path>` 映射为 `pretrained=True, weight_url=<path>`（强制加载该路径，覆盖默认的 `assets/checkpoints/...`）。

### 11.10 端到端验证（v1.0.2）

| 检查 | 命令 | 结果 |
|------|------|------|
| 语法 | `python -m py_compile phase1/scripts/baseline_inference.py` | ✅ PASS |
| Dry-run | `... --allow-random-weights --resolution 512 1024 --dry-run` | ✅ exit 0, model/input 构建成功 |
| Smoke test | `... --allow-random-weights --resolution 512 1024 --warmup 3 --measure 5` | ✅ exit 0, latency mean=23.9ms, JSON 落盘 |
| JSON schema | 读取上述 smoke JSON | ✅ 全字段齐全，`triton_stubbed=true`, `status="ok"`, `is_smoke_test=true` |
| GPU 内存 | smoke run 中观察 | ✅ peak 575MB / reserved 614MB（MX250 2GB 还有 ~70% 余量） |

### 11.11 未来迁移点

当项目迁到 Jetson Orin（Linux + aarch64 + sm_87）时：

- 官方 triton wheel 应可用，`_install_triton_stub()` 自动 no-op（检测到 `sys.modules["triton"]` 已存在则 `return False`）
- 本 stub 代码仍保留，不需删除（对实装环境零开销、零影响）
- 可考虑抽取到 `phase1/scripts/_compat.py` 供 Phase 2 的 `export_onnx.py` 复用
