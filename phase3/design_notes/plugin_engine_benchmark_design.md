# Plugin Engine Benchmark Design

> **状态**：Phase 3 Step 7 落盘设计。
>
> **目标**：在同一套输入、同一测量协议下，对比 Phase 2 TensorRT FP32 baseline engine 与 Phase 3 Plugin FP32 engine，判断 `relu_linear_att-only` Plugin 接入真实 EfficientViT graph 后的端到端正确性和 latency 净收益。

---

## 1. 设计目标

Step 7 回答三个问题：

1. Plugin engine 能否在运行时正确加载 Plugin DLL 并成功反序列化。
2. Plugin engine 输出是否仍与 PyTorch reference / Phase 2 TensorRT baseline 对齐。
3. Plugin 替换两个 `stage2/context` `relu_linear_att` 子路径后，端到端 latency 是改善、持平还是退化。

本步骤不做 Nsight attribution。runtime kernel 归因属于 Step 8。

---

## 2. 比较口径

| 对象 | 含义 |
|---|---|
| Phase 2 baseline engine | 原始 ONNX 经 TensorRT parser/build 自动优化得到的 FP32 engine |
| Phase 3 Plugin engine | patched ONNX 经 TensorRT parser/build 自动优化得到的 FP32 engine，其中两个 `stage2/context` `relu_linear_att` 子路径被 Plugin node 替换 |

因此 Plugin engine 不是“在 baseline engine 上额外加速”，而是：

```text
Plugin 之外：继续由 TensorRT 自动优化
Plugin 之内：由自定义 CUDA kernel 实现，TensorRT 不再拆解内部子图
```

最终结论必须看端到端净收益，而不是只看 Plugin 单层 microbenchmark。

---

## 3. 计时策略

Step 7 沿用 Phase 2 benchmark 口径：

- 输入 / 输出 GPU buffer 在计时前分配。
- H2D / D2H / preprocess 不进入计时。
- 每次 iteration 用 CUDA Events 包住一次 `context.execute_async_v2()`。
- 默认 `warmup=20`、`measure=100`。
- 统计 mean/std/min/max/p50/p95/p99。

这保持了 Phase 1/2/3 的 latency 数字可比性。

---

## 4. 输出对齐策略

脚本记录三组 comparison：

| comparison | 目的 |
|---|---|
| PyTorch reference vs TensorRT baseline | 复核 Phase 2 baseline engine 仍可用 |
| PyTorch reference vs Plugin engine | 验证 Plugin engine 没破坏模型输出 |
| TensorRT baseline vs Plugin engine | 直接判断 Plugin 替换是否改变 TensorRT 端输出 |

每组记录：

- `max_abs_diff`
- `mean_abs_diff`
- `max_rel_diff`
- `cosine_similarity`
- relaxed allclose
- argmax pixel agreement

若 Plugin engine correctness 不通过，latency 只作为调试信息，不作为优化结论。

---

## 5. 实现取舍

### D1：是否修改 Phase 2 benchmark 脚本

选择：新增 `phase3/scripts/benchmark_plugin_engine.py`，不直接修改 `phase2/scripts/benchmark_trt_engine.py`。

原因：

- Phase 2 脚本代表纯 TensorRT baseline，应该保持语义稳定。
- Phase 3 脚本需要加载 Plugin DLL 和检查 Plugin creator，职责不同。
- 新脚本仍复用 Phase 2 的 binding allocation / CUDA Event timing 等函数，避免重复实现核心计时逻辑。

### D2：是否同进程跑 baseline 与 Plugin

选择：同一脚本、同一进程顺序跑两个 engine。

原因：

- 输入、PyTorch reference、warmup/measure、CUDA Event 计时口径完全一致。
- 输出 JSON 可以直接给出 speedup / regression ratio。
- 避免后续靠手工拼接两份 JSON。

### D3：是否加载 Plugin DLL 后再跑 baseline engine

选择：是。

原因：

- Plugin DLL 对 baseline engine 无影响。
- 同一进程统一 runtime path 和 TensorRT logger，更容易复现。
- Plugin engine 反序列化前必须注册 Plugin creator，否则 TensorRT runtime 无法恢复自定义层。

---

## 6. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| Plugin DLL 未加载或 creator 未注册 | Plugin engine 反序列化失败 | metadata 记录 Plugin DLL 路径、sha256、creator 查找结果 |
| Plugin engine 输出不对齐 | 不能进入性能结论 | 同时比较 PyTorch、baseline TRT、Plugin TRT |
| 顺序测量带来 GPU 频率偏置 | speedup 有轻微误差 | 默认 20 warmup + 100 measure；必要时可反向顺序复跑 |
| Plugin 阻断 TensorRT 内部优化 | latency 可能退化 | 这是 Step 7 需要显式暴露的真实结果 |

---

## 7. 通过条件

Step 7 通过条件：

1. baseline engine 和 Plugin engine 均可反序列化并执行。
2. Plugin engine 输出与 PyTorch reference / baseline TRT 输出在约定阈值内对齐，或明确记录不通过。
3. JSON 写入 `phase3/results/metrics/relu_linear_attention_plugin_engine_benchmark.json`。
4. README 记录 Step 7 的实际结果和下一步 Step 8 是否需要继续 Nsight attribution。

---

## 8. 实测结果

Step 7 已按本设计执行，正式命令使用 `warmup=20`、`measure=100`，输入为 `phase1/data/city_asset_cityscapes_like.png`，权重为 `phase1/weights/efficientvit_seg_b0_cityscapes.pt`。

| 指标 | Phase 2 TensorRT FP32 baseline | Phase 3 Plugin FP32 engine |
|---|---:|---:|
| P0 p50 latency (`both`) | 54.3877 ms | 53.2234 ms |
| P0 mean latency (`both`) | 54.4337 ms | 53.2354 ms |
| P1a-1c p50 latency (`both`, baseline -> plugin) | 54.4988 ms | 55.8372 ms |
| P1a-1c p50 latency (standalone probe) | 54.503 ms | 53.109 ms |

结论：

- P0 shared-memory VK cache 后，`both` 口径 p50 speedup 为 `1.0219x`，Plugin engine 比 baseline 快约 `1.1643 ms`。
- P1a-1c 后，单独进程 probe 显示 baseline-only p50 `54.503 ms`、plugin-only p50 `53.109 ms`，但同进程 `baseline -> plugin` 的 `both` 口径显示 speedup `0.9760x`。
- Plugin TRT vs baseline TRT `allclose=True`，P1a-1c `both` run 中 `max_abs_diff=7.24792e-05`，argmax pixel agreement 为 `1.0`。
- Plugin TRT vs PyTorch 严格 `1e-4` allclose 未通过，但 `1e-3` relaxed allclose 通过，argmax pixel agreement 为 `1.0`，与 Phase 2 的 TensorRT 数值误差口径一致。

因此 Step 7 可以判定为通过：Plugin engine 已经进入真实整网并保持输出对齐；但 1ms 量级端到端差异对执行顺序和 GPU 频率状态敏感，不能只用单次 `both` run 宣称稳定整网加速。P1a-1c 的性能判断主要依赖 Step 8 plugin-only Nsight attribution。
