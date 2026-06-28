# P1mix: Stage2 P1b + Stage3 P1a Design

> 目的：验证 `stage2=P1b-7 aggregation+cat+relu_linear_att` 与 `stage3=P1a-3b relu_linear_att-only` 的组合是否优于当前最强参照 `P1a-3b stage2+stage3`。

## 1. 为什么做 P1mix

当前有效结果：

| Experiment | Baseline TRT p50 | Plugin TRT p50 | p50 speedup |
|---|---:|---:|---:|
| P1a stage2-only | `54.394 ms` | `52.168 ms` | `1.043x` |
| P1b-7 stage2-only | `54.380 ms` | `52.311 ms` | `1.040x` |
| P1a stage2+stage3 | `54.3995 ms` | `50.8380 ms` | `1.070x` |

因此 P1mix 的验收线必须是：

```text
P1mix p50 < 50.838 ms
```

如果 P1mix 只优于旧的 stage2-only P1a/P1b，但不优于 `P1a-3b stage2+stage3`，就不应作为新主线。

## 2. 替换范围

P1mix 由两步 ONNX surgery 组成：

1. 在 Phase 2 ONNX 上先应用 P1b stage2 surgery：
   - `/backbone/stages.2/op_list.{1,2}/context_module/main`
   - input: `qkv/conv/Conv_output_0`
   - output: `Cast_1_output_0`
   - Plugin: `EdgesegAggregationReluLinearAttention_TRT`

2. 在上述 ONNX 上再应用 P1a stage3 surgery：
   - `/backbone/stages.3/op_list.{1,2}/context_module/main`
   - input: `Concat_output_0`
   - output: `Cast_1_output_0`
   - Plugin: `EdgesegReluLinearAttention_TRT`

这样可以保持变量清晰：

- stage2 使用已知最强 P1b-7 边界；
- stage3 使用已验证有效的 P1a-3b 边界；
- 不引入 stage3 P1b shape/group/shared-memory 新风险。

## 3. 风险

- 同一个 engine 中同时包含两个 Plugin Creator，build/benchmark 必须显式检查两者都已注册。
- P1b-7 自写 aggregation 对 stage2 有价值，但端到端是否能叠加 stage3 P1a 收益需要实测。
- 如果 P1mix 不优于 `50.838 ms`，当前 Phase 3 MVP 应继续保留为 `P1a-3b stage2+stage3`。

## 4. 结果记录

2026-06-16 运行结果：

| Item | Result |
|---|---:|
| P1b Plugin nodes | 2 |
| P1a Plugin nodes | 2 |
| TensorRT build | ok |
| `both` baseline TRT p50 | `55.2637 ms` |
| `both` P1mix p50 | `57.2959 ms` |
| `both` p50 speedup | `0.9645x` |
| P1mix plugin-only p50 | `50.674 ms` |
| P1a stage2+stage3 plugin-only p50 | `50.769 ms` |
| Plugin vs baseline allclose | `True` |
| Plugin vs baseline argmax agreement | `1.000000` |

Nsight attribution:

| Item | P1a stage2+stage3 | P1mix |
|---|---:|---:|
| `trt/execute` kernel avg | `50.680 ms` | `50.784 ms` |
| `trt/execute` launches | `163.0` | `146.0` |
| selected context total | `6.436 ms` | `6.624 ms` |
| selected context launches | `92.0` | `71.0` |
| total Plugin layer time | `1.950 ms` | `3.737 ms` |
| stage2 context total | `4.033 ms` | `4.138 ms` |

Artifacts:

- [`../results/metrics/p1mix_stage2_p1b_onnx_integration.json`](../results/metrics/p1mix_stage2_p1b_onnx_integration.json)
- [`../results/metrics/p1mix_stage3_p1a_onnx_integration.json`](../results/metrics/p1mix_stage3_p1a_onnx_integration.json)
- [`../results/metrics/p1mix_stage2_p1b_stage3_p1a_engine_build.json`](../results/metrics/p1mix_stage2_p1b_stage3_p1a_engine_build.json)
- [`../results/metrics/p1mix_stage2_p1b_stage3_p1a_engine_benchmark.json`](../results/metrics/p1mix_stage2_p1b_stage3_p1a_engine_benchmark.json)
- [`../results/metrics/p1mix_stage2_p1b_stage3_p1a_engine_benchmark_plugin_only.json`](../results/metrics/p1mix_stage2_p1b_stage3_p1a_engine_benchmark_plugin_only.json)
- [`../results/metrics/p1mix_stage2_p1b_stage3_p1a_nsys_attribution_summary.md`](../results/metrics/p1mix_stage2_p1b_stage3_p1a_nsys_attribution_summary.md)

结论：

1. P1mix build 和 correctness 成立，说明一个 TensorRT engine 中同时使用 P1b 与 P1a 两类 Plugin 是可行的。
2. P1mix 没有稳定优于 `P1a-3b stage2+stage3`。plugin-only p50 只快 `0.095 ms`，低于当前 MX250 的可信差异阈值；Nsight execute kernel avg 反而略慢 `0.104 ms`。
3. P1mix 减少了 launch 数（`163 -> 146`），但 P1b stage2 Plugin layer 边界更重，selected context total 从 `6.436 ms` 增到 `6.624 ms`。
4. 因此当前主线仍应保留为 `P1a-3b stage2+stage3` 两阶段 FP32 Plugin。P1mix 记录为 `evaluated, not adopted as mainline`，除非后续能显著优化 P1b stage2 kernel。
