# P1a Stage2+Stage3 All-Context Design

> 目的：在进入更复杂的 P1mix（stage2=P1b-7、stage3=P1a）之前，先验证当前最稳定的 P1a-3b `relu_linear_att-only` Plugin 是否可以同时作用于 `stage2/context` 和 `stage3/context` 四个 LiteMLA block。

## 1. 为什么先做这个实验

当前已经确认：

- P1a-3b 是目前最稳定的最小边界，替换 `relu_linear_att-only`，不接管 aggregation / cat / proj / residual。
- P1b-7 在 `stage2/context` 上有更大的中段覆盖范围，但它的 CUDA 实现依赖 stage2 contract：`qkvC=192`、`outputC=128`、`H=64`、`W=128`、groups=12。
- stage3 的 LiteMLA 仍然是同一种 `relu_linear_att` 数学结构，但 context block 的张量尺寸变为更小的 `[1,768,32,64] -> [1,256,32,64]`。

因此先做 P1a stage2+stage3 是更干净的消融：

1. 不改变 Plugin 数学边界，只扩大相同边界覆盖范围。
2. 可以判断 stage3 的 `relu_linear_att` 是否也值得接入 Plugin。
3. 可以为后续 P1mix 提供参照，避免把“覆盖更多 block”和“换成 P1b 边界”两个变量混在一起。

## 2. 替换范围

新增 `integrate_relu_linear_attention_plugin_onnx.py --target-scope stage2-stage3`。

| Scope | Block | Plugin input | Plugin output | attrs |
|---|---|---|---|---|
| stage2 | `/backbone/stages.2/op_list.1/context_module/main` | `Concat_output_0` | `Cast_1_output_0` | `input_c=384, height=64, width=128, dim=16` |
| stage2 | `/backbone/stages.2/op_list.2/context_module/main` | `Concat_output_0` | `Cast_1_output_0` | `input_c=384, height=64, width=128, dim=16` |
| stage3 | `/backbone/stages.3/op_list.1/context_module/main` | `Concat_output_0` | `Cast_1_output_0` | `input_c=768, height=32, width=64, dim=16` |
| stage3 | `/backbone/stages.3/op_list.2/context_module/main` | `Concat_output_0` | `Cast_1_output_0` | `input_c=768, height=32, width=64, dim=16` |

## 3. 预期与风险

- 预期：Plugin node 数量从 2 个增加到 4 个；TensorRT parser/build 应该仍能接受，因为 Plugin field 只改变 `input_c/height/width`。
- 风险：stage3 spatial size 更小，单个 Plugin 的 kernel launch 固定开销占比更高，端到端收益未必线性增加。
- 验收：必须满足 `allclose_pass=true`，再看 p50/p95 是否相对 Phase 2 TensorRT baseline 和 stage2-only P1a 有净收益。

## 4. 结果记录

2026-06-16 冷机重测结果：

| Item | Result |
|---|---:|
| Plugin node count | 4 |
| TensorRT build | ok |
| Baseline TRT p50 | `54.3995 ms` |
| Plugin TRT p50 | `50.8380 ms` |
| p50 delta | `-3.5615 ms` |
| p50 speedup | `1.0701x` |
| Baseline TRT mean | `54.4019 ms` |
| Plugin TRT mean | `50.9334 ms` |
| mean speedup | `1.0681x` |
| Plugin vs baseline allclose | `True` |
| Plugin vs baseline max abs diff | `4.48227e-05` |
| Plugin vs baseline argmax agreement | `1.000000` |

Artifacts:

- [`../results/metrics/relu_linear_attention_plugin_stage2_stage3_onnx_integration.json`](../results/metrics/relu_linear_attention_plugin_stage2_stage3_onnx_integration.json)
- [`../results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_build.json`](../results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_build.json)
- [`../results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_benchmark.json`](../results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_benchmark.json)
- [`../results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_benchmark_summary.md`](../results/metrics/relu_linear_attention_plugin_stage2_stage3_engine_benchmark_summary.md)
- [`../results/metrics/relu_linear_attention_plugin_stage2_stage3_nsys_attribution_summary.md`](../results/metrics/relu_linear_attention_plugin_stage2_stage3_nsys_attribution_summary.md)

Nsight attribution:

| Item | Result |
|---|---:|
| `trt/execute` kernel avg | `50.680 ms / iter` |
| `trt/execute` launches | `163.0 / iter` |
| selected context total | `6.436 ms / iter` |
| selected context launches | `92.0 / iter` |
| `relu_linear_att_plugin` total | `1.950 ms / iter` |
| stage2 Plugin layers | `0.655 + 0.654 ms / iter` |
| stage3 Plugin layers | `0.320 + 0.321 ms / iter` |

Interpretation:

1. `stage2+stage3` P1a 是有效扩大覆盖范围：同一 `relu_linear_att-only` Plugin 从 2 个 stage2 block 扩展到 4 个 LiteMLA context block 后，冷机 p50 speedup 从 stage2-only P1a-3b 的约 `1.043x` 提升到 `1.070x`。
2. stage3 block 虽然 feature map 更小，但仍然有足够的 residual attention runtime，值得用 P1a 覆盖。
3. 这次实验不改变 Plugin 边界，不接管 aggregation / cat / proj / residual，因此结论只支持“P1a 可扩到 stage3”，不能替代 P1b 或整体 LiteMLA 的边界判断。
4. 热机样本曾出现 baseline p50 `54.449 ms`、plugin p50 `54.465 ms`、speedup `1.000x`；因此 MX250 上 1-4ms 级收益必须坚持冷机、单任务、同口径复测后再写入结论。
