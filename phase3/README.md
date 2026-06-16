# Phase 3 — TensorRT LiteMLA Plugin

> **阶段目标**：在 Phase 1 PyTorch/Nsight attribution 与 Phase 2 TensorRT baseline 的证据基础上，设计并实现一个面向 EfficientViT `stage2/context` LiteMLA 的 TensorRT Plugin MVP，验证自定义 C++/CUDA/TensorRT Plugin 是否能进一步优化 TensorRT 未自动整体融合的非标准线性注意力路径。
>
> **当前状态**：Phase 3 已完成 P1a Plugin skeleton、Step 5 单层 CUDA 数学验证、Step 5.5 单层 microbenchmark / Nsight 记录、Step 6 真实 EfficientViT ONNX graph replacement + Plugin engine build、Step 7 端到端 correctness / latency 对比、Step 8 Plugin engine Nsight attribution、P1a-4 单 kernel 合并可行性评估，以及 P1b skeleton / parser toy / 真实 EfficientViT ONNX surgery build smoke。Plugin DLL 可编译，TensorRT registry 能找到 P1a/P1b 两个 Creator，toy engine 和真实 P1a/P1b Plugin engine 均可构建；P1a Plugin engine 与 Phase 2 TensorRT baseline 输出对齐，端到端 p50 有轻微净收益；Nsight 证明 P1a 替换减少了目标边界 kernel time 和 launch 数，但整网收益被其他标准算子热点稀释。P1b naive 版 CUDA 数学路径已证明 block-level correctness，但 Nsight 定位到自写 `depthwise5x5Kernel` 与 `groupedPointwise1x1Kernel` 合计约 `4.889ms/iter`，导致端到端性能退化。P1b-1 fused aggregation+cat 版已把中段边界改善到 `4.848ms/iter`、`6 launches/iter`；P1b-2 缓存 grouped pointwise `16x16` 权重到 shared memory，使 P1b Plugin layer 降到 `4.347ms/iter`；P1b-4 进一步缓存 depthwise row tile / halo 到 shared memory，使 P1b Plugin layer 降到 `3.574ms/iter`；P1b-5 将 depthwise tile channel chunk 从 4 扩到 8，使 P1b Plugin layer 降到 `3.241ms/iter`；P1b-7 将 CTA 从 `2x128` spatial tile 改为 `4x128` spatial tile，使 P1b Plugin layer 进一步降到 `3.043ms/iter`、`6 launches/iter`，对比 Phase 2 baseline `aggregation + attention_core` 的 `5.443ms/iter`、`38 launches/iter` 达到 `1.789x` kernel-time speedup。冷机端到端 p50 为 baseline `54.380ms` vs P1b-7 `52.311ms`，speedup `1.040x`。P1b-8/9/10/11a 均已评估但不采纳，当前主线仍保持 P1b-7。当前 P1b 已证明“扩大边界 + 针对 MX250 的 shared-memory 数据复用”可以带来可见收益。

---

## 1. 阶段边界

Phase 3 做：

- 设计 LiteMLA Plugin 的输入输出 tensor contract。
- 先实现最小可行 Plugin MVP，优先验证 `relu_linear_att-only` 的接入链路；`aggregation-only` 保留为 fallback / 对照实验。
- 将 Plugin 集成到 TensorRT engine 构建 / runtime 路径中。
- 复用 Phase 2 benchmark 与 C++ runtime demo 验证 correctness、latency 和 Nsight attribution。
- 在成功 MVP 基础上评估 `aggregation + cat + relu_linear_att` 中段组合边界。

Phase 3 暂不做：

- 不一开始实现整体 LiteMLA Plugin。
- 不把 stage0/head MBConv 作为第一主线。
- 不做完整 ROS 2 节点。
- 不把 INT8 / 混合精度作为第一交付物。

---

## 2. 输入证据

| 证据 | 文件 | 对 Phase 3 的作用 |
|---|---|---|
| Phase 1 bottleneck report | [`../phase1/bottleneck_analysis_report.md`](../phase1/bottleneck_analysis_report.md) | 给出 PyTorch 路径下的 Plugin 候选边界 |
| Phase 1 Plan D attribution | [`../phase1/results/metrics/planD_nsys_attribution_summary.md`](../phase1/results/metrics/planD_nsys_attribution_summary.md) | 证明 `aggregation` / `relu_linear_att` / 中段组合的耗时结构 |
| Phase 2 TensorRT report | [`../phase2/tensorrt_baseline_report.md`](../phase2/tensorrt_baseline_report.md) | 复核 TensorRT 后 LiteMLA residual runtime |
| TensorRT engine inspection | [`../phase2/results/metrics/trt_engine_inspection_summary.md`](../phase2/results/metrics/trt_engine_inspection_summary.md) | 说明 TensorRT 未把 LiteMLA 自动融合成单一算子 |
| TensorRT C++ demo | [`../phase2/cpp_demo/README.md`](../phase2/cpp_demo/README.md) | 作为后续 Plugin engine runtime 验证起点 |
| Stage2 tensor contract | [`design_notes/stage2_context_tensor_contract.md`](design_notes/stage2_context_tensor_contract.md) | 确认 P1a/P1b/P1c 的真实输入输出 shape 与替换边界 |
| Plugin API / CMake design | [`design_notes/plugin_api_cmake_design.md`](design_notes/plugin_api_cmake_design.md) | 确认第一版 Plugin 接口、序列化字段、DLL 构建与加载策略 |
| Plugin toy engine build | [`results/metrics/relu_linear_attention_toy_build.json`](results/metrics/relu_linear_attention_toy_build.json) | 证明 Step 4 skeleton DLL 可注册并能构建含 Plugin 的 toy engine |
| Plugin 单层验证 | [`results/metrics/relu_linear_attention_plugin_validation.json`](results/metrics/relu_linear_attention_plugin_validation.json) | 证明 Step 5 CUDA kernel 在 toy/plugin 单层层面与 PyTorch `relu_linear_att` reference 对齐 |
| Plugin 单层 microbenchmark | [`results/metrics/relu_linear_attention_plugin_microbenchmark_summary.md`](results/metrics/relu_linear_attention_plugin_microbenchmark_summary.md) | 记录 Step 5.5 单层 latency、Nsight kernel summary 与继续 Step 6 的边界判断 |
| Plugin kernel 优化历程 | [`design_notes/plugin_kernel_optimization_history.md`](design_notes/plugin_kernel_optimization_history.md) | 集中记录 P1a 与 P1b kernel 演进：P1a v0/P0/P1a-1/P1a-3、P1b naive/P1b-1 fused aggregation+cat/P1b-2 shared pointwise weight cache 的关键指标变化和下一步瓶颈 |
| P1a single-kernel feasibility | [`design_notes/p1a_single_kernel_feasibility.md`](design_notes/p1a_single_kernel_feasibility.md) | 评估 P1a-4 两阶段合并为单 kernel 的同步、并行度和收益风险，决定不作为下一主线 |
| P1b aggregation + attention design | [`design_notes/p1b_aggregation_attention_design.md`](design_notes/p1b_aggregation_attention_design.md) | 确定 P1b `aggregation + cat + relu_linear_att` 的替换边界、权重输入方式、parser/build 风险与验证顺序 |
| P1b single-block validation design | [`design_notes/p1b_single_block_validation_design.md`](design_notes/p1b_single_block_validation_design.md) | 固定 P1b block-level PyTorch reference、tensor/weight 捕获、数值指标和后续 Plugin correctness 验收口径 |
| P1b stage2 reference capture | [`results/metrics/p1b_stage2_reference_capture.json`](results/metrics/p1b_stage2_reference_capture.json) | 证明两个真实 `stage2/context` block 的 qkv、aggregation、cat、attention output、权重 shape/group 与 reference 口径均可复现 |
| P1b parser toy build | [`results/metrics/p1b_aggregation_attention_toy_build.json`](results/metrics/p1b_aggregation_attention_toy_build.json) | 证明 P1b skeleton Plugin Creator 可注册，带两个 aggregation weight initializer 的 custom op 可被 TensorRT ONNX parser/build 接受 |
| P1b real graph ONNX integration | [`results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json`](results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json) | 证明两个真实 `stage2/context` 的 `qkv/conv/Conv_output_0 -> Cast_1_output_0` 子图可替换为 P1b Plugin node |
| P1b real graph engine build | [`results/metrics/p1b_aggregation_attention_plugin_engine_build.json`](results/metrics/p1b_aggregation_attention_plugin_engine_build.json) | 证明 P1b patched ONNX 可被 TensorRT parser 解析并构建真实 skeleton Plugin engine |
| P1b block-level Plugin validation | [`results/metrics/p1b_aggregation_attention_plugin_validation.json`](results/metrics/p1b_aggregation_attention_plugin_validation.json) | 证明 P1b 第一版 CUDA 数学路径在两个真实 `stage2/context` block 上与 PyTorch `attention_out` reference 对齐 |
| P1b naive real engine benchmark | [`results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.md) | 定位 naive P1b 退化来源：Plugin layer `6.189ms/iter`，其中自写 depthwise/grouped pointwise aggregation kernel 合计约 `4.889ms/iter` |
| P1b-1 fused benchmark | [`results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md) | 证明 fused aggregation+cat 后真实 P1b engine 与 baseline TRT 输出对齐；p50 `53.610ms` vs baseline `54.331ms`，端到端小幅正收益 |
| P1b-1 fused Nsight attribution | [`results/metrics/p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.md) | fused P1b 中段边界 `4.848ms/iter`、`6 launches/iter`，对比 Phase 2 baseline `aggregation + attention_core` `5.443ms/iter`、`38 launches/iter`，kernel-time speedup `1.123x` |
| P1b-2 shared weight benchmark | [`results/metrics/p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_summary.md) | 证明 grouped pointwise 权重 shared-memory cache 后真实 P1b engine 与 baseline TRT 输出对齐；冷机 p50 `53.530ms` vs baseline `54.306ms`，端到端小幅正收益 |
| P1b-2 shared weight Nsight attribution | [`results/metrics/p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.md) | P1b-2 Plugin layer `4.347ms/iter`、`6 launches/iter`，其中 `fusedAggregationCatKernel` 从 P1b-1 的 `3.536ms/iter` 降到 `3.038ms/iter`；中段边界相对 Phase 2 baseline 达到 `1.252x` kernel-time speedup |
| P1b-3 interior fast path probe | [`results/metrics/p1b_aggregation_attention_plugin_interior_fastpath_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_interior_fastpath_engine_benchmark_summary.md) | 证明单纯去掉非边界像素的 depthwise 5x5 越界判断没有收益；冷机 p50 `54.710ms` vs baseline `54.312ms`，speedup `0.9927x`，因此不采纳 |
| P1b-4 depthwise tile benchmark | [`results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_engine_benchmark_summary.md) | 证明 depthwise row-tile shared cache 后真实 P1b engine 与 baseline TRT 输出对齐；冷机 p50 `52.725ms` vs baseline `54.411ms`，端到端 speedup `1.032x` |
| P1b-4 depthwise tile Nsight attribution | [`results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_nsys_attribution_summary.md) | P1b-4 Plugin layer `3.574ms/iter`、`6 launches/iter`，其中 `fusedAggregationCatKernel` 从 P1b-2 的 `3.038ms/iter` 降到 `2.262ms/iter`；中段边界相对 Phase 2 baseline 达到 `1.523x` kernel-time speedup |
| P1b-5 depthwise tile ch8 benchmark | [`results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_ch8_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_ch8_engine_benchmark_summary.md) | 证明 depthwise tile channel chunk 从 4 扩到 8 后真实 P1b engine 与 baseline TRT 输出对齐；冷机 p50 `52.455ms` vs baseline `54.297ms`，端到端 speedup `1.035x` |
| P1b-5 depthwise tile ch8 Nsight attribution | [`results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_ch8_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_ch8_nsys_attribution_summary.md) | P1b-5 Plugin layer `3.241ms/iter`、`6 launches/iter`，其中 `fusedAggregationCatKernel` 从 P1b-4 的 `2.262ms/iter` 降到 `1.926ms/iter`；中段边界相对 Phase 2 baseline 达到 `1.679x` kernel-time speedup |
| P1b-7 CTA512 benchmark | [`results/metrics/p1b_aggregation_attention_plugin_cta512_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_cta512_engine_benchmark_summary.md) | 证明一个 CTA 覆盖 4 行 spatial tile 后真实 P1b engine 与 baseline TRT 输出对齐；冷机 p50 `52.311ms` vs baseline `54.380ms`，端到端 speedup `1.040x` |
| P1b-7 CTA512 Nsight attribution | [`results/metrics/p1b_aggregation_attention_plugin_cta512_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_cta512_nsys_attribution_summary.md) | P1b-7 Plugin layer `3.043ms/iter`、`6 launches/iter`，其中 `fusedAggregationCatKernel` 从 P1b-5 的 `1.926ms/iter` 降到 `1.730ms/iter`；中段边界相对 Phase 2 baseline 达到 `1.789x` kernel-time speedup |
| P1b-9 pointwise accumulator probe | [`results/metrics/p1b_aggregation_attention_plugin_pointwise_accum_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_pointwise_accum_engine_benchmark_summary.md) | 证明把 grouped pointwise 16 个输出改为单线程内 16 个标量累加器没有收益；p50 `52.431ms`，慢于 P1b-7 的 `52.311ms`，因此不采纳 |
| P1b-10 pointwise accum4 probe | [`results/metrics/p1b_aggregation_attention_plugin_pointwise_accum4_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_pointwise_accum4_engine_benchmark_summary.md) | 证明折中为每次 4 个输出累加器仍无收益；p50 `54.197ms`，显著慢于 P1b-7，因此不采纳 |
| P1b-7 fused kernel nvprof | [`results/metrics/nvprof_p1b7_fused_summary.md`](results/metrics/nvprof_p1b7_fused_summary.md) | `fusedAggregationCatKernel` 不是纯 DRAM bandwidth-bound / occupancy-bound；`stall_exec_dependency ~= 53%`，更像 instruction dependency / scheduling 主导 |
| P1b-11a shared pitch probe | [`results/metrics/p1b_aggregation_attention_plugin_shared_pitch133_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_shared_pitch133_engine_benchmark_summary.md) | `kDepthwiseTileWidth=133` 的 p50 只比 P1b-7 快 `0.043ms`，且 `shared_efficiency` 几乎不变、global load transactions 略增，因此不采纳 |
| Plugin ONNX 集成 | [`results/metrics/relu_linear_attention_plugin_onnx_integration.json`](results/metrics/relu_linear_attention_plugin_onnx_integration.json) | 证明两个 `stage2/context` attention 子图已被替换成 Plugin node |
| Plugin engine build | [`results/metrics/relu_linear_attention_plugin_engine_build.json`](results/metrics/relu_linear_attention_plugin_engine_build.json) | 证明 patched ONNX 可被 TensorRT parser 解析并构建真实 Plugin engine |
| Plugin engine benchmark | [`results/metrics/relu_linear_attention_plugin_engine_benchmark_summary.md`](results/metrics/relu_linear_attention_plugin_engine_benchmark_summary.md) | 证明 Plugin engine 可执行、与 baseline TRT 输出对齐，并给出端到端 latency 净收益 |
| Plugin engine Nsight attribution | [`results/metrics/relu_linear_attention_plugin_nsys_attribution_summary.md`](results/metrics/relu_linear_attention_plugin_nsys_attribution_summary.md) | 证明 Plugin 替换后目标边界 kernel time / launch 数下降，并定位 Plugin 内部两个 kernel 的占比 |

---

## 3. Plugin 候选排序

| 优先级 | 候选 | 目标 | 当前判断 |
|---|---|---|---|
| P1a | `relu_linear_att-only` | MVP / 接入验证 | Step 2 已确定真实 contract 为 `[1,384,64,128] -> [1,128,64,128]`；不需要 Plugin 权重，优先实现 |
| P1a-fallback | `aggregation-only` | 对照 / fallback | contract 为 `[1,192,64,128] -> [1,192,64,128]`；更接近卷积分支，展示区分度低于 `relu_linear_att-only` |
| P1b | `aggregation + cat + relu_linear_att` | 主性能评估边界 | Phase 1/2 都支持的中段组合候选，潜在收益更高但 graph 集成更复杂 |
| P1c | 整体 LiteMLA | fallback / 上限方案 | 融合空间最大，但实现、调试和数值验证风险最高，不作为第一步 |
| P2 | stage0/head MBConv | 工程优化候选 | 端到端收益潜力高，但标准算子链多，展示区分度低于 LiteMLA |

---

## 4. 任务清单

- [x] Step 0：从 `master` 创建 `phase3-plugin` 分支。
- [x] Step 1：建立 Phase 3 目录骨架与 `plugin_fusion_design.md` 第一版。
- [x] Step 2：精读 ONNX / TensorRT engine 中 `stage2/context` 的实际 tensor 边界，确定 P1a MVP 的输入输出。产物：[`design_notes/stage2_context_tensor_contract.md`](design_notes/stage2_context_tensor_contract.md)。
- [x] Step 3：设计最小 Plugin API 与 CMake 构建方案。产物：[`design_notes/plugin_api_cmake_design.md`](design_notes/plugin_api_cmake_design.md)。
- [x] Step 4：实现 P1a Plugin skeleton，先跑通 TensorRT Plugin 注册与 engine build。产物：[`results/metrics/relu_linear_attention_toy_build.json`](results/metrics/relu_linear_attention_toy_build.json)。
- [x] Step 5：实现 `relu_linear_att` CUDA kernel / enqueue 路径，并在 toy/plugin 单层层面与 PyTorch reference 做数值对齐。产物：[`results/metrics/relu_linear_attention_plugin_validation.json`](results/metrics/relu_linear_attention_plugin_validation.json)。本步不做完整 EfficientViT graph surgery。
- [x] Step 5.5：补充 Plugin 单层 microbenchmark + Nsight kernel summary，确认当前 kernel 在 MX250 `sm_61` / FP32 约束下的真实性能边界。产物：[`results/metrics/relu_linear_attention_plugin_microbenchmark_summary.md`](results/metrics/relu_linear_attention_plugin_microbenchmark_summary.md)。本步仍不做完整 EfficientViT graph surgery。
- [x] Step 6：将 Plugin 集成进真实 EfficientViT TensorRT graph，优先评估 ONNX graph surgery，若不稳定再评估 TensorRT Network API 局部重建。产物：[`results/metrics/relu_linear_attention_plugin_onnx_integration.json`](results/metrics/relu_linear_attention_plugin_onnx_integration.json)、[`results/metrics/relu_linear_attention_plugin_engine_build.json`](results/metrics/relu_linear_attention_plugin_engine_build.json)。本步只证明 parser/build 闭环，不证明 correctness / latency。
- [x] Step 7：复用 Phase 2 benchmark，比较 TensorRT FP32 baseline vs Plugin engine latency。产物：[`design_notes/plugin_engine_benchmark_design.md`](design_notes/plugin_engine_benchmark_design.md)、[`scripts/benchmark_plugin_engine.py`](scripts/benchmark_plugin_engine.py)、[`results/metrics/relu_linear_attention_plugin_engine_benchmark.json`](results/metrics/relu_linear_attention_plugin_engine_benchmark.json)、[`results/metrics/relu_linear_attention_plugin_engine_benchmark_summary.md`](results/metrics/relu_linear_attention_plugin_engine_benchmark_summary.md)。P1a-3b 冷机重测后 Plugin vs baseline TRT `allclose=True`，Plugin vs PyTorch argmax pixel agreement 100%。当前有效 `both` run 为 baseline p50 54.394ms、Plugin p50 52.168ms，speedup 1.043x；此前热机/并行污染 run 出现负收益，已作为无效测量处理。该结果说明整网 1ms 级差异对温度、频率和系统状态敏感，后续必须保持冷机/顺序运行纪律。
- [x] Step 8：采集 Plugin engine Nsight trace，更新 attribution summary。产物：[`design_notes/plugin_nsys_attribution_design.md`](design_notes/plugin_nsys_attribution_design.md)、[`scripts/analyze_plugin_nsys_attribution.py`](scripts/analyze_plugin_nsys_attribution.py)、[`results/metrics/relu_linear_attention_plugin_engine_benchmark_nsys.json`](results/metrics/relu_linear_attention_plugin_engine_benchmark_nsys.json)、[`results/metrics/relu_linear_attention_plugin_nsys_attribution_summary.md`](results/metrics/relu_linear_attention_plugin_nsys_attribution_summary.md)。P1a-3b 冷机重测后结果：Plugin layer 合计 1.310ms / 4 launches，对比 Phase 2 baseline `attention_core` proxy 的 3.689ms / 12 launches，目标边界 kernel time 约 2.816x 改善；`aggregation + plugin` proxy 为 3.062ms / 30 launches，对比 baseline `aggregation + attention_core` 的 5.443ms / 38 launches，约 1.778x 改善；stage2/context total 约 1.557x 改善。Nsight 采集需脱离 Codex 沙盒执行；普通权限下 CPU sampling / context switch trace 仍会被禁用，但 CUDA/NVTX trace 可用。
- [x] Step 8.5：尝试采集 Plugin VK 归约 kernel 硬件指标。结论：Nsight Compute 2024.1.1 不支持 MX250 (`sm_61`)，但 CUDA 12.4 `nvprof` 在补充 CUPTI DLL 路径后可采集 Pascal 指标。P1a-3a 的 `computeVkKernelDim16Warp4` 更接近 memory-dependency / load-latency 主导，不是 occupancy-bound 或同步归约开销主导；记录见 [`design_notes/plugin_kernel_optimization_history.md`](design_notes/plugin_kernel_optimization_history.md) §10 和 [`results/metrics/nvprof_p1a3a_vk_summary.md`](results/metrics/nvprof_p1a3a_vk_summary.md)。
- [x] Step 8.6：评估 P1a-4 单 kernel 合并可行性。产物：[`design_notes/p1a_single_kernel_feasibility.md`](design_notes/p1a_single_kernel_feasibility.md)。结论：`computeOutput` 依赖完整 VK 跨 `N=8192` 归约结果，当前两阶段 kernel boundary 承担了全局同步语义；naive single-kernel 需要接受低并行度、重复 VK 归约或高风险 device-side barrier，因此记录为 `evaluated, not adopted as mainline`。下一主线优先转向 P1b。
- [x] Step 8.7：启动 P1b `aggregation + cat + relu_linear_att` 落盘前设计。产物：[`design_notes/p1b_aggregation_attention_design.md`](design_notes/p1b_aggregation_attention_design.md)。结论：P1b 只覆盖两个 `stage2/context` 实例，保留 qkv/proj/residual 在 Plugin 外；第一步先验证带 aggregation 权重输入的 TensorRT parser/build 可行性，不直接承诺 CUDA 性能。
- [x] Step 8.8：实现 P1b skeleton / parser toy 验证。产物：[`plugin/include/edgeseg_aggregation_relu_linear_attention_plugin.h`](plugin/include/edgeseg_aggregation_relu_linear_attention_plugin.h)、[`plugin/src/edgeseg_aggregation_relu_linear_attention_plugin.cpp`](plugin/src/edgeseg_aggregation_relu_linear_attention_plugin.cpp)、[`scripts/build_p1b_plugin_toy_engine.py`](scripts/build_p1b_plugin_toy_engine.py)、[`results/metrics/p1b_aggregation_attention_toy_build.json`](results/metrics/p1b_aggregation_attention_toy_build.json)。结论：TensorRT 8.6.1 能创建 `EdgesegAggregationReluLinearAttention_TRT`，`parser_errors=[]`，toy network 只有 `qkv [1,192,64,128]` 一个 runtime input，两个 aggregation 权重保持为 initializer / constant 路径；当前 skeleton 只 zero-fill 输出，不代表 correctness / latency。
- [x] Step 8.9：完成 P1b 真实 EfficientViT ONNX surgery / engine build smoke。产物：[`scripts/integrate_p1b_aggregation_attention_plugin_onnx.py`](scripts/integrate_p1b_aggregation_attention_plugin_onnx.py)、[`scripts/build_p1b_plugin_engine.py`](scripts/build_p1b_plugin_engine.py)、[`results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json`](results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json)、[`results/metrics/p1b_aggregation_attention_plugin_engine_build.json`](results/metrics/p1b_aggregation_attention_plugin_engine_build.json)。结论：真实 ONNX 从 `393 -> 256` nodes，P1b Plugin nodes=2；TensorRT engine build `parser_errors=[]`、network IO 为 `input [1,3,1024,2048] -> segout [1,19,128,256]`、layers=239。当前 skeleton 仍只 zero-fill 输出，不代表 correctness / latency。
- [x] Step 8.10：完成 P1b 单 block 数值验证设计与 PyTorch reference 捕获。产物：[`design_notes/p1b_single_block_validation_design.md`](design_notes/p1b_single_block_validation_design.md)、[`scripts/capture_p1b_stage2_reference.py`](scripts/capture_p1b_stage2_reference.py)、[`results/metrics/p1b_stage2_reference_capture.json`](results/metrics/p1b_stage2_reference_capture.json)。结论：两个 `stage2/context` block 均被捕获，`qkv -> aggreg[0](qkv) -> cat -> relu_linear_att` 的 shape / weight group / projection sanity check 全部通过；当前 zero-fill skeleton 不可用于 correctness / latency。下一步应实现 P1b CUDA 数学，并用该 reference 做 block-level correctness。
- [x] Step 8.11：实现 P1b 第一版 CUDA 数学路径，并完成 block-level toy/plugin correctness 验证。产物：[`plugin/src/aggregation_relu_linear_attention_kernel.cu`](plugin/src/aggregation_relu_linear_attention_kernel.cu)、[`scripts/validate_p1b_aggregation_attention_plugin.py`](scripts/validate_p1b_aggregation_attention_plugin.py)、[`results/metrics/p1b_aggregation_attention_plugin_validation.json`](results/metrics/p1b_aggregation_attention_plugin_validation.json)。结论：P1b CUDA 文件与 P1a attention kernel 分开维护；第一版实现采用 `depthwise 5x5 -> grouped pointwise 1x1 -> cat workspace -> P1a attention launcher` 的正确性优先路径。两个真实 `stage2/context` block 均通过 `allclose(atol=1e-3, rtol=1e-3)`，block1 `max_abs_diff=1.31e-6`、block2 `max_abs_diff=2.38e-6`，`argmax_channel_agreement=1.0`。该结果证明 block-local 数学正确性，不代表真实 P1b engine 端到端 latency 或 Nsight 收益。
- [x] Step 8.12：用第一版 P1b CUDA 数学路径重建真实 EfficientViT P1b engine，并完成冷机端到端 correctness / latency benchmark。产物：[`results/metrics/p1b_aggregation_attention_plugin_engine_build.json`](results/metrics/p1b_aggregation_attention_plugin_engine_build.json)、[`results/metrics/p1b_aggregation_attention_plugin_engine_benchmark.json`](results/metrics/p1b_aggregation_attention_plugin_engine_benchmark.json)、[`results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md)。结论：Plugin TRT vs baseline TRT `allclose=True`、`max_abs_diff=4.43e-05`、argmax agreement `1.0`；Plugin TRT vs PyTorch relaxed allclose `1e-3` 通过、argmax agreement `1.0`。冷机 p50 baseline `54.4532ms`、P1b Plugin `56.3395ms`，p50 speedup `0.9665x`，mean speedup `0.9598x`。这说明第一版 P1b 数学正确，但端到端性能不应采纳；后续若继续 P1b，应先做 Nsight attribution 定位退化来源。
- [x] Step 8.13：采集 P1b Plugin engine Nsight trace，并用 SQLite correlationId 归因定位退化来源。产物：[`results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_nsys.json`](results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_nsys.json)、[`results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.md)、[`results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.json`](results/metrics/p1b_aggregation_attention_plugin_nsys_attribution_summary.json)。结论：P1b Plugin layer 合计 `6.189ms/iter`、`10 launches/iter`；对比 Phase 2 baseline `aggregation + attention_core` proxy 的 `5.443ms/iter`、`38 launches/iter`，P1b 虽然显著减少 launch 数，但 kernel time 变差为 `0.879x`。Plugin 内部 `depthwise5x5Kernel` 约 `2.462ms/iter`、`groupedPointwise1x1Kernel` 约 `2.427ms/iter`，合计占 Plugin kernel time 约 `79%`；P1a attention 相关 `computeVk + computeOutput` 约 `1.300ms/iter`。因此退化主因是 naive 自写 aggregation 替换了 TensorRT/cuDNN 已优化的标准 Conv 路径，不是 attention math 本身。采集纪律：Codex 沙盒内 `nsys profile` 曾 75s 超时，提权后同命令成功完成；普通权限下 CPU sampling / context switch trace 被禁用，但 CUDA/NVTX trace 和 SQLite attribution 可用。
- [x] Step 8.14：实现 P1b-1 fused aggregation+cat kernel，消除 `depthwiseWorkspace` global write/read、D2D cat copy 和一次 aggregation launch。产物：[`plugin/src/aggregation_relu_linear_attention_kernel.cu`](plugin/src/aggregation_relu_linear_attention_kernel.cu)、[`results/metrics/p1b_aggregation_attention_plugin_validation.json`](results/metrics/p1b_aggregation_attention_plugin_validation.json)、[`results/metrics/p1b_aggregation_attention_plugin_engine_build.json`](results/metrics/p1b_aggregation_attention_plugin_engine_build.json)、[`results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_engine_benchmark_summary.md)、[`results/metrics/p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.md)。结论：block-level correctness 继续通过；真实 P1b fused engine rebuild 成功，engine sha256=`dcba4c1d10e692f4922c9b1332cadcadfc0055371e4e945a1216e567a1d2e945`。端到端 baseline p50 `54.331ms`、P1b fused p50 `53.610ms`，speedup `1.0135x`。Nsight 显示 P1b fused Plugin layer `4.848ms/iter`、`6 launches/iter`，对比 Phase 2 baseline `aggregation + attention_core` `5.443ms/iter`、`38 launches/iter`，中段边界 kernel-time speedup `1.123x`；stage2/context total 从 baseline `6.383ms/iter` 改善到 `5.792ms/iter`。该结果修正了 Step 8.13 的判断：P1b 方向仍有价值，但必须避免 naive aggregation 中间 workspace 设计。真实 engine rebuild 耗时约 `342s`；此前 `180s` timeout 不足导致误判为卡住。
- [x] Step 8.15：实现 P1b-2 grouped pointwise 权重 shared-memory cache。产物：[`plugin/src/aggregation_relu_linear_attention_kernel.cu`](plugin/src/aggregation_relu_linear_attention_kernel.cu)、[`results/metrics/p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_summary.md)、[`results/metrics/p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.md)。结论：每个 aggregation CTA 将当前 group 的 `16x16` pointwise 权重缓存到 shared memory 后，block-level correctness 继续通过；冷机端到端 baseline p50 `54.306ms`、P1b-2 p50 `53.530ms`，speedup `1.0145x`。Nsight 显示 P1b Plugin layer 从 P1b-1 的 `4.848ms/iter` 降到 `4.347ms/iter`，`fusedAggregationCatKernel` 从 `3.536ms/iter` 降到 `3.038ms/iter`；中段边界相对 Phase 2 baseline `aggregation + attention_core` 达到 `1.252x` kernel-time speedup。首次热机样本曾显示负收益，已按测量纪律排除。
- [x] Step 8.16：评估 P1b-3 interior fast path。产物：[`results/metrics/p1b_aggregation_attention_plugin_interior_fastpath_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_interior_fastpath_engine_benchmark_summary.md)。结论：在同一个 `fusedAggregationCatKernel` 内给非边界像素去掉 depthwise 5x5 越界判断后，correctness 通过，但冷机 p50 为 baseline `54.312ms`、Plugin `54.710ms`，speedup `0.9927x`。该结果说明当前主瓶颈不在边界判断；该变体不采纳，主线代码已恢复到 P1b-2 shared weight cache。
- [x] Step 8.17：实现 P1b-4 depthwise row-tile shared cache。产物：[`plugin/src/aggregation_relu_linear_attention_kernel.cu`](plugin/src/aggregation_relu_linear_attention_kernel.cu)、[`results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_engine_benchmark_summary.md)、[`results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_nsys_attribution_summary.md)。结论：在真实 `stage2/context` contract 下，一个 CTA 正好覆盖两行 `2x128` spatial tile；P1b-4 每次缓存 4 个 channel 的 `6x132` halo tile 和 depthwise 5x5 权重到 shared memory。冷机端到端 baseline p50 `54.411ms`、P1b-4 p50 `52.725ms`，speedup `1.032x`。Nsight 显示 P1b Plugin layer 从 P1b-2 的 `4.347ms/iter` 降到 `3.574ms/iter`，`fusedAggregationCatKernel` 从 `3.038ms/iter` 降到 `2.262ms/iter`；中段边界相对 Phase 2 baseline `aggregation + attention_core` 达到 `1.523x` kernel-time speedup。
- [x] Step 8.18：实现 P1b-5 depthwise tile channel chunk A/B，采纳 `kDepthwiseTileChannels=8`。产物：[`plugin/src/aggregation_relu_linear_attention_kernel.cu`](plugin/src/aggregation_relu_linear_attention_kernel.cu)、[`results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_ch8_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_ch8_engine_benchmark_summary.md)、[`results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_ch8_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_depthwise_tile_ch8_nsys_attribution_summary.md)。结论：在 P1b-4 row-tile shared cache 基础上，将每个 CTA 的 depthwise channel chunk 从 4 扩到 8 后，block-level correctness 继续通过；冷机端到端 baseline p50 `54.297ms`、P1b-5 p50 `52.455ms`，speedup `1.035x`。Nsight 显示 P1b Plugin layer 从 P1b-4 的 `3.574ms/iter` 降到 `3.241ms/iter`，`fusedAggregationCatKernel` 从 `2.262ms/iter` 降到 `1.926ms/iter`；中段边界相对 Phase 2 baseline `aggregation + attention_core` 达到 `1.679x` kernel-time speedup。
- [x] Step 8.19：评估 P1b-6 `kDepthwiseTileChannels=16` probe。产物：[`design_notes/plugin_kernel_optimization_history.md`](design_notes/plugin_kernel_optimization_history.md)、[`design_notes/p1b_aggregation_attention_design.md`](design_notes/p1b_aggregation_attention_design.md)。结论：编译通过，但 block-level TensorRT Plugin validation 执行失败；该变体 shared memory 约 `53KB/block`，超过常见 Pascal per-block shared memory `48KB` 约束，因此不采纳，主线保持 P1b-5 的 `kDepthwiseTileChannels=8`。
- [x] Step 8.20：实现 P1b-7 CTA512 / 4-row tile A/B，采纳 `kAggregationThreads=512` 与 `kDepthwiseTileRows=8`。产物：[`plugin/src/aggregation_relu_linear_attention_kernel.cu`](plugin/src/aggregation_relu_linear_attention_kernel.cu)、[`results/metrics/p1b_aggregation_attention_plugin_cta512_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_cta512_engine_benchmark_summary.md)、[`results/metrics/p1b_aggregation_attention_plugin_cta512_nsys_attribution_summary.md`](results/metrics/p1b_aggregation_attention_plugin_cta512_nsys_attribution_summary.md)。结论：在 P1b-5 ch8 基础上，一个 CTA 从覆盖两行 `2x128` 改为覆盖四行 `4x128`，用 `8x132` halo tile 减少跨 CTA halo 重复加载；block-level correctness 继续通过。冷机端到端 baseline p50 `54.380ms`、P1b-7 p50 `52.311ms`，speedup `1.040x`。Nsight 显示 P1b Plugin layer 从 P1b-5 的 `3.241ms/iter` 降到 `3.043ms/iter`，`fusedAggregationCatKernel` 从 `1.926ms/iter` 降到 `1.730ms/iter`；中段边界相对 Phase 2 baseline `aggregation + attention_core` 达到 `1.789x` kernel-time speedup。
- [x] Step 8.21：评估 P1b-8 skip-final-sync probe。产物：[`results/metrics/p1b_aggregation_attention_plugin_skip_final_sync_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_skip_final_sync_engine_benchmark_summary.md)、[`design_notes/plugin_kernel_optimization_history.md`](design_notes/plugin_kernel_optimization_history.md)、[`design_notes/p1b_aggregation_attention_design.md`](design_notes/p1b_aggregation_attention_design.md)。结论：只跳过 row-tile 路径最后一个 channel chunk 之后的冗余 `__syncthreads()`，block-level correctness 通过，但端到端 p50 为 `53.123ms`，慢于 P1b-7 的 `52.311ms`；推测条件同步引入的控制流成本超过了省掉一次 CTA 内 barrier 的收益，因此不采纳，主线保持 P1b-7。
- [x] Step 8.22：评估 P1b-9 pointwise accumulator probe。产物：[`results/metrics/p1b_aggregation_attention_plugin_pointwise_accum_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_pointwise_accum_engine_benchmark_summary.md)、[`design_notes/plugin_kernel_optimization_history.md`](design_notes/plugin_kernel_optimization_history.md)、[`design_notes/p1b_aggregation_attention_design.md`](design_notes/p1b_aggregation_attention_design.md)。结论：把 grouped pointwise 16 个输出从嵌套循环改为单线程内 16 个标量累加器，block-level correctness 通过，但端到端 p50 为 `52.431ms`，慢于 P1b-7 的 `52.311ms`；推测寄存器压力和展开指令数增加抵消了减少 `depthwise[i]` 重读的收益，因此不采纳，主线保持 P1b-7。
- [x] Step 8.23：评估 P1b-10 pointwise accum4 probe。产物：[`results/metrics/p1b_aggregation_attention_plugin_pointwise_accum4_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_pointwise_accum4_engine_benchmark_summary.md)、[`design_notes/plugin_kernel_optimization_history.md`](design_notes/plugin_kernel_optimization_history.md)、[`design_notes/p1b_aggregation_attention_design.md`](design_notes/p1b_aggregation_attention_design.md)。结论：把 grouped pointwise 输出改为每次 4 个标量累加器，block-level correctness 通过，但端到端 p50 为 `54.197ms`，明显慢于 P1b-7；这说明当前 pointwise 输出映射方向不应继续投入，主线保持 P1b-7。
- [x] Step 8.24：用 `nvprof` 补充 P1b-7 `fusedAggregationCatKernel` 硬件指标。产物：[`results/metrics/nvprof_p1b7_fused_summary.md`](results/metrics/nvprof_p1b7_fused_summary.md)。结论：该 kernel 不是纯 DRAM bandwidth-bound，也不是 occupancy-bound；`achieved_occupancy ~= 0.496`、`sm_efficiency ~= 99.6%`、local memory overhead 为 0，最大 stall 是 `stall_exec_dependency ~= 53%`，更像 instruction dependency / scheduling 主导。
- [x] Step 8.25：评估 P1b-11a shared row pitch padding probe。产物：[`results/metrics/p1b_aggregation_attention_plugin_shared_pitch133_engine_benchmark_summary.md`](results/metrics/p1b_aggregation_attention_plugin_shared_pitch133_engine_benchmark_summary.md)、[`results/metrics/nvprof_p1b11a_fused_transactions.csv`](results/metrics/nvprof_p1b11a_fused_transactions.csv)。结论：把 `kDepthwiseTileWidth` 从 `132` 改为 `133` 后，p50 只比 P1b-7 快 `0.043ms`，`shared_efficiency` 几乎不变、global load transactions 略增，因此不采纳，主线恢复 P1b-7。
- [ ] Step 9：撰写 `integration_validation_report.md`。

---

## 5. 目录结构

```text
phase3/
|-- README.md
|-- design_notes/
|   |-- plugin_api_cmake_design.md
|   |-- plugin_fusion_design.md
|   |-- plugin_graph_integration_design.md
|   |-- plugin_engine_benchmark_design.md
|   |-- plugin_kernel_optimization_history.md
|   |-- plugin_nsys_attribution_design.md
|   |-- p1a_single_kernel_feasibility.md
|   |-- p1b_aggregation_attention_design.md
|   |-- p1b_single_block_validation_design.md
|   `-- stage2_context_tensor_contract.md
|-- plugin/
|   |-- CMakeLists.txt
|   |-- include/
|   |   |-- edgeseg_aggregation_relu_linear_attention_plugin.h
|   |   `-- edgeseg_relu_linear_attention_plugin.h
|   `-- src/
|       |-- aggregation_relu_linear_attention_kernel.cu
|       |-- edgeseg_aggregation_relu_linear_attention_plugin.cpp
|       |-- edgeseg_relu_linear_attention_plugin.cpp
|       `-- relu_linear_attention_kernel.cu
|-- scripts/
|   |-- .gitkeep
|   |-- benchmark_plugin_engine.py
|   |-- benchmark_relu_linear_attention_plugin.py
|   |-- build_plugin_engine.py
|   |-- build_plugin_toy_engine.py
|   |-- build_p1b_plugin_engine.py
|   |-- build_p1b_plugin_toy_engine.py
|   |-- capture_p1b_stage2_reference.py
|   |-- integrate_relu_linear_attention_plugin_onnx.py
|   |-- integrate_p1b_aggregation_attention_plugin_onnx.py
|   |-- analyze_plugin_nsys_attribution.py
|   |-- validate_p1b_aggregation_attention_plugin.py
|   `-- validate_relu_linear_attention_plugin.py
|-- results/
|   |-- engines/
|   |   `-- .gitkeep
|   |-- metrics/
|   |   |-- .gitkeep
|   |   |-- p1b_stage2_reference_capture.json
|   |   |-- p1b_aggregation_attention_plugin_depthwise_tile_engine_benchmark_summary.md
|   |   |-- p1b_aggregation_attention_plugin_depthwise_tile_nsys_attribution_summary.md
|   |   |-- p1b_aggregation_attention_plugin_depthwise_tile_nsys_attribution_summary.json
|   |   |-- p1b_aggregation_attention_plugin_depthwise_tile_ch8_engine_benchmark_summary.md
|   |   |-- p1b_aggregation_attention_plugin_depthwise_tile_ch8_nsys_attribution_summary.md
|   |   |-- p1b_aggregation_attention_plugin_depthwise_tile_ch8_nsys_attribution_summary.json
|   |   |-- p1b_aggregation_attention_plugin_cta512_engine_benchmark_summary.md
|   |   |-- p1b_aggregation_attention_plugin_cta512_nsys_attribution_summary.md
|   |   |-- p1b_aggregation_attention_plugin_cta512_nsys_attribution_summary.json
|   |   |-- p1b_aggregation_attention_plugin_skip_final_sync_engine_benchmark_summary.md
|   |   |-- p1b_aggregation_attention_plugin_engine_build.json
|   |   |-- p1b_aggregation_attention_plugin_engine_benchmark.json
|   |   |-- p1b_aggregation_attention_plugin_engine_benchmark_nsys.json
|   |   |-- p1b_aggregation_attention_plugin_engine_benchmark_summary.md
|   |   |-- p1b_aggregation_attention_plugin_fused_engine_benchmark_nsys.json
|   |   |-- p1b_aggregation_attention_plugin_fused_engine_benchmark_nsys_summary.md
|   |   |-- p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.md
|   |   |-- p1b_aggregation_attention_plugin_fused_nsys_attribution_summary.json
|   |   |-- p1b_aggregation_attention_plugin_interior_fastpath_engine_benchmark_summary.md
|   |   |-- p1b_aggregation_attention_plugin_nsys_attribution_summary.md
|   |   |-- p1b_aggregation_attention_plugin_nsys_attribution_summary.json
|   |   |-- p1b_aggregation_attention_plugin_onnx_integration.json
|   |   |-- p1b_aggregation_attention_plugin_validation.json
|   |   |-- p1b_aggregation_attention_plugin_weight_shared_engine_benchmark_summary.md
|   |   |-- p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.md
|   |   |-- p1b_aggregation_attention_plugin_weight_shared_nsys_attribution_summary.json
|   |   |-- p1b_aggregation_attention_toy_build.json
|   |   |-- relu_linear_attention_plugin_engine_build.json
|   |   |-- relu_linear_attention_plugin_engine_benchmark.json
|   |   |-- relu_linear_attention_plugin_engine_benchmark_nsys.json
|   |   |-- relu_linear_attention_plugin_engine_benchmark_summary.md
|   |   |-- relu_linear_attention_plugin_nsys_attribution_summary.md
|   |   |-- relu_linear_attention_plugin_nsys_attribution_summary.json
|   |   |-- relu_linear_attention_plugin_microbenchmark.json
|   |   |-- relu_linear_attention_plugin_microbenchmark_kernel_stats_cuda_gpu_kern_sum.csv
|   |   |-- relu_linear_attention_plugin_microbenchmark_nsys.json
|   |   |-- relu_linear_attention_plugin_microbenchmark_summary.md
|   |   |-- relu_linear_attention_plugin_onnx_integration.json
|   |   |-- relu_linear_attention_plugin_validation.json
|   |   `-- relu_linear_attention_toy_build.json
|   |-- onnx/
|   |   `-- .gitkeep
|   |-- figures/
|   |   `-- .gitkeep
|   |-- tensors/
|   |   `-- .gitkeep
|   `-- nsight/
|       `-- .gitkeep
`-- logs/
    `-- .gitkeep
```

---

## 6. 验收口径

Phase 3 的一次有效 Plugin 实验至少需要同时满足：

- TensorRT engine 可构建并能加载 Plugin。
- 输出与 TensorRT FP32 baseline / PyTorch reference 在约定阈值内对齐。
- latency 使用 Phase 2 相同的 CUDA Events execute-only 口径。
- Nsight attribution 能看到 Plugin 前后 residual hotspot 的变化。
- 报告中清楚区分：
  - 端到端 latency 改善；
  - Plugin 覆盖的 layer / kernel 范围；
  - 数值误差；
  - 工程风险与 fallback 路线。
