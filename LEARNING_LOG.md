# LEARNING_LOG

## 2026-05-27

Q: `time.perf_counter()` 和 CUDA Events 测 GPU 推理延迟有什么区别？

A: PyTorch 调 CUDA kernel 通常是异步的，直接用 `time.perf_counter()` 包住 `model(x)` 只会测到 CPU 发射任务的时间，往往严重低估 GPU 推理耗时。若使用 `time.perf_counter()`，必须在计时前后调用 `torch.cuda.synchronize()`，这样测到的是 Python/CPU 视角的一次端到端等待时间，包含 kernel launch、GPU 执行、同步等待和少量系统抖动。CUDA Events 则是在 CUDA stream 中插入 `start` 和 `end` 两个事件，用 GPU 时间线计算两点之间的 elapsed time，主要反映 GPU stream 内 forward 的实际执行时间，受 CPU 抖动影响更小。项目中建议主 benchmark 使用 CUDA Events 报告 `gpu_latency_ms`，同时可用 `perf_counter + synchronize` 作为 `wall_latency_ms` 辅助指标。

Q: `torch.cuda.synchronize()` 的原理是什么？

A: `torch.cuda.synchronize()` 是 CPU-GPU 同步屏障。CUDA 默认异步执行，CPU 提交 kernel 后通常不会等待 GPU 完成就继续执行；调用 `torch.cuda.synchronize()` 会让 CPU 阻塞，直到当前 CUDA device 上已提交的任务完成，包括 kernel、GPU tensor op 和异步内存拷贝。它不会清空显存、不会改变结果，也不是计时函数。它适合用于 wall-clock 计时边界、warmup/measure 边界，以及确保读取 CUDA Event 时间前相关事件已完成。但不要在 NVTX range 或模型每一层内部随意加入同步，否则会改变执行节奏并污染 Nsight profiling 结果。

Q: NVTX 的相关原理是什么？

A: NVTX（NVIDIA Tools Extension）的作用是在程序时间线上打标签，帮助 Nsight Systems / Nsight Compute 把底层 CUDA kernel 归因到高层代码阶段，例如 `backbone`、`stage3`、`LiteMLA`、`SegHead`。`range_push()` / `range_pop()` 标记的是 CPU 代码区间，Nsight 会记录这个区间内发射的 CUDA kernels，并在 timeline 中建立关联。NVTX 不是计时工具，也不会加速模型；CUDA Events 负责“测多久”，NVTX 负责“标是谁”。在项目中，NVTX range 内不要加入 `torch.cuda.synchronize()`，否则会强行打断异步执行、污染 profiling 结果；同步应只放在 warmup/measure 边界。

Q: `cudnn.benchmark=True` 和 `cudnn.deterministic=True` 分别意味着什么？

A: cuDNN 对同一个卷积可能有多种实现算法，例如 direct、implicit GEMM、Winograd 等。`torch.backends.cudnn.benchmark=True` 会让 cuDNN 针对固定输入 shape 尝试多个候选算法并缓存最快方案，因此首次 forward 可能包含算法搜索开销，需要 warmup 排除。`torch.backends.cudnn.deterministic=True` 会限制 cuDNN 只使用确定性算法，更适合训练复现和 debug，但可能放弃最快实现。项目 baseline 的目标是固定分辨率下的 PyTorch 最优推理性能，所以建议 `benchmark=True, deterministic=False`，并在 JSON 中记录该配置。

Q: direct convolution、im2col + GEMM、FFT、Winograd、implicit GEMM、fused variants 这些卷积实现方式有什么区别？

A: direct convolution 按卷积定义直接遍历计算，额外内存少，但不一定最快。`im2col + GEMM` 把滑动窗口展开成大矩阵，再调用高性能矩阵乘，速度好但中间矩阵占用大。FFT-based convolution 利用“空间域卷积等于频域逐点乘法”，大 kernel 可能有优势，小 kernel 通常不划算。Winograd 通过数学变换减少 `3x3` 普通卷积的乘法次数，但有变换开销和数值/适用性限制。implicit GEMM 像 GEMM 一样 tiled 计算卷积，但不显式生成完整 im2col 矩阵，现代 cuDNN 常用，显存更友好。fused variants 把 Conv、Bias、BN、Activation、Add 等合并，减少 kernel launch 和显存读写。对本项目而言，常规 Conv-BN-Act 应优先交给 cuDNN/TensorRT，Plugin 更应关注 LiteMLA 这种非标准算子链。

Q: CUDA lazy initialization、PyTorch kernel/cache 初始化、GPU 频率和显存分配分别是什么？

A: CUDA lazy initialization 指 PyTorch 第一次真正使用 CUDA 时才创建 CUDA context、初始化 runtime/driver、库句柄、默认 stream 和 allocator，因此第一次 CUDA 操作会很慢。PyTorch kernel/cache 初始化包括 cuDNN 算法搜索缓存、cuBLAS/cuBLASLt handle 和 heuristic cache、CUDA caching allocator 初始化，以及部分 kernel dispatch/cache 准备。GPU 频率方面，空闲 GPU 常处于低频，前几次推理会经历升频到稳定性能状态；显存方面，第一次 forward 会触发输入、中间激活、workspace、临时 buffer 的显存申请，后续可由 PyTorch caching allocator 复用。warmup 的目的就是提前消耗这些一次性成本，让正式 benchmark 测到稳定推理路径。
## 2026-06-05

Q: Phase 1 中为什么不能直接把 NVTX range 的时间长度当作组件真实耗时？

A: NVTX 的主要职责是给时间线打结构标签，不是独立计时工具。`Threads -> NVTX` 更接近 CPU/Python enqueue 侧的 range 生命周期，可能包含 Python 调用、CUDA launch、调度等待等；`CUDA HW -> NVTX` 是 Nsight 给出的 GPU 侧投影趋势参考，可以辅助观察结构与 kernel 的对应关系，但不能作为定量归因依据。项目中的口径应是：CUDA Events 负责端到端 latency，Nsight SQLite attribution 负责组件级 GPU 耗时归因，截图负责辅助读者理解时间线形态。

Q: 为什么 Phase 1 要分 Plan A / B / C / D 四种 profiling？

A: 四个 Plan 的职责不同。Plan A 是无 NVTX 的干净端到端 baseline，用来回答“PyTorch 原生路径整体多快”。Plan B 是大区域归因，标出 `stem -> stage0 -> stage1 -> stage2 -> stage3 -> head`，用来回答瓶颈主要落在哪些高层区域。Plan C 是热点区域组件级归因，聚焦 stage0 / stage2 / head 中的关键组件，判断热点到底来自 MBConv、LiteMLA context、local module 还是 head。Plan D 进一步只看 stage2 LiteMLA 内部子路径，用来把 Phase 3 Plugin 候选从“LiteMLA 整体”细化到更具体的可融合路径。

补充口径：这里的 `stage0~3` 是 Phase 1 NVTX / 代码中的 `backbone.stages` 索引，不是架构语义阶段编号；代码/NVTX `stage2` 对应架构语义 stage3。

Q: 为什么 LiteMLA 不是全模型最大耗时模块，却仍然是 Phase 3 Plugin 主线？

A: Plan B/C 显示 stage0、head 等 MBConv/Conv 路径占比较高，但这些模块慢主要来自高分辨率特征图和标准卷积/深度卷积的计算量，TensorRT/cuDNN 已有较成熟优化路径。LiteMLA 的绝对耗时未必最大，但它是 EfficientViT 的核心非标准结构，包含 reshape、cat、aggregation、linear attention、projection 等更难被 TensorRT 自动融合的路径，更适合作为“自定义 CUDA/TensorRT Plugin 展示工程价值”的主线。因此要区分“当前端到端最大耗时”和“最值得写 Plugin 的非标准算子链”。

Q: 为什么要把“端到端收益优先级”和“Plugin 展示价值优先级”分开？

A: 端到端收益优先级只看当前 PyTorch/Nsight 中的耗时占比，stage0/head 这类高分辨率 MBConv 可能排名很靠前。Plugin 展示价值优先级还要看非标准程度、TensorRT 是否已有强优化、融合边界是否清晰、实现风险是否可控、是否能体现项目特色。对求职项目而言，直接重写普通 MBConv 未必比优化 LiteMLA 更有含金量；更合理的排序是把标准 Conv/MBConv 作为 Phase 2 TensorRT baseline 观察对象，把 stage2 LiteMLA 作为 Phase 3 Plugin 主候选。

Q: Plan D 对 LiteMLA Plugin 候选给出了什么结论？

A: Plan D 显示 stage2 LiteMLA 内部主要应关注 `aggregation`、`cat`、`relu_linear_att` 这一段链路，以及必要时的整个 LiteMLA。由此形成三类候选：第一类是单独把 `aggregation` 或 `relu_linear_att` 做成 Plugin，边界小、实现风险较低，适合作为 MVP 或验证点；第二类是融合 `aggregation + cat + relu_linear_att`，能减少中间 tensor 写回和 kernel launch，更像真正的性能优化主线；第三类是 LiteMLA 整体 Plugin，理论融合空间最大，但实现和数值验证风险也最高，可作为进阶方案而不是第一步。

Q: stage0 和 head 很耗时，为什么不优先做它们的自定义 Plugin？

A: stage0 和 head 中的大头主要是 MBConv/Conv 类模块，它们慢是高分辨率输入下的客观结果，不一定说明它们适合自定义 Plugin。标准卷积、depthwise conv、BN/activation 这类路径通常已经被 cuDNN/TensorRT 深度优化，自己写 Plugin 很可能投入大、收益小，还会绕开成熟库的调优。更稳妥的判断是：先在 Phase 2 看 TensorRT baseline 是否已经显著优化这些 MBConv/Conv 热点；如果 TensorRT 后仍有残余瓶颈，再考虑针对标准模块做结构级或图级优化。

Q: 目前普通权限下的 Nsight 结果能否排除 CPU enqueue、多线程调度、WDDM 调度等问题？

A: 不能完全排除。Windows Nsight 的 CPU sampling / context switch / WDDM 相关 tracing 通常需要管理员权限；普通权限下我们只能说“现有 timeline 和 latency 结果没有显示 CPU enqueue 或调度成为主导瓶颈”，不能说已经严格排除。若后续看到明显 GPU 空洞、CUDA API launch 间隔异常、或 p95/p99 被系统抖动拉高，就需要管理员权限重跑带 CPU/WDDM trace 的 Nsight。当前 Phase 1 的结论应保守表述为“主要证据支持 GPU kernel/模块归因瓶颈”，而不是“CPU/OS 因素已被完全排除”。

Q: Phase 1 的 latency 是否包含 dataloader 或 preprocessing？

A: 不包含。`baseline_inference.py` 的测量区间是在输入 tensor 已准备好并搬到 GPU 后，只围绕 `model(x)` 做 warmup 和 measurement。因此 Phase 1 的 latency 是模型 forward latency，不是完整应用 pipeline latency。这个设计适合定位模型结构和 kernel 瓶颈；若后续要评估真实部署吞吐，需要另写 pipeline benchmark，把图像读取、resize/normalize、H2D 拷贝、后处理也纳入。

Q: Phase 1 是否应该测精度指标？

A: Phase 1 的主目标是性能基线和瓶颈归因，不应把 mIoU 作为阶段完成条件。可以做一张真实 Cityscapes 样图的可视化 sanity check，确认权重、输入预处理和输出类别大致正常；但正式精度指标更适合放到 Phase 2/3，在 TensorRT 导出、Plugin 替换后做 PyTorch vs TensorRT/Plugin 的数值一致性和 mIoU 回归验证。这样能避免 Phase 1 被完整评测集下载和精度流程拖偏。

Q: LiteMLA 外部存在类似 `autocast(enabled=False)` 的 FP32 保护，对 Plan D 和 Phase 3 Plugin 有什么影响？

A: 这说明 LiteMLA 内部某些计算对数值稳定性比较敏感，原实现倾向在该区域避免低精度自动混合。Plan D 只添加 NVTX range，不改变数学计算和 dtype，因此不会破坏这个保护；但 Phase 3 如果写 Plugin，就必须显式决定内部计算精度策略，例如关键 accumulation 保持 FP32、输出再按 TensorRT 网络 dtype 转换。Plugin 不能只追求更快，还必须设计数值对齐测试和误差阈值。

Q: Triton / wandb 兼容补丁给项目留下了什么经验？

A: 上游库的 import 链可能引入当前路径根本不用的依赖，例如 Windows 上 `triton` 缺 wheel，或 `wandb` 在退出时触发临时目录清理 PermissionError。项目中的做法是使用透明、最小、import-only 的兼容补丁：Triton stub 必须延迟到 `import torch` 之后、`import efficientvit` 之前注入，避免 PyTorch 把 fake triton 当成真实 triton 走错误路径；wandb 则用禁用/离线式处理避免推理脚本退出噪声。这类补丁应记录到 JSON 或设计文档中，确保后续读结果的人知道环境被怎样修过。

Q: Phase 1 中人工 review 主要纠正了哪些设计决策问题？

A: 主要纠正了三类问题。第一类是测量方法论：单请求 latency 不能用 throughput 式批量 enqueue 口径替代；NVTX 只做标注，不做计时；`synchronize()` 不能放进 NVTX range；组件耗时不能直接读 NVTX range 长度，而要用 Nsight SQLite attribution。第二类是证据链边界：Plan A/B/C/D 必须分别承担干净 latency、大区域归因、热点组件归因、stage2 LiteMLA 内部细分；普通权限 Nsight 不能完全排除 CPU/WDDM 因素；Phase 1 不以 mIoU 为完成条件。第三类是优化候选判断：最大耗时的 stage0/head 多为标准 MBConv/Conv，不等于最适合写 Plugin；LiteMLA 虽不是全模型最大瓶颈，但因其非标准性和项目区分度仍是 Phase 3 主线；Plan D 又把候选进一步细化为 `aggregation-only` / `relu_linear_att-only`、`aggregation + cat + relu_linear_att`、整体 LiteMLA fallback。项目内的完整纠偏审计记录见 `phase1/design_notes/phase1_decision_corrections.md`。

Q: CUDA runtime/kernel `correlationId` 是什么，为什么 Phase 1 attribution 需要它？

A: `correlationId` 是 Nsight/CUPTI 用来把 CPU 侧 CUDA Runtime API 和 GPU 侧 kernel execution 配对的编号。PyTorch forward 在 CPU 线程上发出 `cudaLaunchKernel`、`cudaMemcpyAsync`、`cudaEventRecord` 等 API，而真正计算发生在 CUDA stream 上的 kernel 中；二者异步执行，不能单靠时间重叠判断归属。Nsight 会让一次 runtime launch 和它触发的 GPU kernel 共享同一个 `correlationId`，于是可以从某个 NVTX range 内发出的 CUDA Runtime API 找到对应 kernel，再汇总这些 kernel 的 GPU duration。Phase 1 的正确口径是：NVTX 负责结构标签，CUDA Events 负责端到端 latency，SQLite attribution 通过 `correlationId` 把 kernel time 归因到 Plan B/C/D 的组件 range；不能直接把 NVTX range 的 end-start 当成组件 GPU 耗时。
