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

## 2026-06-08

Q: 为什么 ONNXRuntime 对齐验证中，ONNX 输出和 PyTorch 输出没有逐 bit 完全相同？

A: 这是正常现象。当前比较的是 PyTorch CUDA 输出和 ONNXRuntime CPUExecutionProvider 输出，它们数学语义等价，但底层 kernel、算子分解、执行顺序和 reduction 顺序不同；FP32 又不满足严格结合律，所以最后几位可能不同。ONNX 导出还可能做图转换、常量折叠和运行时优化，也会让实际执行图不完全等同于 PyTorch eager 路径。因此判断导出是否正确不应要求 bitwise identical，而应看 `allclose_pass`、`max_abs_diff`、`mean_abs_diff`、`cosine_similarity` 和输出 shape。当前 Phase 2 首版验证结果为 `allclose_pass=true`、`max_abs_diff≈3.44e-4`、`mean_abs_diff≈1.81e-5`、`cosine_similarity≈0.99999999999`，属于健康范围。`max_rel_diff` 较大时要谨慎解读，因为接近 0 的输出值会把相对误差放大。

Q: TensorRT 的基本工作流是什么？

A: TensorRT 总流程可以概括为 `PyTorch -> ONNX -> TensorRT parser -> TensorRT builder -> engine -> runtime inference / benchmark -> C++ runtime demo`。前半段是“构建期”：ONNX 把 PyTorch 模型变成静态图，TensorRT parser 把 ONNX 转成 TensorRT network，builder 根据 GPU、shape、dtype、workspace 和 tactic 选择生成当前机器专用的 serialized engine。后半段是“运行期”：runtime 不再解析 ONNX，而是直接反序列化 engine、创建 execution context、绑定输入输出 GPU buffer，并 enqueue 执行推理。

TensorRT C++ 工作流更贴近真实部署：第一步用 `ILogger` 创建日志器，用 `nvinfer1::createInferRuntime(logger)` 得到 `IRuntime`；第二步读取 `.engine` 文件字节，通过 `runtime->deserializeCudaEngine(...)` 得到 `ICudaEngine`；第三步用 `engine->createExecutionContext()` 创建 `IExecutionContext`；第四步查询 input/output binding 的 name、shape、dtype 和元素数量，按这些信息分配 host buffer 与 CUDA device buffer；第五步准备输入，把 host input 通过 `cudaMemcpy` 拷到 device input；第六步把 device pointer 放入 bindings 数组，调用 `context->executeV2(...)` 或异步版本 `enqueueV2 / enqueueV3`；第七步同步 CUDA stream，把 device output 拷回 host，再做输出统计或对齐检查；最后释放 CUDA buffer 和 TensorRT 对象。CMake/编译层面还要能找到 TensorRT include/lib，并在运行时让 Windows 能找到 `nvinfer.dll` 等 DLL。

Python 侧的 `build_trt_engine.py` / `benchmark_trt_engine.py` 更适合完成 engine 构建、latency benchmark、输出对齐和 Nsight attribution；C++ demo 的重点不是重新做严谨 benchmark，而是验证部署侧 TensorRT Runtime API 的最小闭环。它的价值在于为 Phase 3 Plugin 链路预热：后续自定义 Plugin 需要在 C++/TensorRT runtime 中注册、反序列化、加载 engine 并执行推理，C++ demo 证明这条工程链已经能跑通。

## 2026-06-10

Q: `logits diff`、`relaxed allclose`、`cosine similarity`、`argmax pixel agreement` 分别在 Phase 2 中说明什么？

A: `logits diff` 直接比较 TensorRT / ONNXRuntime 输出 logits 和 PyTorch reference 的逐元素误差，常看 `max_abs_diff` 与 `mean_abs_diff`；它能发现数值偏移，但不直接等价于语义错误。`relaxed allclose` 是放宽阈值后的逐元素近似一致性判断，例如 TensorRT FP32 strict `1e-4` 未过但 relaxed `1e-3` 通过，说明误差可解释但不能写成“逐元素严格一致”。`cosine similarity` 看两个输出向量方向是否几乎一致，适合衡量整体 logits 分布是否接近。`argmax pixel agreement` 比较每个像素最终类别是否一致；当前样图 `100%` 一致，说明语义分割输出在这张图上没有类别变化。Phase 2 的正确表述是“数值接近且语义输出一致”，不是“bitwise identical”。

Q: bicubic 和 bilinear resize 有什么区别，为什么 SegHead 的 bicubic 在 Phase 2 要单独记录？

A: bilinear 使用 2x2 邻域做线性插值，计算简单、边界清晰、部署框架普遍支持；bicubic 使用 4x4 邻域做三次插值，结果通常更平滑，但算子参数和实现细节更多，例如 cubic 系数、坐标变换模式、align/half-pixel 语义等。EfficientViT-Seg 的 SegHead ONNX 中存在 `mode=cubic` 的 `Resize` 节点，因此一开始担心 TensorRT parser/build 不支持或语义不一致。Phase 2 已验证：当前固定 shape、TensorRT 8.6.1、`mode=cubic`、`half_pixel`、`cubic_coeff_a=-0.75` 组合可以 parse/build/runtime，不是当前阻塞项。但这个结论不能外推到动态 shape、其他 TensorRT 版本或其他 cubic 参数。

Q: `tensorrt==8.6.1` 的 pip 包和 NVIDIA archived Windows zip 有什么区别？

A: pip 包主要提供 Python binding，方便 `import tensorrt`；但 TensorRT 真正运行还依赖本地 DLL、lib、header、parser/runtime 库等。NVIDIA archived Windows zip 提供完整的本地 TensorRT runtime / builder / include / lib 文件，是 Windows 上手动部署 legacy TensorRT 的核心来源。本项目的 MX250 是 Pascal `sm_61`，新版 pip TensorRT / TensorRT 10+ 路线可能安装成功但 builder 不支持该 GPU；最终采用 archived TensorRT 8.6.1 Windows zip，再配合 Python wheel 和显式 DLL path 注入，才让 build / benchmark / C++ demo 都跑通。

Q: 为什么 `benchmark_trt_engine.py` 用 PyTorch CUDA tensor 作为 TensorRT binding buffer？

A: TensorRT C++/Python runtime 执行时需要的是 GPU 内存地址，`bindings[input_idx] = int(input_tensor.data_ptr())` 和 `bindings[output_idx] = int(output_tensor.data_ptr())` 本质上是把 PyTorch CUDA tensor 的底层 device pointer 交给 TensorRT。这样可以复用 PyTorch 的 CUDA runtime、显存分配和 stream，避免第一版引入 pycuda / cuda-python 新依赖。代价是必须严格保证 tensor 生命周期覆盖 TensorRT 执行期，shape/dtype 与 engine binding 完全一致，tensor contiguous，且位于正确 CUDA device。这个方案适合 Python benchmark；C++ demo 则使用 TensorRT/CUDA 原生 buffer，服务 Phase 3 Plugin 集成链路。

Q: HardSwish 是什么运算，为什么在 TensorRT attribution 中经常出现？

A: HardSwish 是 Swish 的分段线性近似，常见公式是 `x * ReLU6(x + 3) / 6`，在移动端网络和 EfficientViT 的 MBConv / Conv-Act 路径中很常见。它比标准 Swish 更容易高效实现，因为不需要 sigmoid。TensorRT attribution 中大量 `PWN(...HardSwish...)` 表示 TensorRT 把 HardSwish 相关 pointwise 操作融合成 pointwise fusion layer。它说明 TensorRT 对标准 activation/pointwise 链有自动融合能力，也支持“标准 MBConv/Conv 链未必适合作为首个自定义 Plugin 主线”的判断。

Q: EngineInspector / ONNX node name 映射能说明 TensorRT 具体优化了什么吗？

A: 能说明结构变化，但不能单独说明真实耗时。EngineInspector 显示 ONNX `393` nodes 被 TensorRT 降到 `155` engine layers，存在 `PWN(...)` pointwise fusion、layer name 中的显式 `+` fusion、Conv/Add 合并、部分 activation fusion 等，这说明 TensorRT 做了图优化和算子融合。但 EngineInspector 当前主要是 layer name 级结构证据，不含完整 tactic metadata，也不提供 GPU kernel duration。真实瓶颈仍要看 Nsight SQLite attribution；报告中应把 EngineInspector 作为“TensorRT 做了哪些结构优化”的辅助证据，而不是 runtime 排序依据。

Q: TensorRT 优化后，LiteMLA 是否已经被自动融合成一个算子？

A: 没有。Phase 2 EngineInspector 和 Nsight attribution 都显示，`stage2/context` LiteMLA 没有被 TensorRT 自动合成一个单独 fused operator；仍能看到 `qkv/Conv`、`aggregation Conv`、`Relu`、`Pad`、`MatMul`、`Add/Div`、`proj/Conv + Add` 等多个相关 engine layers。Nsight 中 `stage2/context` 仍有约 `6.383 ms / iter` 和 `42 launches / iter`。这说明 TensorRT 虽然做了局部 pointwise / Conv+Add 融合，但并没有完成 LiteMLA 级别的整体融合，因此 Phase 3 继续评估 LiteMLA Plugin 仍有依据。

Q: Phase 1 的 `relu_linear_att` 与 Phase 2 TensorRT attribution 中的 `matmul / aggregation / pad / relu_qk / qkv / proj_add / norm_add_div` 怎么对应？

A: Phase 1 Plan D 是 PyTorch 源码语义边界：`qkv`、`aggregation`、`cat`、`relu_linear_att`、`proj`。Phase 2 TensorRT attribution 是 TensorRT layer-name 视角，会把 `relu_linear_att` 内部进一步分散成 `relu_qk`、`pad`、`matmul`、`norm_add_div` 等 proxy component，同时 `aggregation`、`qkv`、`proj_add` 仍能通过 ONNX-like path 大致识别。关键是不能把 Phase 2 的 `attention_core = relu_qk + pad + matmul + norm_add_div` 反向改写成 Phase 1 的 MVP；它只是 TensorRT 侧对 `relu_linear_att` 内部 residual path 的 proxy。Phase 1 候选仍是 `relu_linear_att-only` / `aggregation-only`、`aggregation + cat + relu_linear_att`、整体 LiteMLA fallback。

Q: TensorRT 后各 group 的加速比应该怎么理解？

A: Phase 2 报告新增了同名 group 的近似归因对比：Phase 1 Plan B attributed groups 总计约 `86.818 ms / iter`，TensorRT attributed groups 总计约 `54.454 ms / iter`，group-level 约 `1.59x`，与端到端 p50 speedup `1.57x` 接近，说明 attribution 与 CUDA Events 主结论相互吻合。逐 group 看，`stage0` 绝对节省最大（约 `9.859 ms / iter`），仍是端到端收益最大的工程热点；`stage2` 也被加速但仍保留 `12.179 ms / iter` 和较高 launch density，因此 LiteMLA Plugin 主线仍有 residual evidence。注意 Phase 1 PyTorch NVTX group 和 Phase 2 TensorRT layer-name group 语义接近但不是逐层一一对应。
