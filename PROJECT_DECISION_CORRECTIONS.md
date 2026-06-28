# 项目设计纠偏总账

> 本文记录项目推进过程中由人工 review 纠正的关键设计决策。它不是学习笔记，也不是流水账，而是跨阶段的方法论审计：哪些判断曾经不充分，为什么需要修正，最终如何影响 Phase 1/2/3 的技术路线。
>
> 阶段内完整记录见：
>
> - [`phase1/design_notes/phase1_decision_corrections.md`](./phase1/design_notes/phase1_decision_corrections.md)
> - [`phase2/design_notes/phase2_decision_corrections.md`](./phase2/design_notes/phase2_decision_corrections.md)
> - [`phase3/design_notes/phase3_decision_corrections.md`](./phase3/design_notes/phase3_decision_corrections.md)
>
> [`LEARNING_LOG.md`](./LEARNING_LOG.md) 负责记录概念学习和技术笔记；本文负责记录项目级决策质量控制。

---

## 1. 为什么需要这份文档

本项目大量使用 AI Agent 协作完成代码、实验和文档。但项目的技术含金量不来自“让 Agent 生成文件”，而来自：

- 能发现 Agent 初始方案中的方法论漏洞；
- 能把错误假设改成可复现实验口径；
- 能区分性能数据、结构证据、截图观察和工程叙事；
- 能把 Phase 1 的 profiling 结论延续到 Phase 2/3，而不是每阶段重新发明一套口径。

因此，纠偏记录本身是项目能力展示的一部分：它证明项目 owner 对推理优化、Nsight profiling、TensorRT deployment 和 Plugin 候选选择有实际判断力。

---

## 2. Phase 1 关键纠偏摘要

完整记录见 [`phase1/design_notes/phase1_decision_corrections.md`](./phase1/design_notes/phase1_decision_corrections.md)。

### 2.1 计时、标注、归因必须分开

早期方案曾把 CUDA Events 批量 enqueue 口径用于 latency 主指标，并且容易把 NVTX range duration 当作组件耗时。最终修正为：

- CUDA Events 负责端到端 latency；
- NVTX 只负责结构标注；
- Nsight SQLite attribution 通过 CUDA runtime/kernel `correlationId` 负责组件 GPU kernel time 归因；
- NVTX range 的 `end-start` 不作为组件 GPU 耗时。

这个修正确立了后续 Phase 1/2 报告的核心方法论。

### 2.2 Plan A/B/C/D 必须分层回答不同问题

早期 Plan B/C 的职责边界曾混乱。最终修正为：

| Plan | 目标 |
|---|---|
| A | 无 NVTX 的干净 PyTorch latency baseline |
| B | 全模型大区域归因：`stem/stage0/stage1/stage2/stage3/head` |
| C | 热点区域组件级归因：`stage0/stage2/head` |
| D | `stage2/context` LiteMLA 内部子路径归因 |

这个分层避免了用一个 Nsight 视图同时支撑所有结论。

### 2.3 最大耗时不等于最适合写 Plugin

Phase 1 结果显示 `stage0` 和 `head` 等 MBConv/Conv 路径占比高，但这类模块属于标准算子链，cuDNN/TensorRT 已有成熟优化路径。最终修正为：

- `stage0/head/stage2-local` 是端到端收益候选；
- `stage2/context` LiteMLA 是高区分度 Plugin 主线；
- 报告中必须区分“端到端收益排序”和“Plugin 展示价值排序”。

这个修正防止项目叙事从“非标准 LiteMLA Plugin”滑向“哪里最大就写哪里”的粗糙策略。

### 2.4 LiteMLA Plugin 边界需要由 Plan D 细化

早期候选容易停留在“整体 LiteMLA Plugin”。Plan D 显示 `aggregation`、`cat`、`relu_linear_att` 是更具体的可融合子路径。最终候选分层为：

1. `relu_linear_att-only`：Phase 3 Step 2 后收敛为第一版 MVP，真实 contract 为 `[1,384,64,128] -> [1,128,64,128]`；
2. `aggregation-only`：保留为 fallback / 对照实验；
3. `aggregation + cat + relu_linear_att`：主性能边界，真实 contract 为 `[1,192,64,128] -> [1,128,64,128]`；
4. 整体 LiteMLA Plugin：复杂 fallback / 上限方案。

这个修正把 Phase 3 从口号变成了可执行的 Plugin 边界设计。

---

## 3. Phase 2 关键纠偏摘要

完整记录见 [`phase2/design_notes/phase2_decision_corrections.md`](./phase2/design_notes/phase2_decision_corrections.md)。

### 3.1 TensorRT 端到端加速不等于候选仍成立

早期 Phase 2 设计更偏向 ONNX 导出、engine 构建和 latency benchmark。人工 review 指出：Phase 2 必须回答“TensorRT 自动优化后，Phase 1 的 Plugin 候选是否仍然成立”。最终补充：

- TensorRT Nsight Systems runtime attribution；
- EngineInspector / ONNX node name 映射；
- Phase 1 vs TensorRT 同名 group 近似 speedup；
- `stage2/context` LiteMLA residual attribution。

因此 Phase 2 不只是“TensorRT 有 1.57x 加速”，而是能说明 TensorRT 后的残余热点和 Phase 3 候选变化。

### 3.2 Phase 2 也必须以 Nsight runtime 归因为主

早期曾把 Nsight 视为辅助。最终修正为：

- CUDA Events latency 是端到端速度指标；
- Nsight runtime attribution 是 TensorRT 后热点复核主证据；
- EngineInspector 是结构证据，不能替代真实 GPU kernel duration。

这个修正与 Phase 1 方法论保持一致。

### 3.3 TensorRT 没有自动融合 LiteMLA 成单算子

Phase 2 结果显示 TensorRT 将 ONNX `393` nodes 压缩到 `155` engine layers，并完成大量 pointwise / Conv+Add fusion；但 `stage2/context` LiteMLA 仍残留 `qkv/Conv`、`aggregation Conv`、`Relu`、`Pad`、`MatMul`、`Add/Div`、`proj/Conv + Add` 等多个相关 layers。

最终口径：

- TensorRT 自动优化有效；
- LiteMLA 未被自动合成单个 fused operator；
- Phase 3 继续评估 LiteMLA Plugin 仍有依据。

### 3.4 不能把 TensorRT residual proxy 反向改写成 Phase 1 MVP

Phase 2 attribution 使用 TensorRT layer-name group，例如 `attention_core = relu_qk + pad + matmul + norm_add_div`。人工 review 指出，这只是 TensorRT 残余路径 proxy，不能把 Phase 1 的 MVP 候选改写成 `relu_qk-only`。

最终口径：

- Phase 1 MVP 仍是 `relu_linear_att-only` / `aggregation-only`；
- Phase 1 主性能边界仍是 `aggregation + cat + relu_linear_att`；
- Phase 2 `attention_core` 只是对应 `relu_linear_att` 内部残余路径的分析视角。

### 3.5 C++ Runtime Demo 是 Phase 2 必要工程链路

项目策略文档中 Phase 2 包含 TensorRT C++ 推理 Demo，但早期推进时被低估。最终补齐：

- C++ demo 加载 FP32 engine；
- 使用 TensorRT Runtime API 创建 runtime/context；
- 分配 CUDA buffer、绑定地址、执行推理并读取输出统计；
- 为 Phase 3 Plugin 注册和 C++ engine 加载预热。

这个修正确保 Phase 2 不只是 Python benchmark，而是向 Phase 3 C++/CUDA Plugin 工程自然过渡。

---

## 4. Phase 3 关键纠偏摘要

完整记录见 [`phase3/design_notes/phase3_decision_corrections.md`](./phase3/design_notes/phase3_decision_corrections.md)。

### 4.1 P1b-7 不能直接替代 P1a stage2+stage3

P1b-7 是 `stage2-only` 扩大边界实验，证明 `aggregation + cat + relu_linear_att` 能显著降低 stage2 中段 kernel time / launch 数；但它不能直接和覆盖 stage2+stage3 四个 LiteMLA context block 的 P1a 做公平对照。真正的全范围判断必须看 P1mix（stage2=P1b-7、stage3=P1a）。P1mix 未稳定优于 P1a stage2+stage3，因此当前主交付线仍是 P1a stage2+stage3。

### 4.2 扩大边界和减少 launch 数不是充分条件

Phase 3 证明，Plugin 边界更大不必然更快。P1b naive 版本把标准 aggregation Conv 路径移入自写 Plugin 后反而退化，说明自定义 Plugin 可能绕开 TensorRT/cuDNN 的成熟优化。后续 P1b-1 到 P1b-7 修复了部分问题，但最终仍需要同覆盖范围的端到端结果、Nsight attribution 和 mIoU gate 共同判断。

### 4.3 P1a 两阶段 kernel 是基于数学依赖的选择

`relu_linear_att` 需要先完成跨完整 spatial 维度的 `VK/Z` 归约，每个 spatial position 才能使用完整 `VK/Z` 计算输出。普通 CUDA kernel 没有普通全 grid 同步，因此不能只凭“少一次 launch”就强行合并为单 kernel。Phase 3 的 single-kernel prototype 已作为反证记录，当前主线保持两阶段 kernel。

### 4.4 MX250 测量纪律是结论可信度的一部分

MX250 低功耗、少 SM、无 Tensor Core，且 Windows 笔记本环境容易受温度、功耗墙和调度影响。Phase 3 多次用冷机/同口径复测修正热机误导，因此 1ms 级收益必须结合 CUDA Events、Nsight attribution、温度状态和重复测量判断。

### 4.5 mIoU gate 是最终主线采纳条件

单图 allclose / argmax agreement 只能说明调试输入上的数值路径健康，不能代表完整语义安全。Phase 3 最终 P1a stage2+stage3 通过 Cityscapes val mIoU gate，才形成“性能有收益且语义无回归”的主线结论。

---

## 5. 跨阶段执行纪律：先排除沙盒 / 权限因素

Phase 1/2/3 多次遇到同一类问题：命令本身需要访问 CUDA driver、CUPTI、Nsight Systems、TensorRT DLL、`.git` 或外部进程资源，但在 Codex 沙盒内表现为卡住、超时、权限错误或残留进程。人工 review 多次提醒后，最终形成跨阶段执行纪律：

- `nsys profile`、TensorRT engine benchmark、Plugin engine benchmark、CMake/CUDA build、git commit/push 等命令，如果在沙盒内出现异常，不能第一时间归因到脚本或模型逻辑。
- 应先检查是否属于沙盒 / 权限 / 外部进程交互问题；必要时用提权命令复跑，或先清理残留进程。
- Nsight Systems 在 Windows 普通权限下 CPU sampling / context switch trace 可能被禁用；只要 CUDA/NVTX trace 和 SQLite export 正常，仍可用于本项目的 GPU kernel attribution。
- 对 profiling 结果做技术结论前，必须确认采集路径本身可信。环境问题不能写成性能问题，工具权限问题不能写成 Plugin 问题。

这条纪律已经在 Phase 3 P1b Nsight attribution 中再次验证：沙盒内 `nsys profile` 曾超时；同一命令提权后正常生成 `.nsys-rep`、`.sqlite` 和 attribution summary，最终证明 P1b 退化来自 naive aggregation kernel，而不是 benchmark 脚本或 TensorRT engine 损坏。

---

## 6. 对项目路线的总体影响

这些纠偏共同形成了当前项目路线：

1. Phase 1 建立 PyTorch 端严格 profiling 方法论；
2. Phase 2 复核 TensorRT 自动优化后的 residual hotspot；
3. Phase 3 不盲目追最大热点，而是在 LiteMLA 非标准路径上做有数据支撑的 Plugin MVP；
4. 所有阶段都保留证据边界：什么是 latency，什么是 runtime attribution，什么只是结构证据，什么只能作为截图辅助。

项目后续推进时，如果 AI Agent 提出新的优化方向，必须先检查它是否违反上述纠偏后的方法论边界。

