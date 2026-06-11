# Phase 2 设计纠偏记录

> 本文记录 Phase 2 实施过程中由人工 review 纠正的关键设计决策与取舍问题。它服务于 Phase 2 的方法论审计：哪些地方曾经考虑不充分，最终如何修正，以及这些修正如何影响 Phase 3 Plugin 路线。

---

## 1. Phase 2 目标：不能只证明 TensorRT 有加速

**原问题**：早期 Phase 2 设计偏重 ONNX 导出、TensorRT engine 构建和端到端 benchmark，容易停留在“TensorRT 比 PyTorch 快多少”。

**纠正**：Phase 2 还必须回答一个更关键的问题：TensorRT 自动优化后，Phase 1 确定的瓶颈和 Plugin 候选是否仍然成立。

**最终口径**：

- 端到端 speedup 只是 Phase 2 的第一层结论。
- 必须补 TensorRT Nsight Systems runtime attribution。
- 必须比较 Phase 1 Plan B/C/D 与 TensorRT 后 residual hotspots。
- 必须明确哪些候选被 TensorRT 充分优化，哪些仍有自定义 Plugin 空间。

**影响**：Phase 2 从“部署跑通”升级为“为 Phase 3 候选复核提供证据”。

---

## 2. Nsight 在 Phase 2 中应是主证据，不是辅助

**原问题**：曾把 TensorRT Nsight profiling 视为 benchmark 后的辅助观察。

**纠正**：既然 Phase 1 的瓶颈定位依赖 Nsight attribution，Phase 2 判断“TensorRT 后瓶颈是否变化”也必须使用 Nsight runtime attribution 作为主证据。

**最终口径**：

- CUDA Events：负责 TensorRT 端到端 latency。
- Nsight SQLite attribution：负责 TensorRT runtime 组件 / layer group 的 GPU kernel time。
- EngineInspector：负责结构解释，不负责真实耗时排序。

**影响**：避免只凭 TensorRT 端到端 speedup 推断 TensorRT 优化了哪些模块。

---

## 3. EngineInspector：结构证据不能替代 runtime 归因

**原问题**：补 EngineInspector 后，容易把 engine layer count reduction 或 layer name fusion 直接解释为性能瓶颈消失。

**纠正**：EngineInspector 只能说明 TensorRT 图结构如何压缩和融合，不能说明 GPU kernel 实际耗时。

**最终口径**：

- ONNX `393` nodes -> TensorRT `155` engine layers 说明 TensorRT 做了结构优化。
- `PWN(...)`、layer name 中的 `+` fusion 说明 pointwise / Conv+Add 等路径被融合。
- 真实 runtime hotspot 仍以 Nsight attribution 为准。

**影响**：报告中把“TensorRT 优化了什么结构”和“TensorRT 后哪里仍耗时”分成两类证据。

---

## 4. TensorRT 没有自动把 LiteMLA 融成单算子

**原问题**：如果只看 TensorRT 端到端加速，可能误以为 TensorRT 已经处理掉了 LiteMLA Plugin 空间。

**纠正**：EngineInspector 和 Nsight attribution 都显示 `stage2/context` LiteMLA 仍由多个相关 engine layers 组成。

**最终口径**：

- TensorRT 做了局部 fusion。
- 但 `qkv/Conv`、`aggregation Conv`、`Relu`、`Pad`、`MatMul`、`Add/Div`、`proj/Conv + Add` 等 LiteMLA 相关 layers 仍残留。
- `stage2/context` 仍有可观 runtime 与 launch density。

**影响**：Phase 3 继续评估 LiteMLA Plugin 不是“逆着 TensorRT 优化做重复劳动”，而是针对 TensorRT 未能整体融合的非标准路径。

---

## 5. Phase 2 residual proxy 不能反向改写 Phase 1 Plugin MVP

**原问题**：TensorRT attribution 中出现 `relu_qk / pad / matmul / norm_add_div` 后，曾有风险把 Phase 1 的 MVP 候选改成 `relu_qk-only`。

**纠正**：这些名字是 TensorRT layer-name 视角下对 `relu_linear_att` 内部残余路径的拆分，不等同于 Phase 1 源码语义边界。

**最终口径**：

- Phase 1 MVP 仍是 `relu_linear_att-only` / `aggregation-only`。
- Phase 1 主性能边界仍是 `aggregation + cat + relu_linear_att`。
- Phase 2 的 `attention_core = relu_qk + pad + matmul + norm_add_div` 只是 TensorRT residual-runtime proxy。

**影响**：避免 Phase 3 候选命名和实现边界被 TensorRT layer name 带偏。

---

## 6. C++ Runtime Demo 是 Phase 2 必要任务

**原问题**：早期 Phase 2 推进更关注 Python 侧 ONNX/TensorRT build/benchmark，低估了项目策略文档中“TensorRT C++ 推理 Demo”的必要性。

**纠正**：Phase 3 Plugin 是 C++/CUDA/TensorRT 工程，Phase 2 必须提前跑通 C++ Runtime API 最小闭环。

**最终口径**：

- C++ demo 加载 FP32 engine。
- 使用 `IRuntime` / `ICudaEngine` / `IExecutionContext`。
- 分配 CUDA buffer，绑定 input/output device pointer。
- 执行推理、同步、读取输出统计。
- C++ demo 不作为严谨性能 benchmark。

**影响**：Phase 2 与 Phase 3 Plugin 集成路径衔接起来，不再停留在 Python API 层。

---

## 7. Phase 2 不以完整 mIoU 作为验收条件

**原问题**：Phase 2 是否需要测 mIoU 曾有讨论，容易把部署转换一致性和完整精度评测混在一起。

**纠正**：Phase 2 的目标是 TensorRT 转换链路与 baseline，不是完整语义分割精度评测阶段。

**最终口径**：

- Phase 2 使用 logits diff、relaxed allclose、cosine similarity、argmax pixel agreement 验证转换一致性。
- 不把 Cityscapes 全量 mIoU 作为 Phase 2 完成条件。
- 完整 mIoU 或更大样本集评估放到 Phase 3 Plugin 集成验证或最终验收阶段。

**影响**：避免 Phase 2 被数据集评测流程拖偏，同时保留部署转换的数值安全检查。

---

## 8. FP16 / 混合精度：风险实验不能包装成主线

**原问题**：TensorRT FP16 和混合精度是否应作为 Phase 2 主线曾有不确定性。

**纠正**：MX250 是 Pascal 架构，没有 Tensor Core；FP16 不一定加速。LiteMLA 原实现还存在 `autocast(enabled=False)` 的 FP32 保护，Phase 3 Plugin 需要单独设计数值策略。

**最终口径**：

- FP32 是本机 TensorRT 主 baseline。
- FP16 作为风险实验：可构建、语义一致，但慢于 FP32。
- 混合精度不作为 Phase 2 必做项，留到 Phase 3 Plugin 数值策略中讨论。

**影响**：避免把“看起来更高级的 FP16”误写成性能主线。

---

## 9. warmup / measure 需要沿用 Phase 1 口径

**原问题**：TensorRT Nsight / benchmark 初期可能为了方便使用较少次数。

**纠正**：为了和 Phase 1 baseline 可比，正式 TensorRT benchmark 与 Nsight attribution 应沿用 `warmup=20 / measure=100`。

**最终口径**：

- smoke / debug 可以少量迭代。
- 正式结果使用 `20 / 100`。
- 报告必须记录 warmup / measure。

**影响**：保证 Phase 1 PyTorch 与 Phase 2 TensorRT 的 latency 和 attribution 对比不被测量协议差异污染。

---

## 10. TensorRT 环境：pip 包不等于完整 Windows TensorRT 部署

**原问题**：早期尝试 pip TensorRT 路线时，容易忽视 MX250 `sm_61` 和 Windows DLL/header/lib 的部署边界。

**纠正**：本机需要 NVIDIA archived TensorRT 8.6.1 Windows zip 加 Python wheel 和显式 DLL path 注入。

**最终口径**：

- 新版 pip TensorRT / TensorRT 10+ 不适合当前 MX250。
- TensorRT 8.6.1 Windows zip 提供 runtime / builder / include / lib。
- Python 脚本在 import TensorRT 前显式准备 DLL path。
- C++ demo 需要 CMake / MSVC / include / lib / DLL PATH 全链路。

**影响**：让 Phase 2 环境从“能 import Python 包”变成“Python build/benchmark 与 C++ runtime 都可复现”。

---

## 11. SegHead bicubic Resize：当前支持不等于普遍支持

**原问题**：曾担心 SegHead `bicubic` upsample 是 TensorRT parser/build 阻塞项；构建通过后也容易把结论外推过宽。

**纠正**：当前只能说明固定 shape、TensorRT 8.6.1、当前 ONNX `Resize` 参数组合可 parse/build/runtime。

**最终口径**：

- 当前 `mode=cubic`、`half_pixel`、`cubic_coeff_a=-0.75` 可用。
- 不外推到动态 shape、其他 TensorRT 版本或其他 cubic 参数。
- 不需要在 Phase 2 强行改 bilinear。

**影响**：既避免过度担忧，也避免过度承诺。

---

## 12. 脚本复用：重复 helper 应抽到公共模块

**原问题**：Phase 2 多个脚本中出现重复的 sha256、版本记录、TensorRT runtime path、JSON 写入、git version 等 helper。

**纠正**：重复逻辑应集中到 `_common.py`、`_trt_runtime.py` 等公共模块，脚本只保留各自任务逻辑。

**最终口径**：

- 公共元信息和文件工具放 `_common.py`。
- TensorRT runtime path / DLL path / binding 相关逻辑放 `_trt_runtime.py`。
- 单脚本不再复制粘贴基础工具函数。

**影响**：减少后续 Phase 3 引入 Plugin 脚本时的维护成本，也让项目代码更像可持续工程而不是一次性实验脚本。

---

## 总结

Phase 2 的主要纠偏是把“TensorRT baseline 跑通”提升为“TensorRT 后候选复核”：

- 端到端 latency 证明 TensorRT 有效；
- Nsight attribution 证明 residual hotspots；
- EngineInspector 解释 TensorRT 做了哪些结构融合；
- C++ demo 验证后续 Plugin 集成链路；
- Phase 3 候选保留 Phase 1 源码语义边界，同时吸收 Phase 2 的 residual-runtime 证据。

