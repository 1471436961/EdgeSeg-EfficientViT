# PROJECT_CONVENTIONS.md — EdgeSeg-EfficientViT 项目协作契约

> **文件性质**：横切契约（cross-cutting contract），约束本项目中"AI 与用户如何协作"，跨所有阶段、所有代码生效。
>
> **位置说明**：本文件与 `PROJECT_STRATEGY.md`、`LEARNING_LOG.md` 并列于项目 git 根目录，**不属于** `PROJECT_STRATEGY.md` 战略层的子章节，也**不属于**任何单一 phase 的执行层。它是元规则（meta-rule）。
>
> **修订规则**：本契约的任何修改本身必须遵守 §1 的三段式流程。
>
> **当前版本**：v1.2 · 创建时间：2026-05-26 · 最近更新：2026-06-10
>
> **相关文档**：
> - 项目战略：[`PROJECT_STRATEGY.md`](./PROJECT_STRATEGY.md)
> - 阶段执行：`phase{N}/README.md`
> - 代码设计：脚本 docstring 或 `phase{N}/design_notes/xxx_design.md`

---

## §1 AI 协作工作流契约（核心契约）

### §1.1 三段式提交流程

**所有会产生持久化产物或不可逆变更的 AI 行为，必须严格按以下三步执行**：

```
  ① 解释决策  →  ② 用户确认  →  ③ 落盘 / 执行
```

| 步骤 | 含义 | AI 的义务 | 用户的义务 |
|---|---|---|---|
| ① 解释决策 | AI 用自然语言说明"打算做什么、为什么这么做、有什么取舍" | 必须输出 §1.4 规定的 5 要素 | — |
| ② 用户确认 | 用户在对话中给出明确回应 | 等待，不抢跑 | 给出 `[采纳](!send)` 或具体修改意见 |
| ③ 落盘 / 执行 | AI 调用工具进行实际操作 | 按确认后的方案执行；执行后简短汇报结果 | — |

**禁止行为**：
- ❌ 跳过 ① 直接落盘（"我先写好你看一下"——不允许）
- ❌ 在 ① 中夹带未声明的额外变更（"顺手把另一个文件也改了"——必须先声明）
- ❌ 把多个独立决策打包成"一次确认"，让用户无法逐项把关

### §1.2 何时触发三段式

**强制触发**（必须走三段式）：

| 类别 | 具体场景 |
|---|---|
| **代码变更** | 新建脚本 / 修改 ≥ 10 行的现有代码 / 删除任何代码 |
| **架构决策** | 选择库 / 选择算法路线 / 设计目录结构 / 选择数据格式 |
| **文档新建** | 创建任何 `.md` 文档 / 创建任何 `design_notes/` 文档 |
| **文档大改** | 修改 `PROJECT_STRATEGY.md` 任何章节 / 修改 `PROJECT_CONVENTIONS.md` 任何条款 / 修改 phase README 的"决策"或"任务清单"区段 |
| **git 操作** | 创建分支 / merge / rebase / 推送到 remote / tag |
| **环境变更** | 安装新依赖 / 修改 conda env / 修改 PATH |
| **数据/权重变更** | 下载新权重 / 修改数据集预处理逻辑 |

### §1.3 例外白名单（无需三段式，但仍需简短说明）

| 类别 | 具体场景 |
|---|---|
| **纯查询** | 读源码 / 查文档 / 运行只读命令（`ls` `git status` `nvidia-smi` 等）|
| **纯解释** | 回答概念问题 / 解读已有代码 / 总结对话 |
| **明显笔误** | 拼写错误 / 路径中的反斜杠方向错误 / Markdown 渲染问题 |
| **用户明确放权** | 用户在本轮消息中明确说"直接做"/"不用问了"/"按你说的来" |
| **格式化微调** | 调整缩进 / 加换行 / 修正引号风格（不改变语义）|

⚠️ **白名单边界守则**：当 AI 不确定某个动作是否落入白名单时，**默认按强制触发处理**（即先解释、再确认）。宁可慢一拍，不可越界。

### §1.4 决策说明应包含的 5 要素（必含，顺序自由）

每次"① 解释决策"的输出，必须包含以下 5 个要素。允许灵活组织顺序，允许合并讲述（例如把"取舍"和"风险"合在一起），但**不允许遗漏**：

| 要素 | 含义 | 示例提问形式 |
|---|---|---|
| **① 设计思路** | 整体方案是什么、怎么落地 | "我打算用 …… 来 ……" |
| **② 关键取舍** | 存在哪些可选项、它们的差异是什么 | "可以选 A 或 B，A 的特点是 ……，B 的特点是 ……" |
| **③ 选择原因** | 为什么倾向这个方案（结合本项目目标） | "我选 A，因为 ……" |
| **④ 已知风险** | 这个方案可能在哪里翻车 | "风险点：……。如果发生，我会 ……" |
| **⑤ 备选方案** | 如果用户不认可，退路是什么 | "如果你倾向另一种思路，可以改用 ……" |

**简单决策的写法**：对于明显简单的变更（例如新增一个 5 行的工具函数），5 要素可以高度压缩成一段话，但仍需点到为止。

### §1.5 违约处理

| 违约情形 | 处理方式 |
|---|---|
| AI 跳过 ①，直接落盘 | 用户可要求"撤回"，AI 必须立即执行 git revert / 删除文件 / 还原现场，并补充事后说明 |
| AI 在 ① 中遗漏 5 要素 | 用户可要求"补完决策说明"，AI 必须补齐后再请求确认 |
| AI 在 ③ 中夹带未声明的变更 | 用户可要求"剔除越界变更"，AI 必须 revert 越界部分并保留已确认部分 |
| 用户跳过 ②（默认 AI 已经确认） | AI 应主动提醒："这一步我需要你的明确确认，可以是 `[采纳](!send)` 或你的修改意见" |
| 用户改主意（已落盘后想撤销） | 走 §1.4 流程提出新决策即可，AI 不阻拦撤回 |

### §1.6 三段式的"快通道"协议

为了避免在简单场景下被流程拖慢，约定以下快通道：

- **批量预确认**：AI 一次性列出 N 个独立小决策，用户用一句 `[全部采纳默认](!send)` 一次性确认
- **链式确认**：用户在确认时附加"……顺便把 XX 也做了"，视为对附加项的当场确认，AI 无需再次询问
- **隐式确认**：用户的下一轮消息直接基于 AI 上一轮的决策展开（例如 AI 提议方案 A，用户回复"那 A 方案的 X 步骤会不会有问题"），视为对方案 A 的隐式确认

---

## §2 文档体系与三层结构（项目 / 阶段 / 代码）

本项目的所有文档按"承载内容的粒度"分为三层：

### §2.1 项目级（Project-level）

- **职责**：回答"我们要做什么、为什么、整体怎么做"
- **文件**：
  - `PROJECT_STRATEGY.md` — 战略主文档（三阶段路线、跨阶段决策、版本演进 V3.x）
  - `PROJECT_CONVENTIONS.md` — AI 协作契约（三段式、文档分层、路径/提交约定）
  - `PROJECT_DECISION_CORRECTIONS.md` — 跨阶段设计纠偏总账，记录人工 review 如何修正关键方案
  - `LEARNING_LOG.md` — 学习笔记与技术问答沉淀
- **位置**：项目 git 根目录（`E:\EdgeSeg-EfficientViT\EdgeSeg-EfficientViT\`）
- **变更频率**：低（每个大版本一次，V3.1 / V3.2 …）
- **读者优先级**：面试官 > 未来的项目接手者 > 当前的你

### §2.2 阶段级（Phase-level）

- **职责**：回答"本阶段的目标、范围、产出、当前状态"
- **文件**（均位于 git 仓库的 `phase{N}/` 下）：
  - `phase{N}/README.md` — 阶段总览：任务清单 / 决策记录 / 进度跟踪
  - `phase{N}/architecture_analysis.md` — 架构剖析（如有）
  - `phase{N}/bottleneck_analysis_report.md` — 测量报告（如有）
  - `phase{N}/{其他主题}.md` — 单一主题的阶段性分析
- **变更频率**：中（每个阶段开始时建立，结束时定稿）
- **读者优先级**：当前阶段的执行者 > 跨阶段对照阅读时的你

### §2.3 代码级（Code-level）

- **职责**：回答"这段代码为什么这样写、有哪些取舍"
- **承载形式**（二选一）：
  - **首选：脚本 docstring** — 在 `.py` 文件开头写一段 docstring，覆盖 §1.4 的 5 要素
  - **次选：单独成文** — `phase{N}/design_notes/xxx_design.md`
- **拆分阈值**（B 建议，已确认）：
  - 脚本 ≤ 200 行 ⇒ 用 docstring 即可
  - 脚本 > 200 行 **或** 包含 ≥ 3 个非平凡决策 ⇒ 单独成 `design_notes/xxx_design.md`
- **命名规范**：`{脚本主名}_design.md`，例如：
  - `baseline_inference.py` ↔ `baseline_inference_design.md`
  - `onnx_export.py` ↔ `onnx_export_design.md`
  - `lite_mla_plugin.cu` ↔ `lite_mla_plugin_design.md`
- **变更频率**：高（每个非平凡脚本一份）
- **读者优先级**：未来改这段代码的人（包括 AI）

### §2.4 三层的本质区别速查表

| 层级 | 关心 | 粒度 | 变更频率 | 位置 |
|---|---|---|---|---|
| 项目级 | **Why** 为什么做这个项目 | 跨所有阶段 | 低 | 仓库根目录 |
| 阶段级 | **What** 这一阶段做什么 | 单个 phase | 中 | 仓库内 `phase{N}/` |
| 代码级 | **How** 这段代码怎么实现 | 单个脚本/函数 | 高 | 脚本内 / `design_notes/` |

### §2.5 横切契约的归属

`PROJECT_CONVENTIONS.md` 本身**不属于这三层中的任何一层**，是横切关注点，独立存在于项目 git 根目录。

---

## §3 分支与提交规范

### §3.1 分支命名

- 阶段分支命名：`phase{N}-{主题}`，例如 `phase1-baseline`、`phase2-tensorrt`、`phase3-plugin`。
- `master` 只承载阶段完成后的稳定结果；每个 phase 完整验收后再合入。
- 不在 `master` 上直接做阶段开发。

### §3.2 提交身份

每个阶段允许使用阶段化 git identity，便于从 commit log 看出阶段归属：

| 阶段 | user.name | user.email |
|---|---|---|
| Phase 1 | `EdgeSeg-Phase1` | `phase1@edgeseg.local` |
| Phase 2 | `EdgeSeg-Phase2` | `phase2@edgeseg.local` |
| Phase 3 | `EdgeSeg-Phase3` | `phase3@edgeseg.local` |

若使用真实 GitHub identity，也必须保证 commit message 能看出阶段和改动范围。

### §3.3 Commit Message

Commit message 使用：

```text
type(scope): subject
```

允许的 `type`：

| type | 用途 |
|---|---|
| `feat` | 新功能 / 新脚本 / 新阶段产物 |
| `fix` | bug 修复 / 口径错误修正 |
| `docs` | 文档与报告 |
| `refactor` | 不改变行为的代码整理 |
| `perf` | 性能优化或性能实验 |
| `test` | 测试与验证 |
| `chore` | 环境、目录、gitignore、清理 |

示例：

```text
docs(phase2): add TensorRT baseline report
refactor(phase2): share common script helpers
fix(phase1): correct NVTX timing description
```

### §3.4 Git 操作边界

- `commit`、`merge`、`push`、`tag` 属于 §1.2 的强制触发场景，必须先说明再执行，除非用户本轮明确要求。
- 提交前必须检查 `git status --short`，只 stage 与当前任务相关的文件。
- 不得用 `git reset --hard`、`git checkout --` 等 destructive 操作回滚用户改动，除非用户明确要求。

---

## §4 文件路径与命名规则

### §4.1 项目级文件

以下文件位于项目 git 根目录，并应入 git：

| 文件 | 职责 |
|---|---|
| `README.md` | 项目入口，优先展示个人工作 |
| `UPSTREAM_README.md` | 上游 EfficientViT 原始 README |
| `PROJECT_STRATEGY.md` | 项目战略与阶段路线 |
| `PROJECT_CONVENTIONS.md` | 协作契约 |
| `PROJECT_DECISION_CORRECTIONS.md` | 跨阶段设计纠偏总账 |
| `LEARNING_LOG.md` | 学习笔记与技术问答沉淀 |

### §4.2 阶段目录

阶段目录统一使用：

```text
phase{N}/
|-- README.md
|-- design_notes/
|-- scripts/
|-- results/
`-- logs/
```

阶段报告放在 `phase{N}/` 根下，例如：

- `phase1/bottleneck_analysis_report.md`
- `phase2/tensorrt_baseline_report.md`

代码级设计文档放在 `phase{N}/design_notes/`，命名为 `{script_or_topic}_design.md`。

### §4.3 结果文件入库边界

默认不入 git：

- 权重：`*.pt`、`*.pth`
- ONNX：`*.onnx`
- TensorRT engine：`*.engine`
- Nsight 原始结果：`*.nsys-rep`、`*.sqlite`
- 构建产物：`build/`、`__pycache__/`、`.pyc`

可以入 git：

- 小型 JSON metadata / benchmark summary；
- Nsight attribution 的 Markdown / JSON 汇总；
- 截图和报告用 figures；
- `.gitkeep` 占位文件。

### §4.4 归档与重命名

- 不再使用的旧文档如需保留，后缀使用 `_archived_{YYYYMMDD}.md`。
- 上游 README 不删除，重命名为 `UPSTREAM_README.md`；根 `README.md` 用作个人项目入口。
- 不静默删除上游源码目录，除非单独设计 release/package 流程。

---

## §5 工具与环境约定

### §5.1 Python / Conda

- conda env：`efficientvit`
- 典型路径：`D:\software\anaconda3\envs\efficientvit`
- Windows shell 中裸 `python` 可能命中 Windows Store stub；需要严肃验证时优先使用显式路径：

```powershell
D:\software\anaconda3\envs\efficientvit\python.exe
```

### §5.2 PyTorch / CUDA

- PyTorch：`2.4.1+cu124`
- 选择原因：兼容 Pascal `sm_61`；不随意升级到 PyTorch 2.7+。
- CUDA Events 是 Phase 1/2 latency 主计时工具。
- `torch.cuda.synchronize()` 只放在 warmup/measure 边界或 Event 读取处，不放进 NVTX range 内。

### §5.3 Nsight Systems

- Nsight Systems：`2026.2.1`
- 典型路径：`D:\software\nsight_systems\target-windows-x64\nsys.exe`
- Windows 普通权限下以 `cuda,nvtx` 为主 trace 口径。
- CPU sampling / context switch / WDDM trace 通常需要管理员权限；普通权限结果不能声称完全排除 CPU/WDDM 因素。
- 组件耗时以 SQLite attribution 为主，不直接使用 NVTX range 的 `end-start`。

### §5.4 TensorRT

- 当前可用 TensorRT：NVIDIA archived TensorRT `8.6.1` Windows zip。
- TensorRT root：`E:\NVIDIA\TensorRT-8.6.1.6`
- 选择原因：MX250 是 Pascal `sm_61`，新版 TensorRT / TensorRT 10+ 路线不适合作为本机主 baseline。
- Python 脚本应在 `import tensorrt` 前准备 DLL path。
- C++ demo 需要同时满足 TensorRT include/lib、CUDA、cuDNN/cuBLAS/NVRTC DLL runtime path。

### §5.5 测量协议

- 正式 Phase 1/2 benchmark 使用 `warmup=20 / measure=100`。
- smoke / debug 可以降低次数，但结果不得写成正式性能结论。
- Phase 1/2 不以完整 Cityscapes mIoU 作为阶段完成条件；完整精度回归放到 Phase 3 Plugin 集成验证或最终验收。

---

## §6 AI 输出语言与格式

### §6.1 语言

- 对话回复：中文。
- 项目文档：中文，保留必要英文术语和代码标识符。
- 代码注释：英文为主；关键项目决策点可中英混排，但不要写无意义注释。
- 报告结论：使用保守、可证据支撑的中文表述，避免夸大。

### §6.2 Markdown 格式

- 标题使用 `#` 到 `####`，不跳级过多。
- 决策、结果和对比优先使用表格。
- 流程使用编号列表。
- 真实文件引用尽量使用相对 Markdown 链接。
- 报告中必须区分：
  - 端到端 latency；
  - Nsight runtime attribution；
  - EngineInspector / graph structure evidence；
  - 截图观察；
  - 推测或后续计划。

### §6.3 术语口径

- `stage0/stage1/stage2/stage3` 默认指代码 / NVTX 中的 `backbone.stages` 索引，不是论文语义 stage 编号。
- “LiteMLA 是 Plugin 主线”不等于“LiteMLA 是全模型最大瓶颈”。
- Phase 2 的 `attention_core` 是 TensorRT residual-runtime proxy，不反向替代 Phase 1 的 `relu_linear_att` 源码语义边界。
- `smoke` 只证明链路能跑，不作为正式性能结论。

---

## §7 契约的修订流程

修改本契约的任何条款**必须遵守 §1 的三段式流程**，即使是修改 §1 本身。

- 每次修订需在文件顶部更新 `当前版本` 和 `创建时间`
- 大改（语义变化）：v1.0 → v2.0
- 小改（措辞、补例）：v1.0 → v1.1
- 修订记录追加到文件末尾的"修订历史"

---

## 修订历史

| 版本 | 日期 | 修订内容 | 触发原因 |
|---|---|---|---|
| v1.2 | 2026-06-10 | 补完 §3~§6：分支/提交、文件路径、环境工具、语言格式；新增 `PROJECT_DECISION_CORRECTIONS.md` 的项目级职责 | Phase 2 收口前需要把已实际执行的规则固化，避免进入 Phase 3 后口径漂移 |
| v1.1 | 2026-06-06 | 将 `PROJECT_CONVENTIONS.md`、`PROJECT_STRATEGY.md`、`LEARNING_LOG.md` 纳入项目 git 根目录，并修正文档分层中的位置说明 | 用户决定把三个项目级文档从外层目录移动到项目 git 目录 |
| v1.0 | 2026-05-26 | 初版建立，§1 完整、§2 完整、§3~§7 大纲占位 | 用户在三段式讨论中明确要求"将工作约定写入 PROJECT_CONVENTIONS.md §1" |
