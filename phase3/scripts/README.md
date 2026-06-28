# Phase 3 Scripts Index

本目录保存 Phase 3 的可复现实验脚本。脚本数量较多，是因为 Phase 3 同时覆盖：

- P1a `relu_linear_att-only` 主线。
- P1b `aggregation + cat + relu_linear_att` 消融路线。
- TensorRT engine build / benchmark / Nsight attribution。
- Cityscapes mIoU accuracy gate。

当前不建议把脚本移动到子目录，因为多数脚本依赖 `Path(__file__).resolve().parents[2]` 和同目录直接 import；移动路径会牵连命令、历史 metadata 和脚本间导入。若后续需要大整理，应先抽公共 helper，再移动实验分支脚本。

## 1. P1a 主线脚本

| 脚本 | 用途 |
|---|---|
| [`build_plugin_toy_engine.py`](build_plugin_toy_engine.py) | 构建最小 P1a toy engine，验证 Plugin Creator / parser / build 闭环 |
| [`validate_relu_linear_attention_plugin.py`](validate_relu_linear_attention_plugin.py) | P1a 单层 correctness 验证 |
| [`benchmark_relu_linear_attention_plugin.py`](benchmark_relu_linear_attention_plugin.py) | P1a 单层 microbenchmark 与 kernel-level profiling 输入 |
| [`integrate_relu_linear_attention_plugin_onnx.py`](integrate_relu_linear_attention_plugin_onnx.py) | 将 P1a Plugin node 集成到真实 EfficientViT ONNX graph |
| [`build_plugin_engine.py`](build_plugin_engine.py) | 构建真实 P1a Plugin TensorRT engine |
| [`benchmark_plugin_engine.py`](benchmark_plugin_engine.py) | 对比 TensorRT baseline 与 Plugin engine 的 correctness / latency |
| [`analyze_plugin_nsys_attribution.py`](analyze_plugin_nsys_attribution.py) | 分析 Plugin engine Nsight SQLite attribution |

## 2. Cityscapes mIoU 脚本

| 脚本 | 用途 |
|---|---|
| [`prepare_cityscapes_eval_manifest.py`](prepare_cityscapes_eval_manifest.py) | 扫描本地 Cityscapes val 数据，生成可复现 manifest |
| [`evaluate_cityscapes_miou.py`](evaluate_cityscapes_miou.py) | 对 TensorRT baseline / Plugin engine 跑 Cityscapes mIoU 与 semantic regression |

## 3. P1b 消融脚本

| 脚本 | 用途 |
|---|---|
| [`build_p1b_plugin_toy_engine.py`](build_p1b_plugin_toy_engine.py) | 构建 P1b toy engine，验证带 aggregation weight initializer 的 parser/build |
| [`integrate_p1b_aggregation_attention_plugin_onnx.py`](integrate_p1b_aggregation_attention_plugin_onnx.py) | 将 P1b Plugin node 集成到真实 EfficientViT ONNX graph |
| [`build_p1b_plugin_engine.py`](build_p1b_plugin_engine.py) | 构建真实 P1b Plugin TensorRT engine |
| [`capture_p1b_stage2_reference.py`](capture_p1b_stage2_reference.py) | 捕获 P1b block-level PyTorch reference 与权重 / tensor contract |
| [`validate_p1b_aggregation_attention_plugin.py`](validate_p1b_aggregation_attention_plugin.py) | 验证 P1b Plugin 的 block-level correctness |

## 4. 后续整理建议

Phase 3 集成报告已经完成。若后续还要继续清理脚本目录，建议顺序是：

1. 抽出 `phase3/scripts/_common.py`，集中管理 repo root、TensorRT runtime path、metadata writer、hash/version helper。
2. 再考虑把 P1b 消融脚本移到 `phase3/scripts/experiments/p1b/`。
3. 移动后必须同步修改所有命令示例、文档链接和 `script_version` 记录说明。

在此之前，保持脚本在同一目录更有利于复现实验。
