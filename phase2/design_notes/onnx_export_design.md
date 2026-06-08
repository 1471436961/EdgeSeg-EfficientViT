# `export_onnx.py` — 设计文档

> **关联阶段**：[`phase2/README.md`](../README.md)
> **关联 Phase 1 基线**：[`phase1/bottleneck_analysis_report.md`](../../phase1/bottleneck_analysis_report.md)
> **状态**：v1.2，`export_onnx.py` 与 `_compat.py` 已落盘；该 ONNX 已完成 ONNXRuntime 对齐、TensorRT 8.6.1 FP32/FP16 build 与 benchmark。

---

## 1. 设计目标

`export_onnx.py` 的职责是把 Phase 1 已验证的 EfficientViT-Seg-B0 PyTorch 模型导出为 ONNX，并产生可复现、可验证的导出记录。

它不是 TensorRT builder，也不是精度评估脚本。它只回答：

1. PyTorch 模型能否在固定输入下成功导出 ONNX。
2. ONNX 图是否结构合法。
3. ONNXRuntime 输出是否与 PyTorch 输出基本一致。
4. 导出产物是否记录了足够的复现元信息。

---

## 2. 执行流程

```text
parse_args()
setup_env()
install_compat_patches_if_needed()
load_model()
prepare_input()
run_pytorch_reference()
export_onnx()
check_onnx_model()
run_onnxruntime_validation()
assemble_export_metadata()
save_metadata_json()
print_summary()
```

第一版导出只支持固定 shape：

```text
input:  1 x 3 x 1024 x 2048
output: 1 x 19 x 128 x 256
```

---

## 3. 命令行接口计划

```powershell
python phase2/scripts/export_onnx.py `
  --weights phase1/weights/efficientvit_seg_b0_cityscapes.pt `
  --input phase1/data/city_asset_cityscapes_like.png `
  --resolution 1024x2048 `
  --output phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx `
  --metadata phase2/results/metrics/onnx_export_b0_cityscapes_1024x2048.json
```

计划参数：

| 参数 | 含义 |
|---|---|
| `--weights` | 真实 Cityscapes B0 权重，正式导出必填 |
| `--input` | 固定样图路径 |
| `--resolution` | 固定输入分辨率，默认 `1024x2048` |
| `--output` | ONNX 输出路径 |
| `--metadata` | 导出元信息 JSON 路径 |
| `--opset` | ONNX opset，第一版默认先按当前 PyTorch 推荐稳定值选择 |
| `--skip-ort` | 仅导出与 checker，不跑 ONNXRuntime；用于依赖不全时临时诊断 |

---

## 4. 关键取舍

### D1：固定 shape vs 动态 shape

**选择：固定 `1x3x1024x2048`。**

原因：

- Phase 1 baseline 就是 Cityscapes 原生分辨率。
- LiteMLA 内部存在 `H*W > dim` 的形状分支，固定 shape 可以让 trace 只保留当前部署实际需要的路径。
- Phase 2 目标是建立可靠 TensorRT baseline，不是先做通用动态输入部署。

备选：

- 后续如果确实需要动态 shape，可在 Phase 2 后半另设 `export_onnx_dynamic_design.md`，重新处理 shape branch 与 TensorRT optimization profile。

### D2：第一版保持 bicubic vs 直接改 bilinear

**选择：第一版保持上游 bicubic。**

原因：

- Phase 2 首轮应先记录真实上游模型在 ONNX/TensorRT 下的行为。
- 如果 TensorRT parser/build 因 bicubic 失败，这本身就是重要工程证据。
- 直接改 bilinear 会改变模型语义，必须配套数值/精度说明。

实测结果：

- ONNX 中有 2 个 SegHead `Resize` 节点，属性为 `mode=cubic`、`coordinate_transformation_mode=half_pixel`、`cubic_coeff_a=-0.75`。
- TensorRT 8.6.1 在当前固定 shape 下 parser/build/runtime 均通过。
- 因此第一版不需要把 SegHead bicubic 改成 bilinear，也不把 head resize Plugin 作为当前候选。

边界：

- 该结论只覆盖当前固定输入 shape、opset 17、TensorRT 8.6.1 和上述 Resize 参数；动态 shape、其他 TensorRT 版本或其他 cubic 参数组合仍需单独复验。

### D3：先 ONNXRuntime validation vs 直接 TensorRT

**选择：先 ONNXRuntime validation。**

原因：

- TensorRT 输出异常时，需要先排除 ONNX export 本身错误。
- ONNXRuntime validation 能提供 PyTorch vs ONNX 的第一层数值对齐证据。

备选：

- 若 ONNXRuntime 在本机安装困难，可用 `--skip-ort` 只做导出与 checker，但不能把该产物标为完整验证通过。

### D4：FP32 baseline vs 直接 FP16

**选择：先 FP32。**

原因：

- Phase 1 PyTorch baseline 是 FP32。
- LiteMLA `relu_linear_att` 存在 autocast-disabled 数值策略，FP16 应作为第二阶段风险实验。
- FP32 baseline 成功后，FP16 失败也能被清楚定位为精度/构建策略问题，而不是导出链路问题。

### D5：legacy `torch.onnx.export` vs 新 dynamo exporter

**选择：第一版使用 legacy `torch.onnx.export`，`opset=17`。**

原因：

- 当前 PyTorch 2.4.1 的 `ONNX_DEFAULT_OPSET` 是 17，适合作为第一版稳定基线。
- Phase 2 第一目标是建立可验证导出链路，不追求最新 exporter API。
- 若 legacy 导出失败，再单独设计 dynamo exporter 路线，避免同时引入两个变量。

### D6：compat 内联 vs `_compat.py`

**选择：抽出 `phase2/scripts/_compat.py`。**

原因：

- Phase 2 后续的 TensorRT build / inference 脚本也可能需要导入 EfficientViT。
- `triton_stub` 与 `wandb_stub` 已经是跨脚本 import 兼容逻辑，继续复制会增加维护风险。
- `_compat.py` 只负责 import-time patch，不改变模型数学。

---

## 5. 数值对齐口径

第一版只比较单张固定输入的 logits 输出，不做 mIoU。

计划记录：

| 指标 | 含义 |
|---|---|
| `max_abs_diff` | 最大绝对误差 |
| `mean_abs_diff` | 平均绝对误差 |
| `max_rel_diff` | 最大相对误差，需避免除零 |
| `cosine_similarity` | 展平 logits 后的余弦相似度 |
| `allclose_pass` | 按阈值判断是否通过 |

初始阈值建议：

```text
FP32 PyTorch vs ONNXRuntime:
  atol = 1e-4
  rtol = 1e-4
```

这只是导出 sanity 阈值，不等同于 mIoU 结论。若误差略高但可解释，需要在 metadata 中记录原因。

---

## 6. 元信息 JSON Schema 草案

```jsonc
{
  "status": "ok",
  "model_name": "efficientvit_seg_b0",
  "dataset": "cityscapes",
  "input_resolution": [1024, 2048],
  "batch_size": 1,
  "dtype": "float32",

  "weights_path": "phase1/weights/efficientvit_seg_b0_cityscapes.pt",
  "weights_sha256": "...",
  "input_path": "phase1/data/city_asset_cityscapes_like.png",
  "input_sha256": "...",

  "onnx_path": "phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx",
  "opset": 17,
  "input_names": ["input"],
  "output_names": ["segout"],
  "dynamic_axes": null,

  "versions": {
    "python": "...",
    "torch": "...",
    "onnx": "...",
    "onnxruntime": "..."
  },

  "validation": {
    "onnx_checker_pass": true,
    "onnxruntime_ran": true,
    "max_abs_diff": 0.0,
    "mean_abs_diff": 0.0,
    "max_rel_diff": 0.0,
    "cosine_similarity": 1.0,
    "allclose_pass": true,
    "atol": 1e-4,
    "rtol": 1e-4
  },

  "known_risks": [
    "fixed_shape_export",
    "fixed_shape_bicubic_resize_verified_on_tensorrt_8_6_1",
    "litemla_shape_branch_frozen"
  ],

  "timestamp_utc": "...",
  "script_version": "export_onnx.py@<git_commit>"
}
```

---

## 7. 兼容补丁策略

Phase 1 已遇到：

- Windows 上 `triton` import 链问题。
- `wandb` 退出临时目录清理 PermissionError。

Phase 2 第一版可以复用同类策略，但要保持透明：

- compat patch 只服务 import / export，不改变模型数学。
- 若注入 `triton_stub`，metadata 中记录 `env_patches=["triton_stub"]`。
- 若禁用 wandb，metadata 或 stdout 中记录。
- 不把 compat patch 藏在脚本深处；应放在清晰函数中，例如 `_install_import_compat_patches()`。

第一版已抽出：

```text
phase2/scripts/_compat.py
```

`export_onnx.py` 在 `import torch` 之后、第一次 `import efficientvit.*` 之前调用 `_compat.install_import_compat_patches()`。

---

## 8. 官方资料口径

设计参考：

- PyTorch ONNX export 官方文档：[`torch.onnx`](https://docs.pytorch.org/docs/stable/onnx.html)。导出会基于示例输入捕获模型计算图；opset、输入/输出名、dynamic shape 等都应显式记录。
- PyTorch torch.export-based ONNX exporter 官方说明：[`torch.export-based ONNX Exporter`](https://docs.pytorch.org/docs/stable/onnx_export.html)。该路径会记录 shape constraints；这支持本阶段先固定输入分辨率，避免 LiteMLA shape branch 语义变复杂。
- NVIDIA TensorRT ONNX parser 官方文档：[`Onnx Parser`](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/parsers/Onnx/pyOnnx.html)。parser / builder 错误应显式读取并记录，不能只给出“build failed”。

这些资料只用于确定 Phase 2 的工程口径；具体脚本实现以后续本机环境验证为准。

---

## 9. 当前结果与下一步

已完成：

1. 当前环境已安装 `onnx==1.21.0` / `onnxruntime==1.23.2`。
2. 第一版使用 `opset=17`。
3. 第一版已引入 `phase2/scripts/_compat.py`，不再内联 compat patch。
4. ONNX 导出、`onnx.checker` 与 ONNXRuntime 对齐验证已通过。
5. 该 ONNX 已成功构建 TensorRT 8.6.1 FP32 / FP16 engine，并完成 runtime benchmark。

下一步：撰写 `phase2/tensorrt_baseline_report.md`，其中明确 FP32 是本机主 baseline，FP16 是已验证但不建议采用的风险实验。
