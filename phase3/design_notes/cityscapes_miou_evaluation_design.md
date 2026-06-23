# Phase 3 Cityscapes mIoU Evaluation Design

> 状态：落盘设计与脚本实现。
> 目标：为最终采用的 TensorRT Plugin engine 增加 Cityscapes mIoU / semantic regression 证据链，补齐 Phase 3 不能只看 latency 的验收口径。

## 1. 设计目标

Phase 3 已经通过 logits diff、relaxed allclose、cosine similarity 和 argmax pixel agreement 证明 Plugin engine 与 TensorRT baseline 在固定样图上语义一致。但这些指标仍不是数据集级精度指标，因此新增 mIoU accuracy gate：

1. 对同一组 Cityscapes val 样本运行 Phase 2 TensorRT FP32 baseline 与 Phase 3 Plugin FP32 engine。
2. 使用同一输入预处理、同一 logits upsample 与同一 label 映射口径。
3. 输出 baseline / plugin mIoU、per-class IoU、plugin-baseline mIoU delta、valid label 上的 argmax agreement。

## 2. 数据准备

Cityscapes 需要账号授权，项目不自动下载、不提交原始数据。数据应放在仓库外，或本地忽略目录：

```text
phase3/data/cityscapes/
|-- leftImg8bit/
|   `-- val/<city>/*_leftImg8bit.png
`-- gtFine/
    `-- val/<city>/*_gtFine_labelIds.png
```

新增脚本：

```powershell
D:\software\anaconda3\envs\efficientvit\python.exe phase3\scripts\prepare_cityscapes_eval_manifest.py `
  --cityscapes-root phase3\data\cityscapes `
  --split val `
  --label-kind labelIds `
  --output phase3\results\metrics\cityscapes_val_manifest.json
```

该脚本只校验本地文件与生成 manifest，不负责绕过 Cityscapes 授权下载。

## 3. 评估口径

### 3.1 输出分辨率

EfficientViT-Seg `segout` 是 H/8。对 Cityscapes `1024x2048` 输入，TensorRT engine 输出为 `[1,19,128,256]`。mIoU 评估时必须先把 logits resize 到 label 分辨率，再 `argmax`。

采用口径：

```text
segout logits [1,19,128,256]
  -> bicubic resize, align_corners=False
  -> [1,19,1024,2048]
  -> argmax
  -> compare with trainId label
```

这与上游 `applications/efficientvit_seg/eval_efficientvit_seg_model.py` 中 `resize(output, size=mask.shape[-2:])` 的语义一致。

### 3.2 label 映射

默认使用 `gtFine_labelIds.png`，按上游 `CityscapesDataset.label_map` 映射到 19 类 trainId；忽略类记为 `-1`，不进入 confusion matrix。脚本也支持 `labelTrainIds`，此时 255 视为 ignore。

### 3.3 preprocess 双口径

这里必须区分两个事实：

| 模式 | 含义 | 用途 |
|---|---|---|
| `official` | `[0,1]` 后使用 ImageNet mean/std normalize | 对齐上游 Cityscapes eval，应作为正式 mIoU 口径 |
| `deployment` | 只缩放到 `[0,1]`，不做 mean/std | 对齐 Phase 2/3 既有 latency benchmark 输入，只能作为 regression check |

Phase 2/3 的 latency 结果使用 `deployment` 口径，因为输入数值分布不影响固定 shape engine 的 kernel 序列与 latency 结论。mIoU 则应优先使用 `official` 口径，否则不能与上游 Cityscapes mIoU 对比。

## 4. 评估命令

小集 smoke：

```powershell
D:\software\anaconda3\envs\efficientvit\python.exe phase3\scripts\evaluate_cityscapes_miou.py `
  --manifest phase3\results\metrics\cityscapes_val_manifest.json `
  --target both `
  --preprocess official `
  --max-samples 5 `
  --output phase3\results\metrics\cityscapes_miou_p1a_stage2_stage3_smoke.json
```

正式 val：

```powershell
D:\software\anaconda3\envs\efficientvit\python.exe phase3\scripts\evaluate_cityscapes_miou.py `
  --manifest phase3\results\metrics\cityscapes_val_manifest.json `
  --target both `
  --preprocess official `
  --output phase3\results\metrics\cityscapes_miou_p1a_stage2_stage3.json
```

## 5. 通过条件

最终采用的 Plugin 主线至少需要满足：

1. baseline / plugin 均可在同一 manifest 上完整执行。
2. plugin mIoU 不低于 baseline，或下降在明确阈值内并有数值误差解释。
3. plugin vs baseline 的 valid-label argmax agreement 接近 100%；若不为 100%，必须列出 mismatch 总量。
4. 报告必须说明该 mIoU 是 `official` preprocess 还是 `deployment` regression check。

失败分支如 P1b/P1mix 不强制跑完整 val mIoU；只要它们不作为最终主线，已有 fixed-image correctness 与 latency/Nsight 证据足够支撑“不采纳”结论。

## 6. 当前阻塞

截至本设计落盘时，本机仓库未发现 `leftImg8bit` / `gtFine` 目录。已提供 manifest 与 mIoU 脚本；实际 mIoU 运行依赖用户完成 Cityscapes 官方数据下载与解压。
