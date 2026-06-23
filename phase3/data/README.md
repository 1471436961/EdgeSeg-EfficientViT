# Phase 3 Cityscapes 数据目录

Cityscapes 数据集受官方账号与授权约束，且体积较大，因此数据本体不入 git。本目录只保留占位文件和说明。

正式 mIoU 评估需要把官方 `leftImg8bit_trainvaltest.zip` 与 `gtFine_trainvaltest.zip` 解压到同一个根目录，推荐本地布局如下：

```text
phase3/data/cityscapes/
|-- leftImg8bit/
|   `-- val/
|       `-- <city>/
|           `-- *_leftImg8bit.png
`-- gtFine/
    `-- val/
        `-- <city>/
            `-- *_gtFine_labelIds.png
```

解压后先生成 manifest：

```powershell
D:\software\anaconda3\envs\efficientvit\python.exe phase3\scripts\prepare_cityscapes_eval_manifest.py `
  --cityscapes-root phase3\data\cityscapes `
  --split val `
  --label-kind labelIds `
  --output phase3\results\metrics\cityscapes_val_manifest.json
```

再运行 mIoU accuracy gate：

```powershell
D:\software\anaconda3\envs\efficientvit\python.exe phase3\scripts\evaluate_cityscapes_miou.py `
  --manifest phase3\results\metrics\cityscapes_val_manifest.json `
  --target both `
  --preprocess official `
  --output phase3\results\metrics\cityscapes_miou_p1a_stage2_stage3.json
```

`official` 预处理使用 ImageNet mean/std，对齐上游 Cityscapes 评估口径；`deployment` 预处理只用于和既有 Phase 2/3 benchmark 输入口径一致的回归检查，不能作为正式 mIoU 结论。
