# Phase 3 Cityscapes mIoU Summary

- Status: `ok`
- Target: `both`
- Samples: `500`
- Preprocess: `official`

| Engine | mIoU |
|---|---:|
| baseline | 75.646% |
| plugin | 75.646% |

## Baseline vs Plugin

- mIoU delta (plugin - baseline): `0.000012` percentage points
- Argmax agreement on valid labels: `0.99999992`
- Argmax mismatch pixels: `75 / 917018489`

## Notes

- This is an accuracy gate, not an execute-only latency benchmark.
- `official` preprocess uses ImageNet mean/std to match upstream Cityscapes eval.
- `deployment` preprocess matches the earlier Phase 2/3 benchmark input convention and is only a regression check.
