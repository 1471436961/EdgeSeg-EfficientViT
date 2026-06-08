# TensorRT Engine Inspection Summary

- Engine: `E:\EdgeSeg-EfficientViT\EdgeSeg-EfficientViT\phase2\results\engines\efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine`
- ONNX: `E:\EdgeSeg-EfficientViT\EdgeSeg-EfficientViT\phase2\results\onnx\efficientvit_seg_b0_cityscapes_1024x2048.onnx`
- TensorRT: `8.6.1`
- EngineInspector detail: `layer_names_only`
- ONNX node count: 393
- TensorRT engine layer count: 155
- Overall layer-count reduction: 60.56%

This file is structural evidence. It does not contain runtime timing; use `trt_nsys_attribution_summary.md` for GPU kernel time.

## Group Mapping Summary

| Group | ONNX nodes | TRT layers | Layer reduction | TRT fused layers | PWN layers | Explicit `+` fusion layers |
|---|---:|---:|---:|---:|---:|---:|
| `stage3` | 175 | 50 | 71.43% | 17 | 12 | 5 |
| `stage2` | 157 | 47 | 70.06% | 16 | 12 | 4 |
| `head` | 18 | 21 | -16.67% | 6 | 3 | 3 |
| `stage1` | 11 | 16 | -45.45% | 6 | 4 | 2 |
| `stage0` | 11 | 12 | -9.09% | 5 | 4 | 1 |
| `stem` | 6 | 5 | 16.67% | 3 | 2 | 1 |
| `constant/unnamed` | 15 | 4 | 73.33% | 0 | 0 | 0 |

## ONNX Op Type Summary

| Op type | Count |
|---|---:|
| `Constant` | 157 |
| `Conv` | 57 |
| `HardSwish` | 25 |
| `Slice` | 24 |
| `Add` | 18 |
| `Reshape` | 16 |
| `Unsqueeze` | 15 |
| `Concat` | 14 |
| `Shape` | 9 |
| `Gather` | 9 |
| `Relu` | 8 |
| `Transpose` | 8 |
| `Cast` | 8 |
| `MatMul` | 8 |
| `ConstantOfShape` | 4 |
| `Pad` | 4 |
| `Div` | 4 |
| `Mul` | 3 |
| `Resize` | 2 |

## TensorRT Layer Kind Summary

| Layer kind | Count |
|---|---:|
| `conv` | 56 |
| `pointwise_fusion` | 37 |
| `single_or_other` | 29 |
| `explicit_fusion` | 16 |
| `matmul` | 10 |
| `constant` | 4 |
| `resize` | 3 |

## Stage2 Context Engine Layers

| Index | Kind | TensorRT layer name |
|---:|---|---|
| 44 | `conv` | `/backbone/stages.2/op_list.1/context_module/main/qkv/conv/Conv` |
| 45 | `conv` | `/backbone/stages.2/op_list.1/context_module/main/aggreg.0/aggreg.0.0/Conv` |
| 46 | `conv` | `/backbone/stages.2/op_list.1/context_module/main/aggreg.0/aggreg.0.1/Conv` |
| 47 | `single_or_other` | `/backbone/stages.2/op_list.1/context_module/main/Reshape` |
| 48 | `pointwise_fusion` | `PWN(/backbone/stages.2/op_list.1/context_module/main/kernel_func/Relu)` |
| 49 | `pointwise_fusion` | `PWN(/backbone/stages.2/op_list.1/context_module/main/kernel_func_1/Relu)` |
| 50 | `single_or_other` | `/backbone/stages.2/op_list.1/context_module/main/Pad` |
| 51 | `matmul` | `/backbone/stages.2/op_list.1/context_module/main/MatMul` |
| 52 | `matmul` | `/backbone/stages.2/op_list.1/context_module/main/MatMul_1` |
| 53 | `matmul` | `Reformatting CopyNode for Output Tensor 0 to /backbone/stages.2/op_list.1/context_module/main/MatMul_1` |
| 54 | `pointwise_fusion` | `PWN(/backbone/stages.2/op_list.1/context_module/main/Constant_30_output_0 + (Unnamed Layer* 80) [Shuffle] + /backbone/stages.2/op_list.1/context_module/main/Add, /backbone/stages.2/op_list.1/context_module/main/Div)` |
| 55 | `single_or_other` | `Reformatting CopyNode for Input Tensor 0 to /backbone/stages.2/op_list.1/context_module/main/Reshape_3` |
| 56 | `single_or_other` | `/backbone/stages.2/op_list.1/context_module/main/Reshape_3` |
| 57 | `single_or_other` | `/backbone/stages.2/op_list.1/context_module/main/Cast_1` |
| 58 | `explicit_fusion` | `/backbone/stages.2/op_list.1/context_module/main/proj/conv/Conv + /backbone/stages.2/op_list.1/context_module/Add` |
| 64 | `conv` | `/backbone/stages.2/op_list.2/context_module/main/qkv/conv/Conv` |
| 65 | `conv` | `/backbone/stages.2/op_list.2/context_module/main/aggreg.0/aggreg.0.0/Conv` |
| 66 | `conv` | `/backbone/stages.2/op_list.2/context_module/main/aggreg.0/aggreg.0.1/Conv` |
| 67 | `single_or_other` | `/backbone/stages.2/op_list.2/context_module/main/Reshape` |
| 68 | `pointwise_fusion` | `PWN(/backbone/stages.2/op_list.2/context_module/main/kernel_func/Relu)` |
| 69 | `pointwise_fusion` | `PWN(/backbone/stages.2/op_list.2/context_module/main/kernel_func_1/Relu)` |
| 70 | `single_or_other` | `/backbone/stages.2/op_list.2/context_module/main/Pad` |
| 71 | `matmul` | `/backbone/stages.2/op_list.2/context_module/main/MatMul` |
| 72 | `matmul` | `/backbone/stages.2/op_list.2/context_module/main/MatMul_1` |
| 73 | `matmul` | `Reformatting CopyNode for Output Tensor 0 to /backbone/stages.2/op_list.2/context_module/main/MatMul_1` |
| 74 | `pointwise_fusion` | `PWN(/backbone/stages.2/op_list.2/context_module/main/Constant_34_output_0 + (Unnamed Layer* 145) [Shuffle] + /backbone/stages.2/op_list.2/context_module/main/Add, /backbone/stages.2/op_list.2/context_module/main/Div)` |
| 75 | `single_or_other` | `Reformatting CopyNode for Input Tensor 0 to /backbone/stages.2/op_list.2/context_module/main/Reshape_3` |
| 76 | `single_or_other` | `/backbone/stages.2/op_list.2/context_module/main/Reshape_3` |
| 77 | `single_or_other` | `/backbone/stages.2/op_list.2/context_module/main/Cast_1` |
| 78 | `explicit_fusion` | `/backbone/stages.2/op_list.2/context_module/main/proj/conv/Conv + /backbone/stages.2/op_list.2/context_module/Add` |

## Interpretation Notes

- `PWN(...)` layers indicate TensorRT pointwise/activation fusion.
- Layer names containing ` + ` indicate TensorRT fused multiple ONNX-named operations into one engine layer.
- The current FP32 engine exposes layer names only, not detailed tactic metadata. For tactic-level evidence, rebuild with detailed profiling verbosity or capture verbose builder logs.
- EngineInspector structure is auxiliary evidence; Nsight SQLite attribution remains the source of runtime GPU time.
