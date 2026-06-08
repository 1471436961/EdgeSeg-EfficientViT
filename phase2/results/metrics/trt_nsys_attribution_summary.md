# TensorRT Nsight Attribution Summary

- SQLite: `phase2/results/nsight/trt_fp32_fullres.sqlite`
- Metrics: `phase2/results/metrics/trt_benchmark_b0_cityscapes_1024x2048_fp32_nsys.json`
- Precision: `fp32`
- Warmup / measure: 20 / 100
- CUDA Events latency mean / p50: 55.242 ms / 55.237 ms
- `trt/execute` kernel avg: 54.454 ms / iter
- `trt/execute` launches: 185.0 / iter
- Layer-attributed kernel avg: 54.454 ms / iter
- Layer attribution / execute kernel time: 100.00%

Attribution method: TensorRT/NVTX layer range -> CUDA runtime launch inside range -> CUDA kernel with same `correlationId`.
NVTX range duration itself is not used as GPU component time.

## Group Summary

| Group | Avg kernel ms / iter | Share of execute kernel | Share of latency mean | Launches / iter | Layer count |
|---|---:|---:|---:|---:|---:|
| `stage0` | 14.669 | 26.94% | 26.55% | 10.0 | 12 |
| `stage2` | 12.179 | 22.37% | 22.05% | 57.0 | 47 |
| `stage3` | 7.511 | 13.79% | 13.60% | 88.0 | 50 |
| `stage1` | 7.431 | 13.65% | 13.45% | 10.0 | 16 |
| `head` | 6.608 | 12.14% | 11.96% | 15.0 | 21 |
| `stem` | 6.056 | 11.12% | 10.96% | 5.0 | 5 |
| `constant/unnamed` | 0.000 | 0.00% | 0.00% | 0.0 | 4 |

## Top 25 TensorRT Layer Ranges

| Rank | Group | Layer / NVTX range | Avg kernel ms / iter | Share of execute kernel | Launches / iter |
|---:|---|---|---:|---:|---:|
| 1 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.0/main/inverted_conv/act/HardSwish_2)` | 2.718 | 4.99% | 1.0 |
| 2 | `stage0` | `/backbone/stages.0/op_list.0/main/depth_conv/conv/Conv` | 2.301 | 4.22% | 1.0 |
| 3 | `stem` | `/backbone/input_stem/op_list.0/conv/Conv` | 2.205 | 4.05% | 1.0 |
| 4 | `stage0` | `/backbone/stages.0/op_list.0/main/inverted_conv/conv/Conv` | 1.859 | 3.41% | 1.0 |
| 5 | `stage0` | `/backbone/stages.0/op_list.1/main/depth_conv/conv/Conv` | 1.752 | 3.22% | 1.0 |
| 6 | `stem` | `/backbone/input_stem/op_list.1/main/point_conv/conv/Conv + /backbone/input_stem/op_list.1/Add` | 1.556 | 2.86% | 1.0 |
| 7 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/inverted_conv/act/HardSwish_4)` | 1.360 | 2.50% | 1.0 |
| 8 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.0/main/inverted_conv/act/HardSwish_6)` | 1.360 | 2.50% | 1.0 |
| 9 | `stage0` | `PWN(PWN(/backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.0/op_list.1/main/depth_conv/act/HardSwish_5)` | 1.358 | 2.49% | 1.0 |
| 10 | `stage1` | `/backbone/stages.1/op_list.0/main/depth_conv/conv/Conv` | 1.141 | 2.09% | 1.0 |
| 11 | `stage0` | `/backbone/stages.0/op_list.1/main/point_conv/conv/Conv + /backbone/stages.0/op_list.1/Add` | 1.089 | 2.00% | 1.0 |
| 12 | `stage0` | `/backbone/stages.0/op_list.1/main/inverted_conv/conv/Conv` | 0.969 | 1.78% | 1.0 |
| 13 | `stage1` | `/backbone/stages.1/op_list.0/main/inverted_conv/conv/Conv` | 0.963 | 1.77% | 1.0 |
| 14 | `stem` | `/backbone/input_stem/op_list.1/main/depth_conv/conv/Conv` | 0.935 | 1.72% | 1.0 |
| 15 | `stage1` | `/backbone/stages.1/op_list.1/main/depth_conv/conv/Conv` | 0.909 | 1.67% | 1.0 |
| 16 | `head` | `/head/middle/op_list.0/main/depth_conv/conv/Conv` | 0.907 | 1.67% | 1.0 |
| 17 | `stage2` | `/backbone/stages.2/op_list.2/context_module/main/MatMul` | 0.723 | 1.33% | 1.0 |
| 18 | `stage2` | `/backbone/stages.2/op_list.1/context_module/main/MatMul` | 0.723 | 1.33% | 1.0 |
| 19 | `head` | `PWN(PWN(/head/middle/op_list.0/main/inverted_conv/act/HardSwish), /head/middle/op_list.0/main/inverted_conv/act/HardSwish_188)` | 0.681 | 1.25% | 1.0 |
| 20 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/inverted_conv/act/HardSwish_8)` | 0.680 | 1.25% | 1.0 |
| 21 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.0/act/HardSwish), /backbone/input_stem/op_list.0/act/HardSwish_0)` | 0.680 | 1.25% | 1.0 |
| 22 | `stage2` | `PWN(PWN(/backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish), /backbone/stages.2/op_list.0/main/inverted_conv/act/HardSwish_10)` | 0.680 | 1.25% | 1.0 |
| 23 | `head` | `PWN(PWN(/head/output_ops.0/op_list.0/act/HardSwish), /head/output_ops.0/op_list.0/act/HardSwish_190)` | 0.680 | 1.25% | 1.0 |
| 24 | `stem` | `PWN(PWN(/backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish), /backbone/input_stem/op_list.1/main/depth_conv/act/HardSwish_1)` | 0.680 | 1.25% | 1.0 |
| 25 | `stage1` | `PWN(PWN(/backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish), /backbone/stages.1/op_list.1/main/depth_conv/act/HardSwish_9)` | 0.679 | 1.25% | 1.0 |

## Top 25 CUDA Kernel Names

| Rank | Kernel | Avg ms / iter | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `generatedNativePointwise` | 17.467 | 32.08% | 3700 |
| 2 | `trt_maxwell_scudnn_128x32_relu_interior_nn_v1` | 7.501 | 13.77% | 1000 |
| 3 | `sm50_xmma_fprop_conv2d_c1_k1_f32f32_f32_f32_nchwkcrs_nchw_tilesize10x40x8_threadsize4x4_r3s3_u1v1_xcorr_zerobeta_execute_kernel_trt` | 5.448 | 10.01% | 600 |
| 4 | `trt_maxwell_scudnn_128x64_relu_interior_nn_v1` | 4.371 | 8.03% | 1100 |
| 5 | `sm50_xmma_fprop_conv2d_c1_k1_f32f32_f32_f32_nchwkcrs_nchw_tilesize10x10x8_threadsize2x2_r3s3_u2v2_xcorr_zerobeta_execute_kernel_trt` | 4.039 | 7.42% | 300 |
| 6 | `trt_maxwell_scudnn_128x32_relu_interior_nn_v0` | 2.727 | 5.01% | 7500 |
| 7 | `trt_maxwell_scudnn_128x32_relu_medium_nn_v1` | 2.205 | 4.05% | 100 |
| 8 | `sm50_xmma_fprop_conv2d_c1_k1_f32f32_f32_f32_nchwkcrs_nchw_tilesize10x40x8_threadsize4x4_r5s5_u1v1_xcorr_zerobeta_execute_kernel_trt` | 1.452 | 2.67% | 400 |
| 9 | `trt_maxwell_sgemm_128x32_tn_v1` | 1.446 | 2.66% | 200 |
| 10 | `trt_maxwell_sgemm_64x64_relu_nn_v1` | 1.245 | 2.29% | 500 |
| 11 | `void cuSliceLayer::naiveSlice<float, (cuSliceLayer::Mode)2, (int)4>(cuSliceLayer::LaunchParams<T1>)` | 1.045 | 1.92% | 400 |
| 12 | `ResizeBicubicKernel` | 0.931 | 1.71% | 200 |
| 13 | `trt_maxwell_sgemm_128x32_nn_v1` | 0.762 | 1.40% | 400 |
| 14 | `sm50_xmma_convolution_trt_depthwiseFP32NHWC4_fp32nhwcx4_fp32nhwcx4_fp32kcrs_execute_kernel_trt` | 0.723 | 1.33% | 300 |
| 15 | `trt_maxwell_sgemm_128x32_relu_tn_v1` | 0.677 | 1.24% | 200 |
| 16 | `trt_maxwell_sgemm_128x64_relu_nn_v1` | 0.536 | 0.98% | 200 |
| 17 | `trt_maxwell_scudnn_128x32_relu_small_nn_v1` | 0.449 | 0.82% | 100 |
| 18 | `trt_maxwell_sgemm_128x64_nn_v1` | 0.422 | 0.77% | 200 |
| 19 | `void cuInt8::nhwcToNchwKernel<float>(int, int, int, int, int, int, int, int, const T1 *, T1 *)` | 0.264 | 0.48% | 200 |
| 20 | `void cuEltwise::eltwise<cuEltwise::SimpleAlgo<float, float>, cuEltwise::Compute<(nvinfer1::ElementWiseOperation)0>>(cuEltwise::LaunchParams)` | 0.255 | 0.47% | 100 |
| 21 | `__myl_bb0_1_Mov` | 0.176 | 0.32% | 200 |
| 22 | `trt_maxwell_sgemm_128x128_relu_tn_v1` | 0.162 | 0.30% | 100 |
| 23 | `void cuInt8::nchwToNhwcKernel<float>(int, int, int, int, int, int, int, int, bool, const T1 *, T1 *)` | 0.088 | 0.16% | 100 |
| 24 | `trt_maxwell_sgemm_32x128_relu_tn_v1` | 0.040 | 0.07% | 100 |
| 25 | `void genericReformat::copyPackedKernel<float, float, (bool)0, (bool)1, genericReformat::IdentityCoordMapper<(int)4>, (int)4>(unsigned int, unsigned int, const void *, genericReformat::ArrayN<T6>, genericReformat::ArrayNWithReducedDivisors<T6>, genericReformat::ArrayN<T6>, int, int, int, const float *, void *, genericReformat::ArrayN<T6>, genericReformat::ArrayNWithReducedDivisors<T6>, genericReformat::ArrayNWithReducedDivisors<T6>, genericReformat::ArrayN<T6>, int, int, int, const float *, T5)` | 0.022 | 0.04% | 100 |

## Memory Activity

| Type | Count | Avg ms / iter | Bytes |
|---|---:|---:|---:|
| Memcpy | 0 | 0.000 | 0 |
| Memset | 0 | 0.000 | 0 |

## Interpretation Notes

- This summary uses TensorRT-emitted layer NVTX ranges, not PyTorch module hooks.
- Group names are inferred from ONNX-like layer paths such as `/backbone/stages.2/...`.
- This can answer residual hotspot trends after TensorRT, but it is not a one-to-one replay of Phase 1 Plan B/C/D ranges.
