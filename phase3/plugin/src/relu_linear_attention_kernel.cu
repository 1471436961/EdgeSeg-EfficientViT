#include "edgeseg_relu_linear_attention_plugin.h"

namespace {

__global__ void zeroFillKernel(float* output, int32_t n) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        output[idx] = 0.0F;
    }
}

} // namespace

namespace edgeseg {

int launchReluLinearAttentionSkeleton(
    float const* input, float* output, int32_t outputElements, cudaStream_t stream) noexcept {
    (void) input;
    if (output == nullptr || outputElements < 0) {
        return 1;
    }
    if (outputElements == 0) {
        return 0;
    }

    constexpr int32_t threads = 256;
    const int32_t blocks = (outputElements + threads - 1) / threads;
    zeroFillKernel<<<blocks, threads, 0, stream>>>(output, outputElements);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

} // namespace edgeseg
