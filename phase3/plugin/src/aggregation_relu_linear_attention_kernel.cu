#include "edgeseg_aggregation_relu_linear_attention_plugin.h"
#include "edgeseg_relu_linear_attention_plugin.h"

namespace {

constexpr int32_t kAggregationThreads = 256;
constexpr int32_t kSpecializedDim = 16;

__global__ void depthwise5x5Kernel(
    float const* input,
    float const* weight,
    float* output,
    int32_t channels,
    int32_t height,
    int32_t width) {
    const int32_t spatialSize = height * width;
    const int32_t index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t total = channels * spatialSize;
    if (index >= total) {
        return;
    }

    const int32_t n = index % spatialSize;
    const int32_t channel = index / spatialSize;
    const int32_t h = n / width;
    const int32_t w = n - h * width;

    float sum = 0.0F;
#pragma unroll
    for (int32_t ky = 0; ky < 5; ++ky) {
        const int32_t ih = h + ky - 2;
        if (ih < 0 || ih >= height) {
            continue;
        }
#pragma unroll
        for (int32_t kx = 0; kx < 5; ++kx) {
            const int32_t iw = w + kx - 2;
            if (iw < 0 || iw >= width) {
                continue;
            }
            const int32_t inputIndex = channel * spatialSize + ih * width + iw;
            const int32_t weightIndex = channel * 25 + ky * 5 + kx;
            sum += input[inputIndex] * weight[weightIndex];
        }
    }
    output[index] = sum;
}

__global__ void groupedPointwise1x1Kernel(
    float const* input,
    float const* weight,
    float* output,
    int32_t channels,
    int32_t spatialSize,
    int32_t groups,
    int32_t channelsPerGroup) {
    const int32_t index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t total = channels * spatialSize;
    if (index >= total) {
        return;
    }

    const int32_t n = index % spatialSize;
    const int32_t outChannel = index / spatialSize;
    const int32_t group = outChannel / channelsPerGroup;
    if (group >= groups) {
        return;
    }

    const int32_t inputChannelBase = group * channelsPerGroup;
    const int32_t weightBase = outChannel * channelsPerGroup;
    float sum = 0.0F;
#pragma unroll
    for (int32_t i = 0; i < 16; ++i) {
        sum += input[(inputChannelBase + i) * spatialSize + n] * weight[weightBase + i];
    }
    output[index] = sum;
}

} // namespace

namespace edgeseg {

int launchAggregationReluLinearAttention(
    float const* qkv,
    float const* depthwiseWeight,
    float const* pointwiseWeight,
    float* output,
    void* workspace,
    size_t workspaceBytes,
    AggregationReluLinearAttentionPluginConfig const& config,
    cudaStream_t stream) noexcept {
    if (qkv == nullptr || depthwiseWeight == nullptr || pointwiseWeight == nullptr || output == nullptr
        || workspace == nullptr || config.dim != kSpecializedDim || config.qkvC != 192 || config.outputC != 128
        || config.height <= 0 || config.width <= 0 || config.attentionInputC() != 384) {
        return 1;
    }

    const int32_t spatialSize = config.height * config.width;
    const int32_t attentionInputC = config.attentionInputC();
    const int32_t attentionHeads = attentionInputC / (3 * config.dim);
    const size_t depthwiseElements = static_cast<size_t>(config.qkvC) * static_cast<size_t>(spatialSize);
    const size_t attentionInputElements = static_cast<size_t>(attentionInputC) * static_cast<size_t>(spatialSize);
    const size_t vkElements = static_cast<size_t>(attentionHeads) * static_cast<size_t>(config.dim + 1)
        * static_cast<size_t>(config.dim);
    const size_t depthwiseBytes = depthwiseElements * sizeof(float);
    const size_t attentionInputBytes = attentionInputElements * sizeof(float);
    const size_t vkBytes = vkElements * sizeof(float);
    if (workspaceBytes < depthwiseBytes + attentionInputBytes + vkBytes) {
        return 1;
    }

    auto* cursor = static_cast<char*>(workspace);
    auto* depthwiseWorkspace = reinterpret_cast<float*>(cursor);
    cursor += depthwiseBytes;
    auto* attentionInput = reinterpret_cast<float*>(cursor);
    cursor += attentionInputBytes;
    auto* vkWorkspace = reinterpret_cast<float*>(cursor);

    const size_t qkvBytes = static_cast<size_t>(config.qkvC) * static_cast<size_t>(spatialSize) * sizeof(float);
    cudaError_t status = cudaMemcpyAsync(attentionInput, qkv, qkvBytes, cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        return 1;
    }

    const int32_t aggregationElements = config.qkvC * spatialSize;
    const int32_t aggregationBlocks = (aggregationElements + kAggregationThreads - 1) / kAggregationThreads;
    depthwise5x5Kernel<<<aggregationBlocks, kAggregationThreads, 0, stream>>>(
        qkv, depthwiseWeight, depthwiseWorkspace, config.qkvC, config.height, config.width);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return 1;
    }

    float* aggregatedQkv = attentionInput + static_cast<size_t>(config.qkvC) * static_cast<size_t>(spatialSize);
    groupedPointwise1x1Kernel<<<aggregationBlocks, kAggregationThreads, 0, stream>>>(
        depthwiseWorkspace,
        pointwiseWeight,
        aggregatedQkv,
        config.qkvC,
        spatialSize,
        12,
        16);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return 1;
    }

    ReluLinearAttentionPluginConfig attentionConfig{};
    attentionConfig.dim = config.dim;
    attentionConfig.eps = config.eps;
    attentionConfig.inputC = attentionInputC;
    attentionConfig.height = config.height;
    attentionConfig.width = config.width;
    return launchReluLinearAttention(
        attentionInput,
        output,
        vkWorkspace,
        vkBytes,
        attentionConfig,
        stream);
}

} // namespace edgeseg
