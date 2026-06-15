#include "edgeseg_aggregation_relu_linear_attention_plugin.h"
#include "edgeseg_relu_linear_attention_plugin.h"

namespace {

constexpr int32_t kAggregationThreads = 256;
constexpr int32_t kSpecializedDim = 16;
constexpr int32_t kAggregationGroups = 12;
constexpr int32_t kChannelsPerAggregationGroup = 16;

__global__ void fusedAggregationCatKernel(
    float const* input,
    float const* depthwiseWeight,
    float const* pointwiseWeight,
    float* attentionInput,
    int32_t height,
    int32_t width) {
    const int32_t spatialSize = height * width;
    const int32_t n = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t group = static_cast<int32_t>(blockIdx.y);
    if (n >= spatialSize || group >= kAggregationGroups) {
        return;
    }

    const int32_t h = n / width;
    const int32_t w = n - h * width;

    float depthwise[kChannelsPerAggregationGroup];
#pragma unroll
    for (int32_t channelInGroup = 0; channelInGroup < kChannelsPerAggregationGroup; ++channelInGroup) {
        const int32_t channel = group * kChannelsPerAggregationGroup + channelInGroup;
        const int32_t channelBase = channel * spatialSize;
        attentionInput[channelBase + n] = input[channelBase + n];

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
                const int32_t inputIndex = channelBase + ih * width + iw;
                const int32_t weightIndex = channel * 25 + ky * 5 + kx;
                sum += input[inputIndex] * depthwiseWeight[weightIndex];
            }
        }
        depthwise[channelInGroup] = sum;
    }

#pragma unroll
    for (int32_t outputInGroup = 0; outputInGroup < kChannelsPerAggregationGroup; ++outputInGroup) {
        const int32_t outChannel = group * kChannelsPerAggregationGroup + outputInGroup;
        const int32_t weightBase = outChannel * kChannelsPerAggregationGroup;
        float sum = 0.0F;
#pragma unroll
        for (int32_t i = 0; i < kChannelsPerAggregationGroup; ++i) {
            sum += depthwise[i] * pointwiseWeight[weightBase + i];
        }
        attentionInput[(192 + outChannel) * spatialSize + n] = sum;
    }
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
    const size_t attentionInputElements = static_cast<size_t>(attentionInputC) * static_cast<size_t>(spatialSize);
    const size_t vkElements = static_cast<size_t>(attentionHeads) * static_cast<size_t>(config.dim + 1)
        * static_cast<size_t>(config.dim);
    const size_t attentionInputBytes = attentionInputElements * sizeof(float);
    const size_t vkBytes = vkElements * sizeof(float);
    if (workspaceBytes < attentionInputBytes + vkBytes) {
        return 1;
    }

    auto* cursor = static_cast<char*>(workspace);
    auto* attentionInput = reinterpret_cast<float*>(cursor);
    cursor += attentionInputBytes;
    auto* vkWorkspace = reinterpret_cast<float*>(cursor);

    const int32_t spatialBlocks = (spatialSize + kAggregationThreads - 1) / kAggregationThreads;
    const dim3 aggregationGrid(spatialBlocks, kAggregationGroups);
    fusedAggregationCatKernel<<<aggregationGrid, kAggregationThreads, 0, stream>>>(
        qkv, depthwiseWeight, pointwiseWeight, attentionInput, config.height, config.width);
    cudaError_t status = cudaGetLastError();
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
