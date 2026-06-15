#include "edgeseg_aggregation_relu_linear_attention_plugin.h"
#include "edgeseg_relu_linear_attention_plugin.h"

namespace {

constexpr int32_t kAggregationThreads = 512;
constexpr int32_t kSpecializedDim = 16;
constexpr int32_t kAggregationGroups = 12;
constexpr int32_t kChannelsPerAggregationGroup = 16;
constexpr int32_t kDepthwiseKernel = 5;
constexpr int32_t kDepthwiseKernelElements = kDepthwiseKernel * kDepthwiseKernel;
constexpr int32_t kDepthwiseTileChannels = 8;
constexpr int32_t kDepthwiseTileRows = 8;
constexpr int32_t kDepthwiseTileWidth = 132;
constexpr int32_t kSpecializedStage2Width = 128;

__global__ void fusedAggregationCatKernel(
    float const* input,
    float const* depthwiseWeight,
    float const* pointwiseWeight,
    float* attentionInput,
    int32_t height,
    int32_t width) {
    __shared__ float pointwiseTile[kChannelsPerAggregationGroup * kChannelsPerAggregationGroup];
    __shared__ float depthwiseInputTile[kDepthwiseTileChannels * kDepthwiseTileRows * kDepthwiseTileWidth];
    __shared__ float depthwiseWeightTile[kDepthwiseTileChannels * kDepthwiseKernelElements];

    const int32_t group = static_cast<int32_t>(blockIdx.y);
    if (group >= kAggregationGroups) {
        return;
    }

#pragma unroll
    for (int32_t idx = static_cast<int32_t>(threadIdx.x);
         idx < kChannelsPerAggregationGroup * kChannelsPerAggregationGroup;
         idx += static_cast<int32_t>(blockDim.x)) {
        const int32_t outputInGroup = idx / kChannelsPerAggregationGroup;
        const int32_t inputInGroup = idx - outputInGroup * kChannelsPerAggregationGroup;
        const int32_t outChannel = group * kChannelsPerAggregationGroup + outputInGroup;
        pointwiseTile[idx] = pointwiseWeight[outChannel * kChannelsPerAggregationGroup + inputInGroup];
    }
    __syncthreads();

    const int32_t spatialSize = height * width;
    const int32_t n = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool useRowTile = (width == kSpecializedStage2Width && (spatialSize % kAggregationThreads) == 0);
    if (n >= spatialSize) {
        return;
    }

    const int32_t h = n / width;
    const int32_t w = n - h * width;

    float depthwise[kChannelsPerAggregationGroup];

    if (useRowTile) {
        const int32_t tileBaseH = static_cast<int32_t>(blockIdx.x * blockDim.x) / width;
        const int32_t localH = static_cast<int32_t>(threadIdx.x) / width;
        const int32_t localW = static_cast<int32_t>(threadIdx.x) - localH * width;
        constexpr int32_t tilePlaneElements = kDepthwiseTileRows * kDepthwiseTileWidth;

#pragma unroll
        for (int32_t channelChunk = 0; channelChunk < kChannelsPerAggregationGroup;
             channelChunk += kDepthwiseTileChannels) {
            for (int32_t idx = static_cast<int32_t>(threadIdx.x);
                 idx < kDepthwiseTileChannels * tilePlaneElements;
                 idx += static_cast<int32_t>(blockDim.x)) {
                const int32_t tileChannel = idx / tilePlaneElements;
                const int32_t rem = idx - tileChannel * tilePlaneElements;
                const int32_t tileY = rem / kDepthwiseTileWidth;
                const int32_t tileX = rem - tileY * kDepthwiseTileWidth;
                const int32_t globalH = tileBaseH + tileY - 2;
                const int32_t globalW = tileX - 2;
                const int32_t channel = group * kChannelsPerAggregationGroup + channelChunk + tileChannel;
                float value = 0.0F;
                if (globalH >= 0 && globalH < height && globalW >= 0 && globalW < width) {
                    value = input[channel * spatialSize + globalH * width + globalW];
                }
                depthwiseInputTile[idx] = value;
            }

            for (int32_t idx = static_cast<int32_t>(threadIdx.x);
                 idx < kDepthwiseTileChannels * kDepthwiseKernelElements;
                 idx += static_cast<int32_t>(blockDim.x)) {
                const int32_t tileChannel = idx / kDepthwiseKernelElements;
                const int32_t kernelIndex = idx - tileChannel * kDepthwiseKernelElements;
                const int32_t channel = group * kChannelsPerAggregationGroup + channelChunk + tileChannel;
                depthwiseWeightTile[idx] = depthwiseWeight[channel * kDepthwiseKernelElements + kernelIndex];
            }

            __syncthreads();

#pragma unroll
            for (int32_t tileChannel = 0; tileChannel < kDepthwiseTileChannels; ++tileChannel) {
                const int32_t channelInGroup = channelChunk + tileChannel;
                const int32_t channel = group * kChannelsPerAggregationGroup + channelInGroup;
                const int32_t channelBase = channel * spatialSize;
                attentionInput[channelBase + n] = input[channelBase + n];

                float sum = 0.0F;
#pragma unroll
                for (int32_t ky = 0; ky < kDepthwiseKernel; ++ky) {
#pragma unroll
                    for (int32_t kx = 0; kx < kDepthwiseKernel; ++kx) {
                        const int32_t inputIndex =
                            tileChannel * tilePlaneElements + (localH + ky) * kDepthwiseTileWidth + localW + kx;
                        const int32_t weightIndex = tileChannel * kDepthwiseKernelElements + ky * kDepthwiseKernel + kx;
                        sum += depthwiseInputTile[inputIndex] * depthwiseWeightTile[weightIndex];
                    }
                }
                depthwise[channelInGroup] = sum;
            }

            __syncthreads();
        }
    } else {
#pragma unroll
        for (int32_t channelInGroup = 0; channelInGroup < kChannelsPerAggregationGroup; ++channelInGroup) {
            const int32_t channel = group * kChannelsPerAggregationGroup + channelInGroup;
            const int32_t channelBase = channel * spatialSize;
            attentionInput[channelBase + n] = input[channelBase + n];

            float sum = 0.0F;
#pragma unroll
            for (int32_t ky = 0; ky < kDepthwiseKernel; ++ky) {
                const int32_t ih = h + ky - 2;
                if (ih < 0 || ih >= height) {
                    continue;
                }
#pragma unroll
                for (int32_t kx = 0; kx < kDepthwiseKernel; ++kx) {
                    const int32_t iw = w + kx - 2;
                    if (iw < 0 || iw >= width) {
                        continue;
                    }
                    const int32_t inputIndex = channelBase + ih * width + iw;
                    const int32_t weightIndex = channel * kDepthwiseKernelElements + ky * kDepthwiseKernel + kx;
                    sum += input[inputIndex] * depthwiseWeight[weightIndex];
                }
            }
            depthwise[channelInGroup] = sum;
        }
    }

#pragma unroll
    for (int32_t outputInGroup = 0; outputInGroup < kChannelsPerAggregationGroup; ++outputInGroup) {
        const int32_t outChannel = group * kChannelsPerAggregationGroup + outputInGroup;
        float sum = 0.0F;
#pragma unroll
        for (int32_t i = 0; i < kChannelsPerAggregationGroup; ++i) {
            sum += depthwise[i] * pointwiseTile[outputInGroup * kChannelsPerAggregationGroup + i];
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
