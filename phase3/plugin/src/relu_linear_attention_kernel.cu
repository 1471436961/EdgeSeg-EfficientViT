#include "edgeseg_relu_linear_attention_plugin.h"

namespace {

constexpr int32_t kVkThreads = 128;
constexpr int32_t kOutputThreads = 256;
constexpr int32_t kMaxDim = 32;
constexpr int32_t kSpecializedDim = 16;
constexpr int32_t kSpecializedVkElements = (kSpecializedDim + 1) * kSpecializedDim;

__device__ __forceinline__ float relu(float value) {
    return value > 0.0F ? value : 0.0F;
}

__device__ __forceinline__ float warpReduceSum(float value) {
    for (int32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ float blockReduceSum(float value) {
    __shared__ float warpSums[32];
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31U);
    const int32_t warp = static_cast<int32_t>(threadIdx.x >> 5U);
    const int32_t warpCount = (static_cast<int32_t>(blockDim.x) + warpSize - 1) / warpSize;

    value = warpReduceSum(value);
    if (lane == 0) {
        warpSums[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < static_cast<uint32_t>(warpCount) ? warpSums[lane] : 0.0F;
    if (warp == 0) {
        value = warpReduceSum(value);
    }
    return value;
}

__global__ void computeVkKernel(
    float const* input, float* vkWorkspace, int32_t heads, int32_t dim, int32_t spatialSize) {
    (void) heads;
    const int32_t d = static_cast<int32_t>(blockIdx.x % dim);
    const int32_t row = static_cast<int32_t>((blockIdx.x / dim) % (dim + 1));
    const int32_t head = static_cast<int32_t>(blockIdx.x / (dim * (dim + 1)));

    float sum = 0.0F;
    const int32_t kChannel = head * 3 * dim + dim + d;
    for (int32_t n = static_cast<int32_t>(threadIdx.x); n < spatialSize; n += static_cast<int32_t>(blockDim.x)) {
        const float k = relu(input[kChannel * spatialSize + n]);
        const float v = row == dim ? 1.0F : input[(head * 3 * dim + 2 * dim + row) * spatialSize + n];
        sum += v * k;
    }

    const float total = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        vkWorkspace[(head * (dim + 1) + row) * dim + d] = total;
    }
}

__global__ void computeVkKernelDim16(float const* input, float* vkWorkspace, int32_t spatialSize) {
    const int32_t d = static_cast<int32_t>(blockIdx.x & (kSpecializedDim - 1));
    const int32_t row = static_cast<int32_t>((blockIdx.x / kSpecializedDim) % (kSpecializedDim + 1));
    const int32_t head = static_cast<int32_t>(blockIdx.x / kSpecializedVkElements);

    float sum = 0.0F;
    const int32_t kChannel = head * 3 * kSpecializedDim + kSpecializedDim + d;
    for (int32_t n = static_cast<int32_t>(threadIdx.x); n < spatialSize; n += static_cast<int32_t>(blockDim.x)) {
        const float k = relu(input[kChannel * spatialSize + n]);
        const float v = row == kSpecializedDim
            ? 1.0F
            : input[(head * 3 * kSpecializedDim + 2 * kSpecializedDim + row) * spatialSize + n];
        sum += v * k;
    }

    const float total = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        vkWorkspace[(head * (kSpecializedDim + 1) + row) * kSpecializedDim + d] = total;
    }
}

__global__ void computeOutputKernel(
    float const* input,
    float const* vkWorkspace,
    float* output,
    int32_t heads,
    int32_t dim,
    int32_t spatialSize,
    float eps) {
    const int32_t head = static_cast<int32_t>(blockIdx.x);
    const int32_t n = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= heads || n >= spatialSize) {
        return;
    }

    const int32_t qBase = head * 3 * dim * spatialSize;
    const int32_t vkBase = head * (dim + 1) * dim;
    const int32_t vkElements = (dim + 1) * dim;

    extern __shared__ float vkShared[];
    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < vkElements; idx += static_cast<int32_t>(blockDim.x)) {
        vkShared[idx] = vkWorkspace[vkBase + idx];
    }
    __syncthreads();

    float q[kMaxDim];
    float denominator = 0.0F;
    for (int32_t d = 0; d < dim; ++d) {
        q[d] = relu(input[qBase + d * spatialSize + n]);
        denominator += vkShared[dim * dim + d] * q[d];
    }
    denominator += eps;

    for (int32_t row = 0; row < dim; ++row) {
        float numerator = 0.0F;
        for (int32_t d = 0; d < dim; ++d) {
            numerator += vkShared[row * dim + d] * q[d];
        }
        output[(head * dim + row) * spatialSize + n] = numerator / denominator;
    }
}

__global__ void computeOutputKernelDim16(
    float const* input, float const* vkWorkspace, float* output, int32_t spatialSize, float eps) {
    const int32_t head = static_cast<int32_t>(blockIdx.x);
    const int32_t n = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (n >= spatialSize) {
        return;
    }

    const int32_t qBase = head * 3 * kSpecializedDim * spatialSize;
    const int32_t vkBase = head * kSpecializedVkElements;

    extern __shared__ float vkShared[];
    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < kSpecializedVkElements;
         idx += static_cast<int32_t>(blockDim.x)) {
        vkShared[idx] = vkWorkspace[vkBase + idx];
    }
    __syncthreads();

    float q[kSpecializedDim];
    float denominator = 0.0F;
#pragma unroll
    for (int32_t d = 0; d < kSpecializedDim; ++d) {
        q[d] = relu(input[qBase + d * spatialSize + n]);
        denominator += vkShared[kSpecializedDim * kSpecializedDim + d] * q[d];
    }
    denominator += eps;

#pragma unroll
    for (int32_t row = 0; row < kSpecializedDim; ++row) {
        float numerator = 0.0F;
#pragma unroll
        for (int32_t d = 0; d < kSpecializedDim; ++d) {
            numerator += vkShared[row * kSpecializedDim + d] * q[d];
        }
        output[(head * kSpecializedDim + row) * spatialSize + n] = numerator / denominator;
    }
}

} // namespace

namespace edgeseg {

int launchReluLinearAttention(
    float const* input,
    float* output,
    void* workspace,
    size_t workspaceBytes,
    ReluLinearAttentionPluginConfig const& config,
    cudaStream_t stream) noexcept {
    if (input == nullptr || output == nullptr || workspace == nullptr || config.dim <= 0 || config.dim > kMaxDim
        || config.inputC <= 0 || config.height <= 0 || config.width <= 0 || config.inputC % (3 * config.dim) != 0) {
        return 1;
    }

    const int32_t heads = config.inputC / (3 * config.dim);
    const int32_t spatialSize = config.height * config.width;
    const size_t expectedWorkspaceBytes =
        static_cast<size_t>(heads) * static_cast<size_t>(config.dim + 1) * static_cast<size_t>(config.dim) * sizeof(float);
    if (workspaceBytes < expectedWorkspaceBytes) {
        return 1;
    }

    auto* vkWorkspace = static_cast<float*>(workspace);
    if (config.dim == kSpecializedDim) {
        const int32_t vkBlocks = heads * kSpecializedVkElements;
        computeVkKernelDim16<<<vkBlocks, kVkThreads, 0, stream>>>(input, vkWorkspace, spatialSize);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            return 1;
        }

        const dim3 outputGrid(
            static_cast<uint32_t>(heads), static_cast<uint32_t>((spatialSize + kOutputThreads - 1) / kOutputThreads), 1U);
        computeOutputKernelDim16<<<outputGrid, kOutputThreads, kSpecializedVkElements * sizeof(float), stream>>>(
            input, vkWorkspace, output, spatialSize, config.eps);
        status = cudaGetLastError();
        return status == cudaSuccess ? 0 : 1;
    }

    const int32_t vkBlocks = heads * (config.dim + 1) * config.dim;
    computeVkKernel<<<vkBlocks, kVkThreads, 0, stream>>>(input, vkWorkspace, heads, config.dim, spatialSize);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return 1;
    }

    const dim3 outputGrid(
        static_cast<uint32_t>(heads), static_cast<uint32_t>((spatialSize + kOutputThreads - 1) / kOutputThreads), 1U);
    const size_t outputSharedBytes =
        static_cast<size_t>(config.dim + 1) * static_cast<size_t>(config.dim) * sizeof(float);
    computeOutputKernel<<<outputGrid, kOutputThreads, outputSharedBytes, stream>>>(
        input, vkWorkspace, output, heads, config.dim, spatialSize, config.eps);
    status = cudaGetLastError();
    return status == cudaSuccess ? 0 : 1;
}

} // namespace edgeseg
