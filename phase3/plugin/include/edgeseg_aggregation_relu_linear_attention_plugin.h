#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgeseg {

constexpr char const* kAggregationReluLinearAttentionPluginName = "EdgesegAggregationReluLinearAttention_TRT";
constexpr char const* kAggregationReluLinearAttentionPluginVersion = "1";
constexpr char const* kAggregationReluLinearAttentionPluginNamespace = "edgeseg";

struct AggregationReluLinearAttentionPluginConfig {
    int32_t dim{16};
    float eps{1.0e-15F};
    int32_t qkvC{192};
    int32_t outputC{128};
    int32_t height{64};
    int32_t width{128};

    int32_t attentionInputC() const noexcept {
        return qkvC * 2;
    }
};

int launchAggregationReluLinearAttention(
    float const* qkv,
    float const* depthwiseWeight,
    float const* pointwiseWeight,
    float* output,
    void* workspace,
    size_t workspaceBytes,
    AggregationReluLinearAttentionPluginConfig const& config,
    cudaStream_t stream) noexcept;

class EdgesegAggregationReluLinearAttentionPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit EdgesegAggregationReluLinearAttentionPlugin(AggregationReluLinearAttentionPluginConfig config);
    EdgesegAggregationReluLinearAttentionPlugin(void const* serialData, size_t serialLength);

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::PluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,
        int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs,
        int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index,
        nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    AggregationReluLinearAttentionPluginConfig config_{};
    std::string namespace_{kAggregationReluLinearAttentionPluginNamespace};
};

class EdgesegAggregationReluLinearAttentionPluginCreator final : public nvinfer1::IPluginCreator {
public:
    EdgesegAggregationReluLinearAttentionPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name,
        void const* serialData,
        size_t serialLength) noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string namespace_{kAggregationReluLinearAttentionPluginNamespace};
    std::vector<nvinfer1::PluginField> fields_{};
    nvinfer1::PluginFieldCollection fieldCollection_{};
};

} // namespace edgeseg

extern "C" __declspec(dllexport) int edgesegAggregationReluLinearAttentionPluginAbiVersion();
