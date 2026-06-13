#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgeseg {

constexpr char const* kReluLinearAttentionPluginName = "EdgesegReluLinearAttention_TRT";
constexpr char const* kReluLinearAttentionPluginVersion = "1";
constexpr char const* kReluLinearAttentionPluginNamespace = "edgeseg";

struct ReluLinearAttentionPluginConfig {
    int32_t dim{16};
    float eps{1.0e-15F};
    int32_t inputC{384};
    int32_t height{64};
    int32_t width{128};

    int32_t outputC() const noexcept {
        return inputC / 3;
    }
};

int launchReluLinearAttentionSkeleton(
    float const* input, float* output, int32_t outputElements, cudaStream_t stream) noexcept;

class EdgesegReluLinearAttentionPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit EdgesegReluLinearAttentionPlugin(ReluLinearAttentionPluginConfig config);
    EdgesegReluLinearAttentionPlugin(void const* serialData, size_t serialLength);

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
    ReluLinearAttentionPluginConfig config_{};
    std::string namespace_{kReluLinearAttentionPluginNamespace};
};

class EdgesegReluLinearAttentionPluginCreator final : public nvinfer1::IPluginCreator {
public:
    EdgesegReluLinearAttentionPluginCreator();

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
    std::string namespace_{kReluLinearAttentionPluginNamespace};
    std::vector<nvinfer1::PluginField> fields_{};
    nvinfer1::PluginFieldCollection fieldCollection_{};
};

} // namespace edgeseg

extern "C" __declspec(dllexport) int edgesegReluLinearAttentionPluginAbiVersion();
