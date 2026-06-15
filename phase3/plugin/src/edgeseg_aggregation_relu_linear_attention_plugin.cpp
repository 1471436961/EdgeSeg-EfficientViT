#include "edgeseg_aggregation_relu_linear_attention_plugin.h"

#include <NvInferRuntime.h>

#include <cstring>

namespace edgeseg {
namespace {

template <typename T>
void writeValue(char*& buffer, T value) noexcept {
    std::memcpy(buffer, &value, sizeof(T));
    buffer += sizeof(T);
}

template <typename T>
T readValue(char const*& buffer) noexcept {
    T value{};
    std::memcpy(&value, buffer, sizeof(T));
    buffer += sizeof(T);
    return value;
}

bool dimsEqual(nvinfer1::Dims const& dims, int32_t n, int32_t c, int32_t h, int32_t w) noexcept {
    return dims.nbDims == 4 && dims.d[0] == n && dims.d[1] == c && dims.d[2] == h && dims.d[3] == w;
}

AggregationReluLinearAttentionPluginConfig parseFields(nvinfer1::PluginFieldCollection const* fc) noexcept {
    AggregationReluLinearAttentionPluginConfig config{};
    if (fc == nullptr) {
        return config;
    }

    for (int32_t i = 0; i < fc->nbFields; ++i) {
        nvinfer1::PluginField const& field = fc->fields[i];
        if (field.name == nullptr || field.data == nullptr) {
            continue;
        }
        if (std::strcmp(field.name, "dim") == 0 && field.type == nvinfer1::PluginFieldType::kINT32) {
            config.dim = *static_cast<int32_t const*>(field.data);
        } else if (std::strcmp(field.name, "eps") == 0 && field.type == nvinfer1::PluginFieldType::kFLOAT32) {
            config.eps = *static_cast<float const*>(field.data);
        } else if (std::strcmp(field.name, "qkv_c") == 0 && field.type == nvinfer1::PluginFieldType::kINT32) {
            config.qkvC = *static_cast<int32_t const*>(field.data);
        } else if (std::strcmp(field.name, "output_c") == 0 && field.type == nvinfer1::PluginFieldType::kINT32) {
            config.outputC = *static_cast<int32_t const*>(field.data);
        } else if (std::strcmp(field.name, "height") == 0 && field.type == nvinfer1::PluginFieldType::kINT32) {
            config.height = *static_cast<int32_t const*>(field.data);
        } else if (std::strcmp(field.name, "width") == 0 && field.type == nvinfer1::PluginFieldType::kINT32) {
            config.width = *static_cast<int32_t const*>(field.data);
        }
    }
    return config;
}

bool validConfig(AggregationReluLinearAttentionPluginConfig const& config) noexcept {
    return config.dim == 16 && config.qkvC == 192 && config.outputC == 128 && config.height == 64
        && config.width == 128 && config.attentionInputC() == 384;
}

} // namespace

EdgesegAggregationReluLinearAttentionPlugin::EdgesegAggregationReluLinearAttentionPlugin(
    AggregationReluLinearAttentionPluginConfig config)
    : config_(config) {}

EdgesegAggregationReluLinearAttentionPlugin::EdgesegAggregationReluLinearAttentionPlugin(
    void const* serialData,
    size_t serialLength) {
    if (serialData == nullptr || serialLength != getSerializationSize()) {
        return;
    }
    auto const* cursor = static_cast<char const*>(serialData);
    config_.dim = readValue<int32_t>(cursor);
    config_.eps = readValue<float>(cursor);
    config_.qkvC = readValue<int32_t>(cursor);
    config_.outputC = readValue<int32_t>(cursor);
    config_.height = readValue<int32_t>(cursor);
    config_.width = readValue<int32_t>(cursor);
}

nvinfer1::IPluginV2DynamicExt* EdgesegAggregationReluLinearAttentionPlugin::clone() const noexcept {
    auto* plugin = new EdgesegAggregationReluLinearAttentionPlugin(config_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

int32_t EdgesegAggregationReluLinearAttentionPlugin::getNbOutputs() const noexcept {
    return 1;
}

nvinfer1::DimsExprs EdgesegAggregationReluLinearAttentionPlugin::getOutputDimensions(
    int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    if (outputIndex != 0 || inputs == nullptr || nbInputs != 3) {
        return nvinfer1::DimsExprs{};
    }

    nvinfer1::DimsExprs output = inputs[0];
    if (output.nbDims == 4) {
        output.d[1] = exprBuilder.constant(config_.outputC);
    }
    return output;
}

bool EdgesegAggregationReluLinearAttentionPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (inOut == nullptr || nbInputs != 3 || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }
    return inOut[pos].type == nvinfer1::DataType::kFLOAT
        && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void EdgesegAggregationReluLinearAttentionPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    (void) out;
    (void) nbOutputs;
    if (in == nullptr || nbInputs != 3 || !validConfig(config_)) {
        return;
    }
    (void) dimsEqual(in[0].desc.dims, 1, config_.qkvC, config_.height, config_.width);
    (void) dimsEqual(in[1].desc.dims, config_.qkvC, 1, 5, 5);
    (void) dimsEqual(in[2].desc.dims, config_.qkvC, 16, 1, 1);
}

size_t EdgesegAggregationReluLinearAttentionPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
    (void) inputs;
    (void) nbInputs;
    (void) outputs;
    (void) nbOutputs;
    if (!validConfig(config_)) {
        return 0;
    }
    const int32_t spatialSize = config_.height * config_.width;
    const int32_t attentionInputC = config_.attentionInputC();
    const int32_t attentionHeads = attentionInputC / (3 * config_.dim);
    const size_t depthwiseBytes =
        static_cast<size_t>(config_.qkvC) * static_cast<size_t>(spatialSize) * sizeof(float);
    const size_t attentionInputBytes =
        static_cast<size_t>(attentionInputC) * static_cast<size_t>(spatialSize) * sizeof(float);
    const size_t vkBytes = static_cast<size_t>(attentionHeads) * static_cast<size_t>(config_.dim + 1)
        * static_cast<size_t>(config_.dim) * sizeof(float);
    return depthwiseBytes + attentionInputBytes + vkBytes;
}

int32_t EdgesegAggregationReluLinearAttentionPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr || outputs[0] == nullptr
        || inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr || workspace == nullptr
        || !validConfig(config_)) {
        return 1;
    }
    if (!dimsEqual(inputDesc[0].dims, 1, config_.qkvC, config_.height, config_.width)
        || !dimsEqual(inputDesc[1].dims, config_.qkvC, 1, 5, 5)
        || !dimsEqual(inputDesc[2].dims, config_.qkvC, 16, 1, 1)
        || !dimsEqual(outputDesc[0].dims, 1, config_.outputC, config_.height, config_.width)) {
        return 1;
    }

    const size_t workspaceBytes = getWorkspaceSize(inputDesc, 3, outputDesc, 1);
    return launchAggregationReluLinearAttention(
        static_cast<float const*>(inputs[0]),
        static_cast<float const*>(inputs[1]),
        static_cast<float const*>(inputs[2]),
        static_cast<float*>(outputs[0]),
        workspace,
        workspaceBytes,
        config_,
        stream);
}

nvinfer1::DataType EdgesegAggregationReluLinearAttentionPlugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
    (void) inputTypes;
    return index == 0 && nbInputs == 3 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kFLOAT;
}

char const* EdgesegAggregationReluLinearAttentionPlugin::getPluginType() const noexcept {
    return kAggregationReluLinearAttentionPluginName;
}

char const* EdgesegAggregationReluLinearAttentionPlugin::getPluginVersion() const noexcept {
    return kAggregationReluLinearAttentionPluginVersion;
}

int32_t EdgesegAggregationReluLinearAttentionPlugin::initialize() noexcept {
    return validConfig(config_) ? 0 : 1;
}

void EdgesegAggregationReluLinearAttentionPlugin::terminate() noexcept {}

size_t EdgesegAggregationReluLinearAttentionPlugin::getSerializationSize() const noexcept {
    return sizeof(int32_t) + sizeof(float) + 4 * sizeof(int32_t);
}

void EdgesegAggregationReluLinearAttentionPlugin::serialize(void* buffer) const noexcept {
    if (buffer == nullptr) {
        return;
    }
    auto* cursor = static_cast<char*>(buffer);
    writeValue<int32_t>(cursor, config_.dim);
    writeValue<float>(cursor, config_.eps);
    writeValue<int32_t>(cursor, config_.qkvC);
    writeValue<int32_t>(cursor, config_.outputC);
    writeValue<int32_t>(cursor, config_.height);
    writeValue<int32_t>(cursor, config_.width);
}

void EdgesegAggregationReluLinearAttentionPlugin::destroy() noexcept {
    delete this;
}

void EdgesegAggregationReluLinearAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept {
    namespace_ = pluginNamespace == nullptr ? "" : pluginNamespace;
}

char const* EdgesegAggregationReluLinearAttentionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

EdgesegAggregationReluLinearAttentionPluginCreator::EdgesegAggregationReluLinearAttentionPluginCreator() {
    fields_.emplace_back(nvinfer1::PluginField{"dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"qkv_c", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"output_c", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"height", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"width", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    fieldCollection_.nbFields = static_cast<int32_t>(fields_.size());
    fieldCollection_.fields = fields_.data();
}

char const* EdgesegAggregationReluLinearAttentionPluginCreator::getPluginName() const noexcept {
    return kAggregationReluLinearAttentionPluginName;
}

char const* EdgesegAggregationReluLinearAttentionPluginCreator::getPluginVersion() const noexcept {
    return kAggregationReluLinearAttentionPluginVersion;
}

nvinfer1::PluginFieldCollection const* EdgesegAggregationReluLinearAttentionPluginCreator::getFieldNames() noexcept {
    return &fieldCollection_;
}

nvinfer1::IPluginV2* EdgesegAggregationReluLinearAttentionPluginCreator::createPlugin(
    char const* name,
    nvinfer1::PluginFieldCollection const* fc) noexcept {
    (void) name;
    AggregationReluLinearAttentionPluginConfig config = parseFields(fc);
    if (!validConfig(config)) {
        return nullptr;
    }
    auto* plugin = new EdgesegAggregationReluLinearAttentionPlugin(config);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::IPluginV2* EdgesegAggregationReluLinearAttentionPluginCreator::deserializePlugin(
    char const* name,
    void const* serialData,
    size_t serialLength) noexcept {
    (void) name;
    auto* plugin = new EdgesegAggregationReluLinearAttentionPlugin(serialData, serialLength);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

void EdgesegAggregationReluLinearAttentionPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept {
    namespace_ = pluginNamespace == nullptr ? "" : pluginNamespace;
}

char const* EdgesegAggregationReluLinearAttentionPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

} // namespace edgeseg

namespace {

edgeseg::EdgesegAggregationReluLinearAttentionPluginCreator gAggregationCreator;

struct EdgesegAggregationReluLinearAttentionRegistrar {
    EdgesegAggregationReluLinearAttentionRegistrar() {
        getPluginRegistry()->registerCreator(
            gAggregationCreator,
            edgeseg::kAggregationReluLinearAttentionPluginNamespace);
    }
};

EdgesegAggregationReluLinearAttentionRegistrar gAggregationRegistrar;

} // namespace

extern "C" __declspec(dllexport) int edgesegAggregationReluLinearAttentionPluginAbiVersion() {
    return 1;
}
