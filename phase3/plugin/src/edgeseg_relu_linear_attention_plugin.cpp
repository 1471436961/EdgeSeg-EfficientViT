#include "edgeseg_relu_linear_attention_plugin.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>

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

int32_t volume(nvinfer1::Dims const& dims) noexcept {
    if (dims.nbDims <= 0) {
        return 0;
    }
    int64_t result = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return -1;
        }
        result *= dims.d[i];
    }
    return result > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ? -1 : static_cast<int32_t>(result);
}

ReluLinearAttentionPluginConfig parseFields(nvinfer1::PluginFieldCollection const* fc) noexcept {
    ReluLinearAttentionPluginConfig config{};
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
        } else if (std::strcmp(field.name, "input_c") == 0 && field.type == nvinfer1::PluginFieldType::kINT32) {
            config.inputC = *static_cast<int32_t const*>(field.data);
        } else if (std::strcmp(field.name, "height") == 0 && field.type == nvinfer1::PluginFieldType::kINT32) {
            config.height = *static_cast<int32_t const*>(field.data);
        } else if (std::strcmp(field.name, "width") == 0 && field.type == nvinfer1::PluginFieldType::kINT32) {
            config.width = *static_cast<int32_t const*>(field.data);
        }
    }
    return config;
}

bool validConfig(ReluLinearAttentionPluginConfig const& config) noexcept {
    return config.dim > 0 && config.inputC == 3 * config.outputC() && config.inputC % (3 * config.dim) == 0
        && config.height > 0 && config.width > 0;
}

} // namespace

EdgesegReluLinearAttentionPlugin::EdgesegReluLinearAttentionPlugin(ReluLinearAttentionPluginConfig config)
    : config_(config) {}

EdgesegReluLinearAttentionPlugin::EdgesegReluLinearAttentionPlugin(void const* serialData, size_t serialLength) {
    if (serialData == nullptr || serialLength != getSerializationSize()) {
        return;
    }
    auto const* cursor = static_cast<char const*>(serialData);
    config_.dim = readValue<int32_t>(cursor);
    config_.eps = readValue<float>(cursor);
    config_.inputC = readValue<int32_t>(cursor);
    config_.height = readValue<int32_t>(cursor);
    config_.width = readValue<int32_t>(cursor);
}

nvinfer1::IPluginV2DynamicExt* EdgesegReluLinearAttentionPlugin::clone() const noexcept {
    auto* plugin = new EdgesegReluLinearAttentionPlugin(config_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

int32_t EdgesegReluLinearAttentionPlugin::getNbOutputs() const noexcept {
    return 1;
}

nvinfer1::DimsExprs EdgesegReluLinearAttentionPlugin::getOutputDimensions(
    int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    if (outputIndex != 0 || inputs == nullptr || nbInputs != 1) {
        return nvinfer1::DimsExprs{};
    }

    nvinfer1::DimsExprs output = inputs[0];
    if (output.nbDims == 4) {
        output.d[1] = exprBuilder.constant(config_.outputC());
    }
    return output;
}

bool EdgesegReluLinearAttentionPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (inOut == nullptr || nbInputs != 1 || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }
    return inOut[pos].type == nvinfer1::DataType::kFLOAT
        && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void EdgesegReluLinearAttentionPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    (void) out;
    (void) nbOutputs;
    if (in == nullptr || nbInputs != 1 || !validConfig(config_)) {
        return;
    }
    // Build/runtime validation is intentionally conservative in the skeleton.
    (void) dimsEqual(in[0].desc.dims, 1, config_.inputC, config_.height, config_.width);
}

size_t EdgesegReluLinearAttentionPlugin::getWorkspaceSize(
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
    const int32_t heads = config_.inputC / (3 * config_.dim);
    const int64_t elements = static_cast<int64_t>(heads) * (config_.dim + 1) * config_.dim;
    return static_cast<size_t>(elements) * sizeof(float);
}

int32_t EdgesegReluLinearAttentionPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr || inputs[0] == nullptr
        || outputs[0] == nullptr || !validConfig(config_)) {
        return 1;
    }
    if (!dimsEqual(inputDesc[0].dims, 1, config_.inputC, config_.height, config_.width)
        || !dimsEqual(outputDesc[0].dims, 1, config_.outputC(), config_.height, config_.width)) {
        return 1;
    }
    return launchReluLinearAttention(
        static_cast<float const*>(inputs[0]),
        static_cast<float*>(outputs[0]),
        workspace,
        getWorkspaceSize(inputDesc, 1, outputDesc, 1),
        config_,
        stream);
}

nvinfer1::DataType EdgesegReluLinearAttentionPlugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
    (void) inputTypes;
    return index == 0 && nbInputs == 1 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kFLOAT;
}

char const* EdgesegReluLinearAttentionPlugin::getPluginType() const noexcept {
    return kReluLinearAttentionPluginName;
}

char const* EdgesegReluLinearAttentionPlugin::getPluginVersion() const noexcept {
    return kReluLinearAttentionPluginVersion;
}

int32_t EdgesegReluLinearAttentionPlugin::initialize() noexcept {
    return validConfig(config_) ? 0 : 1;
}

void EdgesegReluLinearAttentionPlugin::terminate() noexcept {}

size_t EdgesegReluLinearAttentionPlugin::getSerializationSize() const noexcept {
    return sizeof(int32_t) + sizeof(float) + 3 * sizeof(int32_t);
}

void EdgesegReluLinearAttentionPlugin::serialize(void* buffer) const noexcept {
    if (buffer == nullptr) {
        return;
    }
    auto* cursor = static_cast<char*>(buffer);
    writeValue<int32_t>(cursor, config_.dim);
    writeValue<float>(cursor, config_.eps);
    writeValue<int32_t>(cursor, config_.inputC);
    writeValue<int32_t>(cursor, config_.height);
    writeValue<int32_t>(cursor, config_.width);
}

void EdgesegReluLinearAttentionPlugin::destroy() noexcept {
    delete this;
}

void EdgesegReluLinearAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept {
    namespace_ = pluginNamespace == nullptr ? "" : pluginNamespace;
}

char const* EdgesegReluLinearAttentionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

EdgesegReluLinearAttentionPluginCreator::EdgesegReluLinearAttentionPluginCreator() {
    fields_.emplace_back(nvinfer1::PluginField{"dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"input_c", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"height", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"width", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    fieldCollection_.nbFields = static_cast<int32_t>(fields_.size());
    fieldCollection_.fields = fields_.data();
}

char const* EdgesegReluLinearAttentionPluginCreator::getPluginName() const noexcept {
    return kReluLinearAttentionPluginName;
}

char const* EdgesegReluLinearAttentionPluginCreator::getPluginVersion() const noexcept {
    return kReluLinearAttentionPluginVersion;
}

nvinfer1::PluginFieldCollection const* EdgesegReluLinearAttentionPluginCreator::getFieldNames() noexcept {
    return &fieldCollection_;
}

nvinfer1::IPluginV2* EdgesegReluLinearAttentionPluginCreator::createPlugin(
    char const* name,
    nvinfer1::PluginFieldCollection const* fc) noexcept {
    (void) name;
    ReluLinearAttentionPluginConfig config = parseFields(fc);
    if (!validConfig(config)) {
        return nullptr;
    }
    auto* plugin = new EdgesegReluLinearAttentionPlugin(config);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::IPluginV2* EdgesegReluLinearAttentionPluginCreator::deserializePlugin(
    char const* name,
    void const* serialData,
    size_t serialLength) noexcept {
    (void) name;
    auto* plugin = new EdgesegReluLinearAttentionPlugin(serialData, serialLength);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

void EdgesegReluLinearAttentionPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept {
    namespace_ = pluginNamespace == nullptr ? "" : pluginNamespace;
}

char const* EdgesegReluLinearAttentionPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

} // namespace edgeseg

namespace {

edgeseg::EdgesegReluLinearAttentionPluginCreator gCreator;

struct EdgesegReluLinearAttentionRegistrar {
    EdgesegReluLinearAttentionRegistrar() {
        getPluginRegistry()->registerCreator(gCreator, edgeseg::kReluLinearAttentionPluginNamespace);
    }
};

EdgesegReluLinearAttentionRegistrar gRegistrar;

} // namespace

extern "C" __declspec(dllexport) int edgesegReluLinearAttentionPluginAbiVersion() {
    return 1;
}
