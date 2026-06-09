#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << '\n';
        }
    }
};

struct DeviceBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t nbytes) : bytes(nbytes) {
        if (bytes > 0) {
            checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
        other.ptr = nullptr;
        other.bytes = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    ~DeviceBuffer() {
        release();
    }

    static void checkCuda(cudaError_t status, const char* what) {
        if (status != cudaSuccess) {
            std::ostringstream oss;
            oss << what << " failed: " << cudaGetErrorString(status);
            throw std::runtime_error(oss.str());
        }
    }

private:
    void release() noexcept {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }
};

struct CudaStream {
    cudaStream_t stream{nullptr};

    CudaStream() {
        DeviceBuffer::checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    ~CudaStream() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open engine file: " + path);
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("engine file is empty: " + path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<std::size_t>(size));
    file.read(data.data(), size);
    if (!file) {
        throw std::runtime_error("failed to read full engine file: " + path);
    }
    return data;
}

std::string dimsToString(const nvinfer1::Dims& dims) {
    std::ostringstream oss;
    oss << '[';
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dims.d[i];
    }
    oss << ']';
    return oss.str();
}

std::size_t volume(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) {
        return 0;
    }
    std::size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            throw std::runtime_error("dynamic dimension is not supported by this demo");
        }
        v *= static_cast<std::size_t>(dims.d[i]);
    }
    return v;
}

std::size_t elementSize(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        throw std::runtime_error("unsupported TensorRT dtype");
    }
}

std::string dtypeToString(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return "float32";
    case nvinfer1::DataType::kHALF:
        return "float16";
    case nvinfer1::DataType::kINT32:
        return "int32";
    case nvinfer1::DataType::kINT8:
        return "int8";
    case nvinfer1::DataType::kBOOL:
        return "bool";
    default:
        return "unknown";
    }
}

std::vector<float> makeInput(std::size_t count) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        const int residue = static_cast<int>(i % 251);
        values[i] = (static_cast<float>(residue) / 125.0F) - 1.0F;
    }
    return values;
}

void printStats(const std::vector<float>& values) {
    if (values.empty()) {
        std::cout << "output is empty\n";
        return;
    }
    const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double mean = sum / static_cast<double>(values.size());

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "output elements: " << values.size() << '\n';
    std::cout << "output sum:      " << sum << '\n';
    std::cout << "output mean:     " << mean << '\n';
    std::cout << "output min/max:  " << *minIt << " / " << *maxIt << '\n';
    std::cout << "first values:    ";
    const std::size_t n = std::min<std::size_t>(values.size(), 8);
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << values[i];
    }
    std::cout << '\n';
}

std::string defaultEnginePath() {
    return "phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string enginePath = argc >= 2 ? argv[1] : defaultEnginePath();
        const int iterations = argc >= 3 ? std::max(1, std::stoi(argv[2])) : 1;

        std::cout << "engine:     " << enginePath << '\n';
        std::cout << "iterations: " << iterations << '\n';

        Logger logger;
        initLibNvInferPlugins(&logger, "");

        const std::vector<char> engineBytes = readFile(enginePath);
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
        if (runtime == nullptr) {
            throw std::runtime_error("createInferRuntime returned null");
        }

        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        if (engine == nullptr) {
            runtime->destroy();
            throw std::runtime_error("deserializeCudaEngine returned null");
        }

        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        if (context == nullptr) {
            engine->destroy();
            runtime->destroy();
            throw std::runtime_error("createExecutionContext returned null");
        }

        const int nbBindings = engine->getNbBindings();
        std::vector<void*> bindings(static_cast<std::size_t>(nbBindings), nullptr);
        std::vector<DeviceBuffer> buffers;
        buffers.reserve(static_cast<std::size_t>(nbBindings));

        int inputIndex = -1;
        int outputIndex = -1;
        std::size_t inputElements = 0;
        std::size_t outputElements = 0;

        std::cout << "\nBindings\n";
        for (int i = 0; i < nbBindings; ++i) {
            const char* name = engine->getBindingName(i);
            const bool isInput = engine->bindingIsInput(i);
            const nvinfer1::Dims dims = context->getBindingDimensions(i);
            const nvinfer1::DataType dtype = engine->getBindingDataType(i);
            const std::size_t elems = volume(dims);
            const std::size_t bytes = elems * elementSize(dtype);

            if (dtype != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("this first-version C++ demo supports FP32 bindings only");
            }

            buffers.emplace_back(bytes);
            bindings[static_cast<std::size_t>(i)] = buffers.back().ptr;

            if (isInput) {
                inputIndex = i;
                inputElements = elems;
            } else {
                outputIndex = i;
                outputElements = elems;
            }

            std::cout << "  [" << i << "] "
                      << (isInput ? "input " : "output")
                      << " name=" << name
                      << " dtype=" << dtypeToString(dtype)
                      << " shape=" << dimsToString(dims)
                      << " bytes=" << bytes << '\n';
        }

        if (inputIndex < 0 || outputIndex < 0) {
            throw std::runtime_error("expected one input and one output binding");
        }

        const std::vector<float> hostInput = makeInput(inputElements);
        std::vector<float> hostOutput(outputElements, 0.0F);

        CudaStream stream;
        DeviceBuffer::checkCuda(
            cudaMemcpyAsync(
                bindings[static_cast<std::size_t>(inputIndex)],
                hostInput.data(),
                hostInput.size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream.stream),
            "cudaMemcpyAsync H2D");

        for (int i = 0; i < iterations; ++i) {
            if (!context->enqueueV2(bindings.data(), stream.stream, nullptr)) {
                throw std::runtime_error("enqueueV2 failed");
            }
        }

        DeviceBuffer::checkCuda(
            cudaMemcpyAsync(
                hostOutput.data(),
                bindings[static_cast<std::size_t>(outputIndex)],
                hostOutput.size() * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream.stream),
            "cudaMemcpyAsync D2H");
        DeviceBuffer::checkCuda(cudaStreamSynchronize(stream.stream), "cudaStreamSynchronize");

        std::cout << "\nOutput summary\n";
        printStats(hostOutput);

        context->destroy();
        engine->destroy();
        runtime->destroy();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
