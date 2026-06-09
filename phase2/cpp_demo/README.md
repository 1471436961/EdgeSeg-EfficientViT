# TensorRT C++ Runtime Demo

这个目录用于 Phase 2 Step 7：验证 TensorRT FP32 engine 能被 C++ Runtime API 加载和执行。

它不是性能 benchmark，也不替代 Python 侧的 PyTorch / TensorRT 输出对齐。第一版只做 runtime smoke：

- 读取 `phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine`
- 反序列化 TensorRT engine
- 打印 binding 信息
- 分配 CUDA input/output buffer
- 用 deterministic dummy input 执行推理
- 打印 output checksum / min / max / mean / first values

## 环境要求

- TensorRT Windows zip：`E:/NVIDIA/TensorRT-8.6.1.6`
- CUDA Toolkit：当前项目使用 CUDA 12.4 路径
- CMake：已安装到 `D:/software/anaconda3/envs/efficientvit/Library/bin/cmake.exe`
- MSVC C++ Build Tools：最小组件已安装到 `E:/VSBuildTools`，包括 `cl.exe` / `nmake.exe` / `VsDevCmd.bat`。

## 构建

在 Visual Studio Developer PowerShell 中执行：

```powershell
cmd /c "`"E:/VSBuildTools/Common7/Tools/VsDevCmd.bat`" -arch=x64 -host_arch=x64 && D:/software/anaconda3/envs/efficientvit/Library/bin/cmake.exe -S phase2/cpp_demo -B phase2/cpp_demo/build -G `"NMake Makefiles`" -DTENSORRT_ROOT=E:/NVIDIA/TensorRT-8.6.1.6 && D:/software/anaconda3/envs/efficientvit/Library/bin/cmake.exe --build phase2/cpp_demo/build --config Release"
```

## 运行

运行前需要让当前进程看到 TensorRT、CUDA、cuDNN、cuBLAS、NVRTC 的 DLL：

```powershell
cmd /c "set PATH=E:\NVIDIA\TensorRT-8.6.1.6\lib;E:\NVIDIA\TensorRT-8.6.1.6\bin;D:\software\anaconda3\envs\efficientvit\Lib\site-packages\nvidia\cudnn\bin;D:\software\anaconda3\envs\efficientvit\Lib\site-packages\nvidia\cublas\bin;D:\software\anaconda3\envs\efficientvit\Lib\site-packages\nvidia\cuda_nvrtc\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%&& phase2\cpp_demo\build\trt_runtime_demo.exe phase2\results\engines\efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine 1"
```

最后一个参数是 inference 迭代次数，默认 `1`。

## 预期输出

程序应输出：

- engine 路径
- input / output binding name、dtype、shape、bytes
- output elements
- output sum / mean / min / max
- first 8 output values

如果看到 DLL 缺失错误，请先确认 TensorRT `lib` / `bin` 与 CUDA `bin` 已加入当前进程 `PATH`。
