#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "Không tìm thấy thiết bị GPU NVIDIA nào hỗ trợ CUDA." << std::endl;
        return 0;
    }

    std::cout << "======================================================" << std::endl;
    std::cout << "   KIỂM TRA THÔNG SỐ PHẦN CỨNG GPU CHO 3D-PACO" << std::endl;
    std::cout << "======================================================" << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "Thiết bị #" << dev << ": " << prop.name << std::endl;
        std::cout << "  - Compute Capability       : " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  - Tổng dung lượng VRAM     : " << prop.totalGlobalMem / (1024 * 1024) << " MB (" 
                  << (double)prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB)" << std::endl;
        std::cout << "  - Số bộ đa xử lý (SMs)     : " << prop.multiProcessorCount << std::endl;
        std::cout << "  - Warp Size (Luồng/Warp)   : " << prop.warpSize << std::endl;
        std::cout << "  - Max Threads / Block      : " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  - Max Threads / SM         : " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  - Shared Memory / Block    : " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  - Constant Memory          : " << prop.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "  - Memory Bus Width         : " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  - Memory Clock Rate        : " << prop.memoryClockRate * 1e-3f << " MHz" << std::endl;
    }
    std::cout << "======================================================" << std::endl;
    return 0;
}
