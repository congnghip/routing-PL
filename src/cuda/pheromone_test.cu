#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// CUDA Kernel: Mỗi thread trên GPU đảm nhận việc làm bay hơi 1 ô pheromone
// tau[i][j][d] được phẳng hóa thành mảng 1D: index = i * (N * 2) + j * 2 + d
__global__ void evaporate_pheromone_kernel(float* d_tau, int total_elements, float rho) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_elements) {
        d_tau[idx] *= (1.0f - rho);
    }
}

int main() {
    int N = 1000; // Thử nghiệm với bài toán lớn: 1000 khách hàng (Depot + Customers)
    int D = 2;    // 2 hình thức: Direct (0) và Locker (1)
    int total_elements = N * N * D; // 2,000,000 phần tử pheromone!
    size_t bytes = total_elements * sizeof(float);
    float rho = 0.1f; // Hệ số bay hơi

    std::cout << "--- THỬ NGHIỆM KERNEL BAY HƠI PHEROMONE 3D TRÊN GPU ---" << std::endl;
    std::cout << "Kích thước bài toán : " << N << " nút" << std::endl;
    std::cout << "Tổng số ô pheromone : " << total_elements << " phần tử (" 
              << (double)bytes / (1024 * 1024) << " MB)" << std::endl;

    // 1. Cấp phát và khởi tạo trên Host (CPU RAM)
    std::vector<float> h_tau(total_elements, 1.0f);

    // 2. Cấp phát trên Device (GPU VRAM)
    float* d_tau = nullptr;
    cudaMalloc(&d_tau, bytes);

    // 3. Sao chép dữ liệu từ CPU -> GPU
    cudaMemcpy(d_tau, h_tau.data(), bytes, cudaMemcpyHostToDevice);

    // 4. Thiết lập Grid & Block
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    std::cout << "Cấu hình CUDA       : " << blocks_per_grid << " Blocks, " 
              << threads_per_block << " Threads/Block" << std::endl;
    std::cout << "Tổng số GPU Threads : " << blocks_per_grid * threads_per_block << " luồng" << std::endl;

    // Warm-up để khởi tạo CUDA context
    evaporate_pheromone_kernel<<<blocks_per_grid, threads_per_block>>>(d_tau, total_elements, 0.0f);
    cudaDeviceSynchronize();

    // Đo thời gian bằng CUDA Event qua 50 chu kỳ lặp (mô phỏng 50 iterations của PACO)
    int iterations = 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int it = 0; it < iterations; ++it) {
        evaporate_pheromone_kernel<<<blocks_per_grid, threads_per_block>>>(d_tau, total_elements, rho);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / iterations;

    // 5. Sao chép kết quả từ GPU -> CPU
    cudaMemcpy(h_tau.data(), d_tau, bytes, cudaMemcpyDeviceToHost);

    // 6. Kiểm tra tính chính xác
    bool correct = true;
    float expected = std::pow(1.0f - rho, iterations);
    for (int i = 0; i < 10; ++i) {
        if (std::abs(h_tau[i] - expected) > 1e-4) {
            correct = false;
            break;
        }
    }

    std::cout << "Thời gian chạy trung bình: " << avg_ms << " ms (" 
              << avg_ms * 1000.0f << " microseconds) cho 2 triệu ô pheromone!" << std::endl;
    std::cout << "Kết quả kiểm tra    : " << (correct ? "CHÍNH XÁC (tau đã bay hơi đúng)" : "SAI") << std::endl;

    // Giải phóng bộ nhớ GPU
    cudaFree(d_tau);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
