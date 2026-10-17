/**
 * ============================================================================
 * CHƯƠNG TRÌNH MẪU: ĐỐI CHIẾU MÃ NGUỒN C (CPU) VÀ CUDA (GPU)
 * Bài toán: 1 Vòng lặp hoàn chỉnh của đàn kiến 3D-PACO
 * 
 * Mã nguồn được chia làm 2 phần rõ rệt:
 *   [PHẦN A]: Chạy trên CARD ĐỒ HỌA (GPU / Device) - Hàng ngàn luồng chạy cùng lúc
 *   [PHẦN B]: Chạy trên MÁY TÍNH (CPU / Host) - Quản lý, điều phối và cấp phát bộ nhớ
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================================
// [PHẦN A]: CÁC HÀM CHẠY TRÊN CARD ĐỒ HỌA (GPU KERNELS)
// Quy tắc nhận biết: Có từ khóa __global__ ở đầu hàm.
// Đặc điểm: Không có vòng lặp duyệt kiến! Mỗi luồng phần cứng GPU là 1 con kiến.
// ============================================================================

/**
 * KERNEL 1: Khởi tạo bộ sinh số ngẫu nhiên cho từng con kiến
 * Mỗi luồng GPU (thread) nhận 1 hạt giống (seed) độc lập để tung xúc xắc
 */
__global__ void init_rng_kernel(curandState* states, unsigned long seed, int num_ants) {
    // 1. Luồng tự xác định vị trí của mình trong GPU:
    int ant_id = blockDim.x * blockIdx.x + threadIdx.x;

    // 2. Nếu vượt quá số kiến cần tạo thì dừng (bảo vệ bộ nhớ):
    if (ant_id >= num_ants) return;

    // 3. Khởi tạo bộ sinh ngẫu nhiên cho riêng con kiến này:
    curand_init(seed, ant_id, 0, &states[ant_id]);
}

/**
 * KERNEL 2: Từng con kiến tự đi tìm đường và tính chi phí lộ trình
 * ĐÂY LÀ TRÁI TIM CỦA THUẬT TOÁN:
 * Nếu có 1.024 con kiến, hàm này sẽ được 1.024 nhân GPU thực thi ĐỒNG THỜI!
 */
__global__ void ant_construct_and_evaluate_kernel(
    int num_customers,           // Số lượng khách hàng
    const float* d_dist,         // Ma trận khoảng cách nằm trên VRAM
    const float* d_tau,          // Ten-xơ Pheromone 3D nằm trên VRAM
    float alpha,                 // Trọng số pheromone
    curandState* states,         // Trạng thái ngẫu nhiên của từng kiến
    int* d_all_routes,           // VRAM lưu lộ trình: [num_ants x num_customers]
    float* d_all_costs,          // VRAM lưu chi phí quãng đường của từng kiến
    int num_ants                 // Tổng số kiến
) {
    // 1. Xác định: "Tôi là con kiến số mấy?"
    int ant_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (ant_id >= num_ants) return;

    // Lấy bộ sinh số ngẫu nhiên của riêng con kiến này
    curandState local_rng = states[ant_id];

    // Con trỏ trỏ tới vùng nhớ lưu lộ trình của riêng con kiến này:
    int* my_route = &d_all_routes[ant_id * num_customers];

    // Mặt nạ Bit (Bitmask 32-bit): Đánh dấu khách đã thăm
    // 1 chu kỳ xung nhịp để kiểm tra, không cần dùng vector hay mảng cồng kềnh
    unsigned int visited_mask = 0;
    int visited_count = 0;
    int current_node = 0; // Bắt đầu từ Kho (Depot = 0)
    float total_distance = 0.0f;

    // Vòng lặp xây dựng đường đi cho con kiến này:
    while (visited_count < num_customers) {
        float scores[32];
        int candidates[32];
        int cand_count = 0;
        float sum_scores = 0.0f;

        // Tìm các khách chưa thăm
        for (int next_cust = 1; next_cust <= num_customers; ++next_cust) {
            bool da_tham = (visited_mask & (1U << next_cust)) != 0;
            if (!da_tham) {
                // Đọc pheromone từ mảng phẳng 1D: tau[current][next][direct=0]
                int tau_idx = current_node * ((num_customers + 1) * 2) + next_cust * 2 + 0;
                float tau_val = d_tau[tau_idx];

                // Tính điểm hấp dẫn = tau ^ alpha
                float score = powf(tau_val, alpha);

                candidates[cand_count] = next_cust;
                scores[cand_count] = score;
                sum_scores += score;
                cand_count++;
            }
        }

        // Bốc thăm ngẫu nhiên theo xác suất Roulette Wheel dùng cuRAND
        float rand_pick = curand_uniform(&local_rng) * sum_scores;
        float accum = 0.0f;
        int chosen_cust = candidates[0];

        for (int c = 0; c < cand_count; ++c) {
            accum += scores[c];
            if (accum >= rand_pick) {
                chosen_cust = candidates[c];
                break;
            }
        }

        // Ghi nhận khách vừa chọn vào lộ trình
        my_route[visited_count] = chosen_cust;
        visited_mask |= (1U << chosen_cust); // Đánh dấu đã thăm bằng phép bật bit
        visited_count++;

        // Cộng quãng đường di chuyển: dist[current][chosen]
        total_distance += d_dist[current_node * (num_customers + 1) + chosen_cust];
        current_node = chosen_cust;
    }

    // Quay về kho Depot
    total_distance += d_dist[current_node * (num_customers + 1) + 0];

    // Ghi kết quả quãng đường của con kiến này vào VRAM
    d_all_costs[ant_id] = total_distance;

    // Lưu lại trạng thái ngẫu nhiên để dùng cho chu kỳ sau
    states[ant_id] = local_rng;
}

/**
 * KERNEL 3: Bay hơi Pheromone trên toàn bộ ma trận 3D
 * Mỗi luồng GPU chịu trách nhiệm giảm nồng độ của đúng 1 ô pheromone
 */
__global__ void evaporate_kernel(float* d_tau, int total_elements, float rho) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_elements) {
        d_tau[idx] *= (1.0f - rho); // tau = tau * (1 - rho)
    }
}

/**
 * KERNEL 4: Con kiến tốt nhất rải Pheromone (Elitist Update)
 * Sử dụng atomicAdd để chống xung đột ghi bộ nhớ
 */
__global__ void deposit_best_ant_kernel(
    const int* d_best_route,
    float best_cost,
    float* d_tau,
    int num_customers,
    float Q
) {
    int step = threadIdx.x; // Mỗi luồng xử lý 1 chặng đường của con kiến vô địch
    if (step < num_customers) {
        int u = (step == 0) ? 0 : d_best_route[step - 1];
        int v = d_best_route[step];
        int tau_idx = u * ((num_customers + 1) * 2) + v * 2 + 0;

        float delta = Q / best_cost;

        // LỆNH ĐẶC TRƯNG CUDA: Cộng nguyên tử an toàn vào VRAM
        atomicAdd(&d_tau[tau_idx], delta);
    }
}


// ============================================================================
// [PHẦN B]: MÃ CHẠY TRÊN CPU (HOST CODE)
// Được viết bằng C++ chuẩn. Đóng vai trò Tổng chỉ huy điều khiển GPU.
// ============================================================================

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "  MINI 3D-PACO: ĐỐI CHIẾU MÃ NGUỒN C++ (CPU) VÀ CUDA (GPU)" << std::endl;
    std::cout << "=========================================================" << std::endl;

    // 1. Khởi tạo bài toán trên CPU (RAM)
    int num_customers = 10;                     // 10 khách hàng (nút 1 đến 10) + Kho (nút 0)
    int total_nodes = num_customers + 1;
    int num_ants = 1024;                        // 1.024 con kiến chạy song song
    int tau_elements = total_nodes * total_nodes * 2; // Ten-xơ Pheromone 3D
    float alpha = 1.0f;
    float rho = 0.1f;                           // Tỉ lệ bay hơi 10%
    float Q = 100.0f;
    int num_iterations = 5;                     // Chạy thử 5 thế hệ

    std::cout << "Quy mô bài toán  : " << num_customers << " khách hàng" << std::endl;
    std::cout << "Số kiến đồng thời: " << num_ants << " con kiến (1.024 GPU threads)" << std::endl;
    std::cout << "Số vòng lặp      : " << num_iterations << " thế hệ" << std::endl;

    // Tạo ma trận khoảng cách giả lập trên RAM (Host)
    std::vector<float> h_dist(total_nodes * total_nodes);
    for (int i = 0; i < total_nodes; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            h_dist[i * total_nodes + j] = (i == j) ? 0.0f : std::abs(i - j) * 10.0f;
        }
    }

    // Khởi tạo pheromone ban đầu = 1.0 trên RAM
    std::vector<float> h_tau(tau_elements, 1.0f);

    // ------------------------------------------------------------------------
    // BƯỚC 2: CẤP PHÁT BỘ NHỚ TRÊN CARD ĐỒ HỌA (VRAM) BẰNG cudaMalloc
    // Chú ý: Các biến có tiền tố d_ (Device) là con trỏ trỏ vào VRAM của GPU!
    // ------------------------------------------------------------------------
    float *d_dist, *d_tau, *d_all_costs;
    int *d_all_routes;
    curandState* d_rng_states;

    cudaMalloc(&d_dist, total_nodes * total_nodes * sizeof(float));
    cudaMalloc(&d_tau, tau_elements * sizeof(float));
    cudaMalloc(&d_all_costs, num_ants * sizeof(float));
    cudaMalloc(&d_all_routes, num_ants * num_customers * sizeof(int));
    cudaMalloc(&d_rng_states, num_ants * sizeof(curandState));

    // ------------------------------------------------------------------------
    // BƯỚC 3: SAO CHÉP DỮ LIỆU TỪ CPU RAM SANG GPU VRAM QUA KHE PCIe
    // ------------------------------------------------------------------------
    cudaMemcpy(d_dist, h_dist.data(), total_nodes * total_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau, h_tau.data(), tau_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Cấu hình mạng lưới luồng CUDA (Grid & Block)
    int threads_per_block = 128;
    int blocks_for_ants = (num_ants + threads_per_block - 1) / threads_per_block;

    // Khởi tạo bộ sinh số ngẫu nhiên trên GPU
    init_rng_kernel<<<blocks_for_ants, threads_per_block>>>(d_rng_states, 2026ULL, num_ants);
    cudaDeviceSynchronize(); // Chờ GPU hoàn thành

    // Mảng trên CPU để nhận kết quả chi phí của 1.024 con kiến
    std::vector<float> h_costs(num_ants);
    std::vector<int> h_routes(num_ants * num_customers);

    std::cout << "\n--- BẮT ĐẦU VÒNG LẶP TIẾN HÓA TRÊN GPU ---" << std::endl;

    // ------------------------------------------------------------------------
    // BƯỚC 4: VÒNG LẶP TIẾN HÓA KHÉP KÍN TRÊN GPU
    // ------------------------------------------------------------------------
    for (int iter = 1; iter <= num_iterations; ++iter) {
        // [Lệnh CUDA]: Bắn 1.024 nhân GPU đi tìm đường đồng thời
        ant_construct_and_evaluate_kernel<<<blocks_for_ants, threads_per_block>>>(
            num_customers, d_dist, d_tau, alpha, d_rng_states,
            d_all_routes, d_all_costs, num_ants
        );

        // Kéo mảng chi phí về CPU để tìm xem con kiến nào vô địch chu kỳ này
        cudaMemcpy(h_costs.data(), d_all_costs, num_ants * sizeof(float), cudaMemcpyDeviceToHost);

        int best_ant_idx = 0;
        float best_cost = h_costs[0];
        for (int a = 1; a < num_ants; ++a) {
            if (h_costs[a] < best_cost) {
                best_cost = h_costs[a];
                best_ant_idx = a;
            }
        }

        std::cout << "Thế hệ " << iter << " | Quãng đường ngắn nhất tìm được: " << best_cost << " km" << std::endl;

        // [Lệnh CUDA]: Cho toàn bộ 242 ô Pheromone bay hơi đồng thời
        int blocks_for_tau = (tau_elements + 255) / 256;
        evaporate_kernel<<<blocks_for_tau, 256>>>(d_tau, tau_elements, rho);

        // [Lệnh CUDA]: Con kiến tốt nhất rải thêm Pheromone
        int* d_best_route = &d_all_routes[best_ant_idx * num_customers];
        deposit_best_ant_kernel<<<1, num_customers>>>(d_best_route, best_cost, d_tau, num_customers, Q);
        
        cudaDeviceSynchronize();
    }

    // ------------------------------------------------------------------------
    // BƯỚC 5: GIẢI PHÓNG BỘ NHỚ VRAM (Tương tự free() trong C)
    // ------------------------------------------------------------------------
    cudaFree(d_dist);
    cudaFree(d_tau);
    cudaFree(d_all_costs);
    cudaFree(d_all_routes);
    cudaFree(d_rng_states);

    std::cout << "\nHoàn thành 5 thế hệ tiến hóa thành công!" << std::endl;
    return 0;
}
