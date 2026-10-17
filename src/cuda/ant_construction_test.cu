#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Cấu trúc dữ liệu bài toán VRPPL trên GPU (POD - Plain Old Data)
struct VRPDataGPU {
    int num_customers;
    int num_lockers;
    int total_nodes;
    int vehicle_capacity;
    int num_vehicles;
    
    // Mảng dữ liệu trên GPU Device
    int* d_cust_type;       // customer_type: 1 (home), 2 (locker), 3 (flex)
    int* d_cust_demand;     // demand từng khách
    int* d_cust_locker;     // locker được gán nếu chọn d=1
    int* d_node_early;      // early time window
    int* d_node_late;       // late time window
    float* d_dist;          // Distance matrix phẳng 1D [total_nodes * total_nodes]
    float* d_tau;           // Pheromone tensor phẳng 1D [(num_customers+1) * (num_customers+1) * 2]
};

// Kernel khởi tạo bộ sinh số ngẫu nhiên cuRAND cho từng kiến (Thread)
__global__ void init_curand_kernel(curandState* state, unsigned long seed, int num_ants) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_ants) {
        curand_init(seed, id, 0, &state[id]);
    }
}

// Kernel xây dựng lộ trình cho từng con kiến (1 Thread = 1 Con kiến)
__global__ void construct_ant_solutions_kernel(
    int num_customers,
    int total_nodes,
    int vehicle_capacity,
    int num_vehicles,
    const int* __restrict__ cust_type,
    const int* __restrict__ cust_demand,
    const int* __restrict__ cust_locker,
    const int* __restrict__ node_early,
    const int* __restrict__ node_late,
    const float* __restrict__ dist,
    const float* __restrict__ tau,
    float alpha,
    int num_ants,
    curandState* state,
    int* __restrict__ d_all_perms,     // [num_ants * num_customers]
    int* __restrict__ d_all_delivery,  // [num_ants * num_customers]
    float* __restrict__ d_all_costs    // [num_ants]
) {
    int ant_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (ant_id >= num_ants) return;

    curandState local_state = state[ant_id];
    int n = num_customers;

    // Bitmask lưu trữ trạng thái các khách đã thăm (hỗ trợ tối đa 1024 khách hàng = 32 ints)
    unsigned int visited_mask[32] = {0};
    int visited_count = 0;

    int* my_perm = &d_all_perms[ant_id * n];
    int* my_delivery = &d_all_delivery[ant_id * n];

    int prev_node = 0; // Bắt đầu từ Depot (ID = 0)

    // VÒNG LẶP XÂY DỰNG LỘ TRÌNH (Từng bước chọn khách kế tiếp)
    while (visited_count < n) {
        float scores[64];      // Mảng tạm lưu xác suất cho tập ứng viên
        int cand_cust[64];
        int cand_d[64];
        int cand_count = 0;
        float sum_scores = 0.0f;

        // Duyệt qua tất cả khách hàng chưa thăm
        for (int j = 1; j <= n; ++j) {
            int word_idx = j / 32;
            int bit_idx = j % 32;
            if ((visited_mask[word_idx] & (1U << bit_idx)) != 0) continue; // Đã thăm

            int c_type = cust_type[j - 1];
            // Quyết định các hình thức giao d được phép:
            // d=0: Direct, d=1: Locker
            int d_start = (c_type == 2) ? 1 : 0;
            int d_end   = (c_type == 1) ? 0 : 1;

            for (int d = d_start; d <= d_end; ++d) {
                // Chỉ số trong mảng phẳng tau: prev * (N+1)*2 + j*2 + d
                int tau_idx = prev_node * ((n + 1) * 2) + j * 2 + d;
                float tau_val = tau[tau_idx];
                if (tau_val > 0.0f) {
                    float score = powf(tau_val, alpha);
                    if (cand_count < 64) {
                        cand_cust[cand_count] = j;
                        cand_d[cand_count] = d;
                        scores[cand_count] = score;
                        sum_scores += score;
                        cand_count++;
                    }
                }
            }
        }

        if (cand_count == 0) break; // Đã chọn hết

        // Rút thăm Roulette Wheel Selection bằng cuRAND
        float rand_val = curand_uniform(&local_state) * sum_scores;
        float accum = 0.0f;
        int selected_idx = 0;
        for (int c = 0; c < cand_count; ++c) {
            accum += scores[c];
            if (accum >= rand_val) {
                selected_idx = c;
                break;
            }
        }

        int chosen_cust = cand_cust[selected_idx];
        int chosen_d = cand_d[selected_idx];

        // Đánh dấu đã thăm vào bitmask
        visited_mask[chosen_cust / 32] |= (1U << (chosen_cust % 32));
        my_perm[visited_count] = chosen_cust;
        my_delivery[visited_count] = chosen_d;
        visited_count++;
        prev_node = chosen_cust;
    }

    state[ant_id] = local_state; // Lưu lại trạng thái RNG

    // ĐÁNH GIÁ CHI PHÍ LỘ TRÌNH (VRP ROUTE EVALUATION) NGAY TRÊN GPU
    float total_distance = 0.0f;
    int curr_node = 0;
    int curr_time = 0;
    int curr_load = 0;
    int vehicle_count = 1;
    int last_locker = -1;

    for (int step = 0; step < n; ++step) {
        int cust_id = my_perm[step];
        int d = my_delivery[step];
        int delivery_node = (d == 0) ? cust_id : cust_locker[cust_id - 1];
        int demand = cust_demand[cust_id - 1];

        float travel_time = dist[curr_node * total_nodes + delivery_node];
        int arr_time = curr_time + (int)travel_time;
        int early = node_early[delivery_node];
        int late  = node_late[delivery_node];

        bool is_locker = (delivery_node > num_customers);
        bool duplicate_locker = is_locker && (delivery_node == last_locker);

        bool violates = (curr_load + demand > vehicle_capacity) || 
                        (arr_time > late) || 
                        duplicate_locker;

        if (violates) {
            // Xe quay về Depot
            total_distance += dist[curr_node * total_nodes + 0];
            vehicle_count++;
            // Khởi tạo xe mới
            curr_node = 0;
            curr_load = 0;
            curr_time = 0;
            last_locker = -1;
            step--; // Đánh giá lại khách hàng này cho xe mới
            continue;
        }

        total_distance += travel_time;
        curr_time = (arr_time > early) ? arr_time : early;
        curr_load += demand;
        curr_node = delivery_node;
        if (is_locker) last_locker = delivery_node;
    }

    // Kết thúc route: quay về Depot
    if (curr_node != 0) {
        total_distance += dist[curr_node * total_nodes + 0];
    }

    // Ghi nhận hàm mục tiêu của con kiến
    d_all_costs[ant_id] = (vehicle_count > num_vehicles) ? (total_distance + 10000.0f) : total_distance;
}

// Đọc instance từ file benchmark
bool load_instance(const std::string& path, 
                   int& num_cust, int& num_lock, int& num_veh, int& cap,
                   std::vector<int>& cust_types,
                   std::vector<int>& demands,
                   std::vector<int>& pref_lockers,
                   std::vector<int>& early_arr,
                   std::vector<int>& late_arr,
                   std::vector<float>& dist_matrix) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    file >> num_cust >> num_lock;
    file >> num_veh >> cap;

    demands.resize(num_cust);
    for (int i = 0; i < num_cust; ++i) file >> demands[i];

    int total_nodes = 1 + num_cust + num_lock;
    std::vector<float> xs(total_nodes), ys(total_nodes);
    early_arr.resize(total_nodes);
    late_arr.resize(total_nodes);
    cust_types.resize(num_cust);

    for (int i = 0; i < total_nodes; ++i) {
        int service, type;
        file >> xs[i] >> ys[i] >> early_arr[i] >> late_arr[i] >> service >> type;
        if (i > 0 && i <= num_cust) {
            cust_types[i - 1] = type;
        }
    }

    // Đọc sở thích locker của khách
    pref_lockers.assign(num_cust, 1 + num_cust); // Mặc định locker đầu tiên
    for (int i = 0; i < num_cust; ++i) {
        for (int l = 0; l < num_lock; ++l) {
            int pref;
            file >> pref;
            if (pref == 1) {
                pref_lockers[i] = 1 + num_cust + l;
            }
        }
    }

    // Tính ma trận khoảng cách Euclid
    dist_matrix.resize(total_nodes * total_nodes);
    for (int i = 0; i < total_nodes; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            float dx = xs[i] - xs[j];
            float dy = ys[i] - ys[j];
            dist_matrix[i * total_nodes + j] = std::sqrt(dx * dx + dy * dy);
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::string instance_file = "src/data/100/C101_co_100.txt"; // Thử ngay với bộ 100 khách hàng chuẩn!
    int num_cust, num_lock, num_veh, cap;
    std::vector<int> cust_types, demands, pref_lockers, early_arr, late_arr;
    std::vector<float> dist_matrix;

    if (!load_instance(instance_file, num_cust, num_lock, num_veh, cap,
                      cust_types, demands, pref_lockers, early_arr, late_arr, dist_matrix)) {
        std::cerr << "Không thể đọc file: " << instance_file << std::endl;
        return 1;
    }

    int total_nodes = 1 + num_cust + num_lock;
    int tau_size = (num_cust + 1) * (num_cust + 1) * 2;
    std::vector<float> h_tau(tau_size, 1.0f); // Khởi tạo tau = 1.0

    std::cout << "===========================================================" << std::endl;
    std::cout << "  BENCHMARK KERNEL ANT TOUR CONSTRUCTION TRÊN GPU (CUDA)   " << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "Tệp thử nghiệm        : " << instance_file << std::endl;
    std::cout << "Số khách hàng         : " << num_cust << " khách" << std::endl;
    std::cout << "Số tủ lockers         : " << num_lock << " tủ" << std::endl;
    std::cout << "Sức chứa xe           : " << cap << std::endl;

    // Cấp phát bộ nhớ trên GPU
    int *d_cust_type, *d_cust_demand, *d_cust_locker, *d_node_early, *d_node_late;
    float *d_dist, *d_tau;

    cudaMalloc(&d_cust_type, num_cust * sizeof(int));
    cudaMalloc(&d_cust_demand, num_cust * sizeof(int));
    cudaMalloc(&d_cust_locker, num_cust * sizeof(int));
    cudaMalloc(&d_node_early, total_nodes * sizeof(int));
    cudaMalloc(&d_node_late, total_nodes * sizeof(int));
    cudaMalloc(&d_dist, total_nodes * total_nodes * sizeof(float));
    cudaMalloc(&d_tau, tau_size * sizeof(float));

    // Nạp dữ liệu vào GPU VRAM
    cudaMemcpy(d_cust_type, cust_types.data(), num_cust * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cust_demand, demands.data(), num_cust * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cust_locker, pref_lockers.data(), num_cust * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_early, early_arr.data(), total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_late, late_arr.data(), total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist_matrix.data(), total_nodes * total_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau, h_tau.data(), tau_size * sizeof(float), cudaMemcpyHostToDevice);

    // THỬ NGHIỆM VỚI QUẦN THỂ KIẾN TÙY BIẾN (Mặc định 4,096 hoặc truyền qua CLI)
    int num_ants = 4096;
    if (argc > 1) {
        num_ants = std::atoi(argv[1]);
    }
    curandState* d_rng_states;
    cudaMalloc(&d_rng_states, num_ants * sizeof(curandState));

    int *d_all_perms, *d_all_delivery;
    float* d_all_costs;
    cudaMalloc(&d_all_perms, num_ants * num_cust * sizeof(int));
    cudaMalloc(&d_all_delivery, num_ants * num_cust * sizeof(int));
    cudaMalloc(&d_all_costs, num_ants * sizeof(float));

    int threads_per_block = 128;
    int blocks = (num_ants + threads_per_block - 1) / threads_per_block;

    // Khởi tạo cuRAND
    init_curand_kernel<<<blocks, threads_per_block>>>(d_rng_states, 1234ULL, num_ants);
    cudaDeviceSynchronize();

    std::cout << "Quy mô quần thể kiến  : " << num_ants << " con kiến đồng thời" << std::endl;
    std::cout << "Cấu hình phân bổ CUDA : " << blocks << " Blocks x " << threads_per_block << " Threads" << std::endl;

    // Đo thời gian GPU tạo và đánh giá 4,096 lộ trình
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    construct_ant_solutions_kernel<<<blocks, threads_per_block>>>(
        num_cust, total_nodes, cap, num_veh,
        d_cust_type, d_cust_demand, d_cust_locker,
        d_node_early, d_node_late, d_dist, d_tau,
        1.0f, num_ants, d_rng_states,
        d_all_perms, d_all_delivery, d_all_costs
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Lấy chi phí các con kiến về Host để tìm lộ trình tốt nhất
    std::vector<float> h_all_costs(num_ants);
    cudaMemcpy(h_all_costs.data(), d_all_costs, num_ants * sizeof(float), cudaMemcpyDeviceToHost);

    float best_cost = 1e9f;
    float sum_cost = 0;
    for (float c : h_all_costs) {
        if (c < best_cost) best_cost = c;
        sum_cost += c;
    }

    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "Thời gian GPU chạy    : " << elapsed_ms << " ms" << std::endl;
    std::cout << "Tốc độ trung bình     : " << (elapsed_ms / num_ants) * 1000.0f << " micro-giây / con kiến!" << std::endl;
    std::cout << "Lộ trình ngắn nhất    : " << best_cost << " km" << std::endl;
    std::cout << "Chi phí trung bình    : " << sum_cost / num_ants << " km" << std::endl;
    std::cout << "===========================================================" << std::endl;

    // Giải phóng bộ nhớ
    cudaFree(d_cust_type);
    cudaFree(d_cust_demand);
    cudaFree(d_cust_locker);
    cudaFree(d_node_early);
    cudaFree(d_node_late);
    cudaFree(d_dist);
    cudaFree(d_tau);
    cudaFree(d_rng_states);
    cudaFree(d_all_perms);
    cudaFree(d_all_delivery);
    cudaFree(d_all_costs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
