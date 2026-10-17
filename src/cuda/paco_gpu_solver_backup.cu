/**
 * ============================================================================
 * WARP-COOPERATIVE 3D-TENSOR PACO ON GPU FOR VRPPL
 * Full Closed-Loop GPU Solver with:
 *   1. Flattened 1D Coalesced 3D Pheromone Tensor tau[i][j][o]
 *   2. Warp-Cooperative Ant Construction (1 Warp = 32 Threads = 1 Ant)
 *      using Register-Only Gumbel-Max Trick & Warp Shuffle (__shfl_xor_sync)
 *   3. 3D Spatio-Temporal Heuristic (Distance + Locker Consolidation + Time Window)
 *   4. Warp-Parallel Local Search (32 simultaneous 2-opt / Swap / Mode-Flip moves)
 *   5. Elitist Atomic Pheromone Update with Sigmoid Adaptive Evaporation
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace fs = std::filesystem;

#define WARP_SIZE 32
#define MAX_CUST 128
#define MAX_NODES 160

struct InstanceHost {
    std::string name;
    int num_customers;
    int num_lockers;
    int total_nodes;
    int num_vehicles;
    int vehicle_capacity;
    std::vector<int> cust_type;       // [num_customers], 1: Home, 2: Locker, 3: Flex
    std::vector<int> cust_demand;     // [num_customers]
    std::vector<int> cust_locker;     // [num_customers], Locker node ID (1+N..N+K)
    std::vector<int> node_early;      // [total_nodes]
    std::vector<int> node_late;       // [total_nodes]
    std::vector<float> dist_matrix;   // [total_nodes * total_nodes]
};

bool parse_vrppl_instance(const std::string& filepath, InstanceHost& inst) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) return false;
    inst.name = fs::path(filepath).filename().string();

    infile >> inst.num_customers >> inst.num_lockers;
    infile >> inst.num_vehicles >> inst.vehicle_capacity;

    inst.cust_demand.resize(inst.num_customers);
    for (int i = 0; i < inst.num_customers; ++i) {
        infile >> inst.cust_demand[i];
    }

    inst.total_nodes = 1 + inst.num_customers + inst.num_lockers;
    std::vector<double> xs(inst.total_nodes), ys(inst.total_nodes);
    inst.node_early.resize(inst.total_nodes);
    inst.node_late.resize(inst.total_nodes);
    inst.cust_type.resize(inst.num_customers);

    for (int i = 0; i < inst.total_nodes; ++i) {
        int service, type;
        infile >> xs[i] >> ys[i] >> inst.node_early[i] >> inst.node_late[i] >> service >> type;
        if (i >= 1 && i <= inst.num_customers) {
            inst.cust_type[i - 1] = type;
        }
    }

    inst.cust_locker.assign(inst.num_customers, (inst.num_lockers > 0) ? (1 + inst.num_customers) : 0);
    for (int i = 0; i < inst.num_customers; ++i) {
        int assigned = -1;
        for (int k = 0; k < inst.num_lockers; ++k) {
            int pref;
            infile >> pref;
            if (pref == 1 && assigned == -1) {
                assigned = 1 + inst.num_customers + k;
            }
        }
        if (assigned == -1 && inst.num_lockers > 0) {
            assigned = 1 + inst.num_customers;
        }
        inst.cust_locker[i] = assigned;
    }

    inst.dist_matrix.resize(inst.total_nodes * inst.total_nodes);
    for (int i = 0; i < inst.total_nodes; ++i) {
        for (int j = 0; j < inst.total_nodes; ++j) {
            double dx = xs[i] - xs[j];
            double dy = ys[i] - ys[j];
            inst.dist_matrix[i * inst.total_nodes + j] = static_cast<float>(std::sqrt(dx * dx + dy * dy));
        }
    }
    return true;
}

/**
 * Exact Route Evaluator on Device (Matches Solver.cpp::construct_routes 100%)
 */
__device__ __forceinline__ float evaluate_solution_device(
    int n,
    int total_nodes,
    int cap,
    const int* __restrict__ perm,
    const int* __restrict__ deliv_mode,
    const int* __restrict__ cust_demand,
    const int* __restrict__ cust_locker,
    const int* __restrict__ node_early,
    const int* __restrict__ node_late,
    const float* __restrict__ dist,
    int* out_vehicles = nullptr
) {
    float total_distance = 0.0f;
    int vehicle = 0;
    int load = 0;
    int time = 0;
    int curr_node = 0;
    int last_node = 0;
    unsigned int visited_lockers_mask = 0U;

    for (int idx = 0; idx < n; ++idx) {
        int cust_id = perm[idx];
        int d = deliv_mode[idx];
        int delivery_node = (d == 0) ? cust_id : cust_locker[cust_id - 1];
        int demand = cust_demand[cust_id - 1];

        int arr_time = time + static_cast<int>(dist[curr_node * total_nodes + delivery_node]);
        int early = node_early[delivery_node];
        int late = node_late[delivery_node];

        bool is_locker = (delivery_node > n);
        bool duplicate_locker = false;
        if (is_locker) {
            int locker_bit = delivery_node - (n + 1);
            if ((visited_lockers_mask & (1U << locker_bit)) != 0U && delivery_node != last_node) {
                duplicate_locker = true;
            }
        }

        bool violates = (load + demand > cap) || (arr_time > late) || duplicate_locker;
        if (violates) {
            if (curr_node != 0) {
                total_distance += dist[curr_node * total_nodes + 0];
            }
            vehicle++;
            load = 0;
            curr_node = 0;
            time = 0;
            visited_lockers_mask = 0U;
            last_node = 0;
            idx--;
            continue;
        }

        total_distance += dist[curr_node * total_nodes + delivery_node];
        time = (arr_time > early) ? arr_time : early;
        load += demand;
        curr_node = delivery_node;
        if (is_locker) {
            int locker_bit = delivery_node - (n + 1);
            visited_lockers_mask |= (1U << locker_bit);
        }
        last_node = delivery_node;
    }

    if (curr_node != 0) {
        total_distance += dist[curr_node * total_nodes + 0];
        vehicle++;
    }
    if (out_vehicles) *out_vehicles = vehicle;
    return total_distance;
}

// Host verification to guarantee 100% consistency with CPU Solver.cpp
void evaluate_solution_host(
    const InstanceHost& inst,
    const std::vector<int>& perm,
    const std::vector<int>& deliv_mode,
    float& out_dist,
    int& out_vehicles
) {
    int n = inst.num_customers;
    int total_nodes = inst.total_nodes;
    int cap = inst.vehicle_capacity;
    double total_distance = 0.0;
    int vehicle = 0;
    int load = 0;
    int time = 0;
    int curr_node = 0;
    int last_node = 0;
    unsigned int visited_lockers_mask = 0U;

    for (int idx = 0; idx < n; ++idx) {
        int cust_id = perm[idx];
        int d = deliv_mode[idx];
        int delivery_node = (d == 0) ? cust_id : inst.cust_locker[cust_id - 1];
        int demand = inst.cust_demand[cust_id - 1];

        int arr_time = time + static_cast<int>(inst.dist_matrix[curr_node * total_nodes + delivery_node]);
        int early = inst.node_early[delivery_node];
        int late = inst.node_late[delivery_node];

        bool is_locker = (delivery_node > n);
        bool duplicate_locker = false;
        if (is_locker) {
            int locker_bit = delivery_node - (n + 1);
            if ((visited_lockers_mask & (1U << locker_bit)) != 0U && delivery_node != last_node) {
                duplicate_locker = true;
            }
        }

        bool violates = (load + demand > cap) || (arr_time > late) || duplicate_locker;
        if (violates) {
            if (curr_node != 0) {
                total_distance += inst.dist_matrix[curr_node * total_nodes + 0];
            }
            vehicle++;
            load = 0;
            curr_node = 0;
            time = 0;
            visited_lockers_mask = 0U;
            last_node = 0;
            idx--;
            continue;
        }

        total_distance += inst.dist_matrix[curr_node * total_nodes + delivery_node];
        time = std::max(arr_time, early);
        load += demand;
        curr_node = delivery_node;
        if (is_locker) {
            int locker_bit = delivery_node - (n + 1);
            visited_lockers_mask |= (1U << locker_bit);
        }
        last_node = delivery_node;
    }

    if (curr_node != 0) {
        total_distance += inst.dist_matrix[curr_node * total_nodes + 0];
        vehicle++;
    }
    out_dist = static_cast<float>(total_distance);
    out_vehicles = vehicle;
}

__global__ void init_rng_kernel(curandState* states, unsigned long long seed, int total_threads) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < total_threads) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

/**
 * WARP-COOPERATIVE ANT CONSTRUCTION & PARALLEL LOCAL SEARCH KERNEL
 * Each Warp (32 threads) collaboratively builds 1 Ant tour and refines it.
 */
__global__ void warp_paco_construct_and_ls_kernel(
    int iter,
    int n,
    int total_nodes,
    int cap,
    int top_t,
    const int* __restrict__ cust_type,
    const int* __restrict__ cust_demand,
    const int* __restrict__ cust_locker,
    const int* __restrict__ node_early,
    const int* __restrict__ node_late,
    const float* __restrict__ dist,
    const float* __restrict__ tau,
    float alpha,
    float beta,
    int ls_steps,
    int num_ants,
    curandState* rng_states,
    const int* __restrict__ d_elite_perms,
    const int* __restrict__ d_elite_modes,
    int* __restrict__ d_all_perms,
    int* __restrict__ d_all_modes,
    float* __restrict__ d_all_costs
) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int ant_id = global_tid / WARP_SIZE;
    int lane = global_tid % WARP_SIZE;

    if (ant_id >= num_ants) return;

    curandState local_rng = rng_states[global_tid];

    // Shared memory per block (4 warps per block = 128 threads)
    __shared__ int s_perm[4][MAX_CUST];
    __shared__ int s_mode[4][MAX_CUST];
    __shared__ int s_cand_perm[4][MAX_CUST];
    __shared__ int s_cand_mode[4][MAX_CUST];

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int* my_perm = s_perm[warp_in_block];
    int* my_mode = s_mode[warp_in_block];
    int* cand_perm = s_cand_perm[warp_in_block];
    int* cand_mode = s_cand_mode[warp_in_block];

    if (iter > 0 && ant_id < top_t) {
        // Elite Intensification Warps: Inherit top-t elite solutions from previous generation
        const int* e_perm = &d_elite_perms[ant_id * n];
        const int* e_mode = &d_elite_modes[ant_id * n];
        for (int k = lane; k < n; k += WARP_SIZE) {
            my_perm[k] = e_perm[k];
            my_mode[k] = e_mode[k];
        }
        __syncwarp();
        // For second half of elite warps, apply a light ACO-guided perturbation to escape local optima
        if (ant_id >= (top_t / 2) && lane == 0) {
            int p1 = curand(&local_rng) % n;
            int p2 = curand(&local_rng) % n;
            int tp = my_perm[p1]; my_perm[p1] = my_perm[p2]; my_perm[p2] = tp;
            int tm = my_mode[p1]; my_mode[p1] = my_mode[p2]; my_mode[p2] = tm;
        }
        __syncwarp();
    } else {
        // ====================================================================
        // PHASE 1: WARP-COOPERATIVE TOUR CONSTRUCTION (3D PHEROMONE + HEURISTIC)
        // ====================================================================
        unsigned int visited_mask[4] = {0U, 0U, 0U, 0U};

        int prev_cust = 0;
        int curr_phys_node = 0;
        int curr_load = 0;
        int curr_time = 0;
        unsigned int visited_lockers_mask = 0U;

        for (int step = 0; step < n; ++step) {
            float best_key = -1.0f;
            int best_j = -1;
            int best_d = 0;

            for (int base = 1; base <= n; base += WARP_SIZE) {
                int j = base + lane;
                if (j <= n) {
                    int w_idx = j >> 5;
                    int b_idx = j & 31;
                    if ((visited_mask[w_idx] & (1U << b_idx)) == 0U) {
                        int c_type = cust_type[j - 1];
                        int d_start = (c_type == 2) ? 1 : 0;
                        int d_end   = (c_type == 1) ? 0 : 1;
                        int demand  = cust_demand[j - 1];

                        for (int d = d_start; d <= d_end; ++d) {
                            int tau_idx = prev_cust * ((n + 1) * 2) + j * 2 + d;
                            float t_val = tau[tau_idx];
                            if (t_val <= 0.0f) continue;

                            int deliv_node = (d == 0) ? j : cust_locker[j - 1];
                            float step_dist = dist[curr_phys_node * total_nodes + deliv_node];
                            int arr_time = curr_time + static_cast<int>(step_dist);
                            int early = node_early[deliv_node];
                            int late  = node_late[deliv_node];

                            bool is_locker = (deliv_node > n);
                            bool dup_locker = false;
                            if (is_locker) {
                                int l_bit = deliv_node - (n + 1);
                                if ((visited_lockers_mask & (1U << l_bit)) != 0U && deliv_node != curr_phys_node) {
                                    dup_locker = true;
                                }
                            }

                            float eta;
                            if (curr_load + demand > cap || arr_time > late || dup_locker) {
                                float d0 = dist[0 * total_nodes + deliv_node];
                                int e0 = node_early[deliv_node];
                                int l0 = node_late[deliv_node];
                                float wait0 = (e0 > static_cast<int>(d0)) ? static_cast<float>(e0 - static_cast<int>(d0)) : 0.0f;
                                float new_veh_metric = d0 + 0.5f * wait0 + 0.12f * static_cast<float>(l0);
                                eta = 1e-4f / (new_veh_metric + 1.0f);
                            } else {
                                if (is_locker && deliv_node == curr_phys_node) {
                                    eta = 50.0f;
                                } else {
                                    int wait_time = (early > arr_time) ? (early - arr_time) : 0;
                                    int urgency = late - ((arr_time > early) ? arr_time : early);
                                    float metric = step_dist + 0.7f * static_cast<float>(wait_time) + 0.12f * static_cast<float>(urgency);
                                    if (is_locker) {
                                        metric *= 0.65f;
                                    }
                                    eta = 1.0f / (metric + 1.0f);
                                }
                            }

                            float weight = __powf(t_val, alpha) * __powf(eta, beta);
                            float u = curand_uniform(&local_rng);
                            u = fmaxf(u, 1e-7f);
                            float key = weight / (-__logf(u));

                            if (key > best_key) {
                                best_key = key;
                                best_j = j;
                                best_d = d;
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                float other_key = __shfl_xor_sync(0xFFFFFFFF, best_key, offset);
                int other_j     = __shfl_xor_sync(0xFFFFFFFF, best_j, offset);
                int other_d     = __shfl_xor_sync(0xFFFFFFFF, best_d, offset);
                if (other_key > best_key || (other_key == best_key && other_j > 0 && (best_j < 0 || other_j < best_j))) {
                    best_key = other_key;
                    best_j = other_j;
                    best_d = other_d;
                }
            }

            if (best_j <= 0) break;

            if (lane == 0) {
                my_perm[step] = best_j;
                my_mode[step] = best_d;
            }
            visited_mask[best_j >> 5] |= (1U << (best_j & 31));

            int deliv_node = (best_d == 0) ? best_j : cust_locker[best_j - 1];
            int demand = cust_demand[best_j - 1];
            int arr_time = curr_time + static_cast<int>(dist[curr_phys_node * total_nodes + deliv_node]);
            int early = node_early[deliv_node];
            int late  = node_late[deliv_node];
            bool is_locker = (deliv_node > n);
            bool dup_locker = false;
            if (is_locker) {
                int l_bit = deliv_node - (n + 1);
                if ((visited_lockers_mask & (1U << l_bit)) != 0U && deliv_node != curr_phys_node) {
                    dup_locker = true;
                }
            }

            if (curr_load + demand > cap || arr_time > late || dup_locker) {
                curr_phys_node = 0;
                curr_load = 0;
                curr_time = 0;
                visited_lockers_mask = 0U;
                arr_time = static_cast<int>(dist[0 * total_nodes + deliv_node]);
            }
            curr_time = (arr_time > early) ? arr_time : early;
            curr_load += demand;
            curr_phys_node = deliv_node;
            if (is_locker) {
                visited_lockers_mask |= (1U << (deliv_node - (n + 1)));
            }
            prev_cust = best_j;
        }
        __syncwarp();
    }

    float current_cost = evaluate_solution_device(
        n, total_nodes, cap, my_perm, my_mode,
        cust_demand, cust_locker, node_early, node_late, dist
    );

    // ========================================================================
    // PHASE 2: WARP-PARALLEL LOCAL SEARCH (32 lanes evaluate 32 moves/step!)
    // ========================================================================
    int local_perm[MAX_CUST];
    int local_mode[MAX_CUST];

    for (int ls = 0; ls < ls_steps; ++ls) {
        for (int k = 0; k < n; ++k) {
            local_perm[k] = my_perm[k];
            local_mode[k] = my_mode[k];
        }

        int i1 = curand(&local_rng) % n;
        int i2 = curand(&local_rng) % n;
        int op = lane & 3; // 0: 2-opt, 1: Swap, 2: Or-opt Block Relocate, 3: Locker Merge & Mode Flip

        if (i1 != i2) {
            if (op == 0) {
                if (i1 > i2) { int tmp = i1; i1 = i2; i2 = tmp; }
                int l = i1, r = i2;
                while (l < r) {
                    int tp = local_perm[l]; local_perm[l] = local_perm[r]; local_perm[r] = tp;
                    int tm = local_mode[l]; local_mode[l] = local_mode[r]; local_mode[r] = tm;
                    l++; r--;
                }
            } else if (op == 1) {
                int tp = local_perm[i1]; local_perm[i1] = local_perm[i2]; local_perm[i2] = tp;
                int tm = local_mode[i1]; local_mode[i1] = local_mode[i2]; local_mode[i2] = tm;
            } else if (op == 2) {
                int blk_len = 1 + (curand(&local_rng) % 4);
                if (i1 + blk_len <= n) {
                    int tmp_p[4], tmp_m[4];
                    for (int b = 0; b < blk_len; ++b) {
                        tmp_p[b] = local_perm[i1 + b];
                        tmp_m[b] = local_mode[i1 + b];
                    }
                    for (int k = i1; k + blk_len < n; ++k) {
                        local_perm[k] = local_perm[k + blk_len];
                        local_mode[k] = local_mode[k + blk_len];
                    }
                    int rem_n = n - blk_len;
                    int ins_pos = i2 % (rem_n + 1);
                    for (int k = rem_n - 1; k >= ins_pos; --k) {
                        local_perm[k + blk_len] = local_perm[k];
                        local_mode[k + blk_len] = local_mode[k];
                    }
                    for (int b = 0; b < blk_len; ++b) {
                        local_perm[ins_pos + b] = tmp_p[b];
                        local_mode[ins_pos + b] = tmp_m[b];
                    }
                }
            } else {
                int c1 = local_perm[i1];
                if (cust_type[c1 - 1] == 3) {
                    local_mode[i1] = 1 - local_mode[i1];
                }
                int dnode1 = (local_mode[i1] == 0) ? c1 : cust_locker[c1 - 1];

                if (dnode1 > n) {
                    for (int scan = 0; scan < n; ++scan) {
                        int idx = (i2 + scan) % n;
                        if (idx == i1 || idx == i1 + 1) continue;
                        int c2 = local_perm[idx];
                        if (cust_type[c2 - 1] >= 2 && cust_locker[c2 - 1] == dnode1) {
                            i2 = idx;
                            local_mode[i2] = 1;
                            break;
                        }
                    }
                }
                if (i1 != i2) {
                    int vp = local_perm[i2];
                    int vm = local_mode[i2];
                    int target = (i1 + 1 < n) ? (i1 + 1) : i1;
                    if (i2 < target) {
                        for (int k = i2; k < target; ++k) {
                            local_perm[k] = local_perm[k + 1];
                            local_mode[k] = local_mode[k + 1];
                        }
                    } else if (i2 > target) {
                        for (int k = i2; k > target; --k) {
                            local_perm[k] = local_perm[k - 1];
                            local_mode[k] = local_mode[k - 1];
                        }
                    }
                    local_perm[target] = vp;
                    local_mode[target] = vm;
                }
            }
        }

        float cand_cost = evaluate_solution_device(
            n, total_nodes, cap, local_perm, local_mode,
            cust_demand, cust_locker, node_early, node_late, dist
        );

        float best_lane_cost = cand_cost;
        int best_lane = lane;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFF, best_lane_cost, offset);
            int other_lane   = __shfl_xor_sync(0xFFFFFFFF, best_lane, offset);
            if (other_cost < best_lane_cost) {
                best_lane_cost = other_cost;
                best_lane = other_lane;
            }
        }

        if (best_lane_cost < current_cost) {
            current_cost = best_lane_cost;
            if (lane == best_lane) {
                for (int k = 0; k < n; ++k) {
                    cand_perm[k] = local_perm[k];
                    cand_mode[k] = local_mode[k];
                }
            }
            __syncwarp();
            for (int k = lane; k < n; k += WARP_SIZE) {
                my_perm[k] = cand_perm[k];
                my_mode[k] = cand_mode[k];
            }
            __syncwarp();
        }
    }

    int* out_perm = &d_all_perms[ant_id * n];
    int* out_mode = &d_all_modes[ant_id * n];
    for (int k = lane; k < n; k += WARP_SIZE) {
        out_perm[k] = my_perm[k];
        out_mode[k] = my_mode[k];
    }
    if (lane == 0) {
        d_all_costs[ant_id] = current_cost;
    }

    rng_states[global_tid] = local_rng;
}

__global__ void copy_elite_solutions_kernel(
    int n,
    int top_t,
    const int* __restrict__ d_elite_indices,
    const int* __restrict__ d_all_perms,
    const int* __restrict__ d_all_modes,
    int* __restrict__ d_elite_perms,
    int* __restrict__ d_elite_modes
) {
    int rank = blockIdx.x;
    int k = threadIdx.x;
    if (rank < top_t && k < n) {
        int ant_id = d_elite_indices[rank];
        d_elite_perms[rank * n + k] = d_all_perms[ant_id * n + k];
        d_elite_modes[rank * n + k] = d_all_modes[ant_id * n + k];
    }
}

__global__ void evaporate_and_clamp_kernel(float* d_tau, const int* d_mask, int total_elements, float rho, float tau_min, float tau_max) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_elements) {
        if (d_mask[idx] == 0) {
            d_tau[idx] = 0.0f;
        } else {
            float val = d_tau[idx] * (1.0f - rho);
            if (val < tau_min) val = tau_min;
            if (val > tau_max) val = tau_max;
            d_tau[idx] = val;
        }
    }
}

__global__ void elitist_deposit_kernel(
    int n,
    int top_t,
    float Q,
    const int* __restrict__ d_elite_indices,
    const int* __restrict__ d_all_perms,
    const int* __restrict__ d_all_modes,
    const float* __restrict__ d_all_costs,
    const int* __restrict__ d_mask,
    float* __restrict__ d_tau
) {
    int rank = blockIdx.x;
    int step = threadIdx.x;
    if (rank >= top_t || step > n) return;

    int ant_id = d_elite_indices[rank];
    float obj = d_all_costs[ant_id];
    if (obj <= 0.0f) return;

    float delta = Q / obj;
    const int* perm = &d_all_perms[ant_id * n];
    const int* mode = &d_all_modes[ant_id * n];

    int u, v, d;
    if (step == 0) {
        u = 0;
        v = perm[0];
        d = mode[0];
    } else if (step < n) {
        u = perm[step - 1];
        v = perm[step];
        d = mode[step];
    } else {
        u = perm[n - 1];
        v = 0;
        d = 0;
    }

    int tau_idx = u * ((n + 1) * 2) + v * 2 + d;
    if (d_mask[tau_idx]) {
        atomicAdd(&d_tau[tau_idx], delta);
    }
}

struct SolveStats {
    float best_dist;
    int best_vehicles;
    double runtime_sec;
};

SolveStats run_gpu_3d_paco_single(
    const InstanceHost& inst,
    int num_ants,
    int max_non_improved,
    int top_t,
    int ls_steps,
    float alpha,
    float beta,
    float rho_ini,
    float Q,
    unsigned long long seed
) {
    int n = inst.num_customers;
    int total_nodes = inst.total_nodes;
    int tau_elements = (n + 1) * (n + 1) * 2;

    std::vector<int> h_mask(tau_elements, 1);
    std::vector<float> h_tau(tau_elements, 1.0f);
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            h_mask[i * ((n + 1) * 2) + j * 2 + 0] = (i != j) ? 1 : 0;
            h_mask[i * ((n + 1) * 2) + j * 2 + 1] = (i != j) ? 1 : 0;
            if (j >= 1 && j <= n) {
                int ctype = inst.cust_type[j - 1];
                if (ctype == 2) h_mask[i * ((n + 1) * 2) + j * 2 + 0] = 0;
                if (ctype == 1) h_mask[i * ((n + 1) * 2) + j * 2 + 1] = 0;
            }
            for (int o = 0; o < 2; ++o) {
                int idx = i * ((n + 1) * 2) + j * 2 + o;
                if (!h_mask[idx]) h_tau[idx] = 0.0f;
            }
        }
    }

    int *d_cust_type, *d_cust_demand, *d_cust_locker, *d_node_early, *d_node_late, *d_mask;
    float *d_dist, *d_tau, *d_all_costs;
    int *d_all_perms, *d_all_modes, *d_elite_indices, *d_elite_perms, *d_elite_modes;
    curandState* d_rng_states;

    int total_threads = num_ants * WARP_SIZE;

    cudaMalloc(&d_cust_type, n * sizeof(int));
    cudaMalloc(&d_cust_demand, n * sizeof(int));
    cudaMalloc(&d_cust_locker, n * sizeof(int));
    cudaMalloc(&d_node_early, total_nodes * sizeof(int));
    cudaMalloc(&d_node_late, total_nodes * sizeof(int));
    cudaMalloc(&d_dist, total_nodes * total_nodes * sizeof(float));
    cudaMalloc(&d_mask, tau_elements * sizeof(int));
    cudaMalloc(&d_tau, tau_elements * sizeof(float));

    cudaMalloc(&d_all_perms, num_ants * n * sizeof(int));
    cudaMalloc(&d_all_modes, num_ants * n * sizeof(int));
    cudaMalloc(&d_all_costs, num_ants * sizeof(float));
    cudaMalloc(&d_elite_indices, top_t * sizeof(int));
    cudaMalloc(&d_elite_perms, top_t * n * sizeof(int));
    cudaMalloc(&d_elite_modes, top_t * n * sizeof(int));
    cudaMalloc(&d_rng_states, total_threads * sizeof(curandState));

    cudaMemcpy(d_cust_type, inst.cust_type.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cust_demand, inst.cust_demand.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cust_locker, inst.cust_locker.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_early, inst.node_early.data(), total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_late, inst.node_late.data(), total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, inst.dist_matrix.data(), total_nodes * total_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), tau_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau, h_tau.data(), tau_elements * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 128; // 4 Warps (4 Ants) per Block
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    init_rng_kernel<<<blocks, threads_per_block>>>(d_rng_states, seed, total_threads);

    float global_best_cost = 1e30f;
    int global_best_ant = 0;
    std::vector<int> global_best_perm(n), global_best_mode(n);

    std::vector<float> h_costs(num_ants);
    std::vector<int> indices(num_ants);

    int non_improved = 0;
    int iter = 0;
    float rho = rho_ini;

    while (non_improved < max_non_improved) {
        warp_paco_construct_and_ls_kernel<<<blocks, threads_per_block>>>(
            iter, n, total_nodes, inst.vehicle_capacity, top_t,
            d_cust_type, d_cust_demand, d_cust_locker,
            d_node_early, d_node_late, d_dist, d_tau,
            alpha, beta, ls_steps, num_ants, d_rng_states,
            d_elite_perms, d_elite_modes,
            d_all_perms, d_all_modes, d_all_costs
        );

        cudaMemcpy(h_costs.data(), d_all_costs, num_ants * sizeof(float), cudaMemcpyDeviceToHost);

        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_t, indices.end(),
                          [&](int a, int b) { return h_costs[a] < h_costs[b]; });

        int iter_best_ant = indices[0];
        float iter_best_cost = h_costs[iter_best_ant];

        if (iter_best_cost + 1e-4f < global_best_cost) {
            global_best_cost = iter_best_cost;
            global_best_ant = iter_best_ant;
            cudaMemcpy(global_best_perm.data(), &d_all_perms[global_best_ant * n], n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(global_best_mode.data(), &d_all_modes[global_best_ant * n], n * sizeof(int), cudaMemcpyDeviceToHost);
            non_improved = 0;
        } else {
            non_improved++;
            float x = 10.0f * (static_cast<float>(non_improved) / max_non_improved - 0.5f);
            float sig = 1.0f / (1.0f + std::exp(-x));
            rho = rho_ini + (0.7f - rho_ini) * sig;
        }

        cudaMemcpy(d_elite_indices, indices.data(), top_t * sizeof(int), cudaMemcpyHostToDevice);

        copy_elite_solutions_kernel<<<top_t, n>>>(
            n, top_t, d_elite_indices, d_all_perms, d_all_modes, d_elite_perms, d_elite_modes
        );

        int tau_blocks = (tau_elements + 255) / 256;
        evaporate_and_clamp_kernel<<<tau_blocks, 256>>>(d_tau, d_mask, tau_elements, rho, 0.01f, 20.0f);

        elitist_deposit_kernel<<<top_t, n + 1>>>(
            n, top_t, Q, d_elite_indices, d_all_perms, d_all_modes, d_all_costs, d_mask, d_tau
        );
        iter++;
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);

    // Host verification of best solution
    float verified_dist = 0.0f;
    int verified_veh = 0;
    evaluate_solution_host(inst, global_best_perm, global_best_mode, verified_dist, verified_veh);

    cudaFree(d_cust_type);
    cudaFree(d_cust_demand);
    cudaFree(d_cust_locker);
    cudaFree(d_node_early);
    cudaFree(d_node_late);
    cudaFree(d_dist);
    cudaFree(d_mask);
    cudaFree(d_tau);
    cudaFree(d_all_perms);
    cudaFree(d_all_modes);
    cudaFree(d_all_costs);
    cudaFree(d_elite_indices);
    cudaFree(d_elite_perms);
    cudaFree(d_elite_modes);
    cudaFree(d_rng_states);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return {verified_dist, verified_veh, elapsed_ms / 1000.0};
}

void run_dataset_benchmark(
    const std::string& data_dir,
    const std::string& out_csv_path,
    int num_runs,
    int num_ants,
    int max_non_improved,
    int top_t,
    int ls_steps
) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".txt") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());

    std::ofstream csv(out_csv_path);
    csv << "instance_name,Num Vehicles,Best Distance,AVG Distance,Std Distance,AVG Runtime (s),Std Runtime (s)\n";

    double sum_best_dist = 0.0;
    double sum_avg_dist = 0.0;
    double sum_runtime = 0.0;
    int count = 0;

    std::cout << "Running benchmark on " << data_dir << " (" << files.size() << " instances, " << num_runs << " runs each)..." << std::endl;

    for (const auto& fpath : files) {
        InstanceHost inst;
        if (!parse_vrppl_instance(fpath, inst)) continue;

        std::vector<double> dists(num_runs);
        std::vector<double> times(num_runs);
        int best_veh = 0;
        double best_dist = 1e30;

        for (int r = 0; r < num_runs; ++r) {
            SolveStats st = run_gpu_3d_paco_single(
                inst, num_ants, max_non_improved, top_t, ls_steps,
                1.25f, 1.5f, 0.3f, 2.0f, 2026ULL + r * 997ULL
            );
            dists[r] = st.best_dist;
            times[r] = st.runtime_sec;
            if (st.best_dist < best_dist) {
                best_dist = st.best_dist;
                best_veh = st.best_vehicles;
            }
        }

        double avg_dist = std::accumulate(dists.begin(), dists.end(), 0.0) / num_runs;
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / num_runs;

        double sq_dist = 0.0, sq_time = 0.0;
        for (int r = 0; r < num_runs; ++r) {
            sq_dist += (dists[r] - avg_dist) * (dists[r] - avg_dist);
            sq_time += (times[r] - avg_time) * (times[r] - avg_time);
        }
        double std_dist = std::sqrt(sq_dist / num_runs);
        double std_time = std::sqrt(sq_time / num_runs);

        csv << inst.name << "," << best_veh << ","
            << best_dist << "," << avg_dist << "," << std_dist << ","
            << avg_time << "," << std_time << "\n";

        sum_best_dist += best_dist;
        sum_avg_dist += avg_dist;
        sum_runtime += avg_time;
        count++;
    }
    csv.close();

    std::cout << "  -> Completed " << count << " instances in " << data_dir << "!\n"
              << "  -> Mean Best Distance : " << std::fixed << std::setprecision(2) << (sum_best_dist / count) << "\n"
              << "  -> Mean AVG Distance  : " << (sum_avg_dist / count) << "\n"
              << "  -> Mean Runtime (s)   : " << std::setprecision(4) << (sum_runtime / count) << " s\n"
              << "  -> Saved to           : " << out_csv_path << "\n" << std::endl;
}

int main(int argc, char** argv) {
    // Warm up GPU context
    cudaFree(0);

    std::string mode = (argc > 1) ? argv[1] : "all";
    if (mode == "single") {
        InstanceHost inst;
        parse_vrppl_instance("src/data/100/C101_co_100.txt", inst);
        auto st = run_gpu_3d_paco_single(inst, 512, 18, 64, 75, 1.25f, 1.5f, 0.3f, 2.0f, 42ULL);
        std::cout << "C101_co_100.txt -> Best Distance: " << st.best_dist
                  << " | Vehicles: " << st.best_vehicles
                  << " | Runtime: " << st.runtime_sec << " s" << std::endl;
        return 0;
    }

    if (mode == "25" || mode == "small") {
        run_dataset_benchmark("src/data/25",  "src/output/solutions/small_gpu_paco.csv",  3, 256, 12, 32, 35);
        return 0;
    }
    if (mode == "50" || mode == "medium") {
        run_dataset_benchmark("src/data/50",  "src/output/solutions/medium_gpu_paco.csv", 3, 384, 14, 48, 45);
        return 0;
    }
    if (mode == "100" || mode == "large") {
        run_dataset_benchmark("src/data/100", "src/output/solutions/large_gpu_paco.csv",  3, 512, 15, 64, 60);
        return 0;
    }

    run_dataset_benchmark("src/data/25",  "src/output/solutions/small_gpu_paco.csv",  3, 256, 12, 32, 35);
    run_dataset_benchmark("src/data/50",  "src/output/solutions/medium_gpu_paco.csv", 3, 384, 14, 48, 45);
    run_dataset_benchmark("src/data/100", "src/output/solutions/large_gpu_paco.csv",  3, 512, 15, 64, 60);

    return 0;
}
