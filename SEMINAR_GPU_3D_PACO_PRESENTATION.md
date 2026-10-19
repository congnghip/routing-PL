# GPU 3D-PACO cho VRPPL: Nền tảng GPU, Biến đổi Tính toán và Luồng Thuật toán

> **Phạm vi:** Tài liệu mô tả đúng cài đặt trong `src/cuda/paco_gpu_solver.cu` (commit `ba9af94`), chế độ mặc định **không** có cờ `--ig`. Mọi hằng số và công thức được đối chiếu với mã nguồn. Các chế độ Lever 1/2/3 được mô tả như các cờ tùy chọn trên cùng một pipeline.
>
> **Phần cứng:** Thông số lấy từ `src/cuda/gpu_info` trên máy chạy mã: NVIDIA GeForce RTX 3060, Compute Capability 8.6, 28 SM, 12 GB VRAM, CUDA toolkit 12.0.

---

## Mục lục

- [Bảng ký hiệu](#bảng-ký-hiệu)
- [I. Nền tảng GPU và kiến trúc GPU](#i-nền-tảng-gpu-và-kiến-trúc-gpu)
- [II. Các biến đổi để tính toán trên GPU](#ii-các-biến-đổi-để-tính-toán-trên-gpu)
- [III. Luồng thuật toán tổng thể (mã giả)](#iii-luồng-thuật-toán-tổng-thể-mã-giả)

---

## Bảng ký hiệu

### Bài toán (VRPPL)

| Ký hiệu | Tên | Ý nghĩa | Trong mã (`paco_gpu_solver.cu`) |
| :--- | :--- | :--- | :--- |
| $n$ | Số khách hàng | Khách đánh số $1, \dots, n$ | `n`, `inst.num_customers` |
| $m$ | Số tủ locker | Tủ đánh số $n+1, \dots, n+m$ | `num_lockers` |
| $0$ | Kho trung tâm | Nút xuất phát và kết thúc của mọi xe | nút `0` |
| $V_{C1}, V_{C2}, V_{C3}$ | Loại khách hàng | Home-only (1), Locker-only (2), Flex (3) | `cust_type` |
| $d_{uv}$ | Khoảng cách | Cạnh $u \to v$, làm tròn xuống bội số 0.1 | `dist[u*total_nodes+v]` (float) |
| $e_v, l_v$ | Cửa sổ thời gian | Thời điểm mở sớm và đóng muộn tại nút $v$ | `node_early`, `node_late` |
| $s_v$ | Thời gian phục vụ | Thời gian dừng tại nút $v$ | `node_service` |
| $q_i$ | Nhu cầu | Khối lượng hàng của khách $i$ | `cust_demand` |
| $C$ | Tải trọng xe | Sức chứa tối đa mỗi xe | `vehicle_capacity` (`cap`) |
| $L(i)$ | Tủ gán cho $i$ | Nút locker mà khách $i$ được gán | `cust_locker` |
| $\text{Node}(j,o)$ | Nút vật lý giao hàng | $j$ nếu $o=0$; $L(j)$ nếu $o=1$ | `deliv_node` |

### Lời giải và hàm mục tiêu

| Ký hiệu | Tên | Ý nghĩa | Trong mã |
| :--- | :--- | :--- | :--- |
| $\pi = (\pi_1, \dots, \pi_n)$ | Giant tour | Hoán vị phẳng của $n$ khách hàng, không chứa kho | `perm[n]` |
| $o = (o_1, \dots, o_n)$ | Vector phương thức | $o_k = 0$: giao tại nhà; $o_k = 1$: gửi locker | `mode[n]` |
| $\mathcal{F}(\pi, o)$ | Hàm chi phí | Tổng quãng đường sau khi tách tuyến (không có chi phí cố định xe) | `evaluate_solution_device` |
| $\mathcal{F}^*$ | Chi phí tốt nhất toàn cục | | `global_best_cost` |
| $S^*$ | Lời giải tốt nhất toàn cục | Cặp $(\pi^*, o^*)$ | `global_best_perm`, `global_best_mode` |

### Mùi, heuristic và xác suất chọn

| Ký hiệu | Tên | Ý nghĩa | Trong mã |
| :--- | :--- | :--- | :--- |
| $\tau_{ijo}$ | Mùi (pheromone) | Mức hấp dẫn của bước "từ khách $i$ sang khách $j$ với phương thức $o$" | `tau[i*(2(n+1))+2j+o]` |
| $M_{ijo}$ | Mặt nạ khả thi | $M_{ijo}=0$ nếu bước bị cấm theo loại khách | `h_mask` |
| $\eta_{ijo}$ | Heuristic | Thông tin tức thời của bước $(i \to j, o)$ | hàm `ETA` |
| $\alpha$ | Trọng số mùi | Mũ của $\tau$ | `alpha` = 1.25 |
| $\beta$ | Trọng số heuristic | Mũ của $\eta$ | `beta` = 1.5 |
| $w_{jo}$ | Trọng số ứng viên | $\tau_{ijo}^{\alpha}\,\eta_{ijo}^{\beta}$ | biến `weight` |
| $U$ | Biến đều | $U \sim \text{Uniform}(0,1]$ | `curand_uniform` |
| $E$ | Biến mũ | $E = -\ln U \sim \text{Exp}(1)$ | `-__logf(u)` |
| $K_{jo}$ | Khóa Gumbel-race | $K_{jo} = w_{jo}/E_{jo}$ | biến `key` |
| $\kappa_{merge}$ | Thưởng gom locker | Gán $\eta = 50$ khi ghé lại đúng tủ đang đứng | literal `50.0f` |

### Mùi bay hơi, cường độ và giới hạn

| Ký hiệu | Tên | Ý nghĩa | Giá trị |
| :--- | :--- | :--- | :--- |
| $\rho$ | Hệ số bay hơi | Biến thiên theo số vòng không cải thiện | $\rho_{ini}=0.3$, $\rho_{max}=0.7$ (hằng số trong mã) |
| $\sigma(x)$ | Hàm sigmoid | $1/(1+e^{-x})$ | |
| $Q$ | Cường độ lắng đọng | Tử số của $\Delta\tau = Q/\mathcal{F}$ | `Q` = 2.0 |
| $\tau_{min}, \tau_{max}$ | Ngưỡng kẹp | Giới hạn mùi sau mỗi bước bay hơi | 0.01, 20.0 |

### Tham số thuật toán theo quy mô

| Ký hiệu | Tên | Ý nghĩa | $n=25$ | $n=50$ | $n=100$ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| $N_{ants}$ | Số kiến | Số warp = số kiến, mỗi kiến 32 luồng | 256 | 384 | 512 |
| $T_{elite}$ | Số kiến tinh hoa | Số kiến top-$t$ được lắng đọng mùi | 32 | 48 | 64 |
| $I_{max}$ | Ngưỡng dừng | Số vòng liên tiếp không cải thiện tối đa | 12 | 14 | 15 |
| $S_{ls}$ | Số bước LS | Số bước tìm kiếm cục bộ mỗi kiến (chế độ cơ bản) | 35 | 45 | 60 |
| $q_r$ | Kích thước phá hủy | Số khách bị gỡ trong Ruin (Lever 2) | 4 | 6 | 10 |

Ghi chú: Lever 1 thay $S_{ls}$ bằng $70 / 90 / 140$ và cộng thêm $+4 / +4 / +7$ vào $I_{max}$ (theo $n \ge 100$ hay không). Số vòng LS thực tế khi bật Lever 1 là $S_{ls} + \lfloor S_{ls}/2 \rfloor$, tức $105 / 135 / 210$.

---

## I. Nền tảng GPU và kiến trúc GPU

### 1. CPU và GPU khác nhau ở đâu

| Tiêu chí | CPU | GPU |
| :--- | :--- | :--- |
| Số lõi | Vài đến vài chục lõi mạnh | Hàng nghìn lõi đơn giản, gom thành SM |
| Mục tiêu tối ưu | Độ trễ thấp cho một luồng | Thông lượng lớn cho hàng nghìn luồng |
| Cách che độ trễ | Cache lớn, dự đoán nhánh, out-of-order | Chuyển đổi nhanh giữa các warp đang chờ |
| Mô hình thực thi | Đa luồng độc lập (MIMD) | Đơn lệnh đa luồng (SIMT) |
| Bộ nhớ chính | RAM hệ thống | VRAM (global memory), băng thông cao |

Bài toán ACO có rất nhiều đánh giá độc lập (mỗi kiến, mỗi ứng viên, mỗi cạnh mùi), nên phù hợp với mô hình GPU hơn.

### 2. Thông số phần cứng đo được (RTX 3060)

| Thông số | Giá trị | Ý nghĩa cho thiết kế |
| :--- | :--- | :--- |
| Compute Capability | 8.6 | Kiến trúc Ampere, hỗ trợ `__shfl_xor_sync`, `atomicAdd` float |
| Số SM | 28 | Mỗi SM chạy đồng thời nhiều block |
| Max threads / SM | 1536 | Tối đa 48 warp đang thường trú trên một SM |
| Max threads / block | 1024 | Mã dùng 128 luồng = 4 warp mỗi block |
| Shared memory / block | 48 KB | Mã dùng khoảng vài KB cho mảng tour tạm |
| Băng thông VRAM | $\approx 7501\,\text{MHz} \times 2 \times \tfrac{192}{8} \approx 360\,\text{GB/s}$ | Giới hạn khi đọc/ghi mảng lớn |
| Warp size | 32 | Đơn vị cơ sở của SIMT, cũng là số lane trong "một kiến" |

### 3. Phân cấp phần cứng: SM và Warp

```
GPU
 ├── SM 0 ──┬── Warp scheduler (chọn warp sẵn sàng để phát lệnh)
 │          ├── Register file (65 536 thanh ghi 32-bit mỗi SM)
 │          ├── Shared memory / L1 (dùng chung trong SM)
 │          └── Các lõi CUDA (thực thi lệnh)
 ├── SM 1 ...
 └── ...
          │
          ▼
    VRAM (global memory) ── L2 cache ── PCIe ── RAM của máy chủ (host)
```

- **Warp** là 32 luồng cùng chạy một lệnh tại một thời điểm (SIMT). Nếu các luồng trong warp rẽ nhánh khác nhau, phần cứng chạy tuần tự từng nhánh và tắt các luồng không thuộc nhánh đang chạy. Đây là hiện tượng **warp divergence**.
- **Block** là nhóm luồng nằm trọn trên một SM, có chung shared memory và có thể đồng bộ bằng `__syncthreads()`.
- **Grid** là tập hợp các block của một lần gọi kernel. Mã đặt `<<<N_ants/4, 128>>>` nghĩa là 4 warp (4 kiến) trên mỗi block.

### 4. Phân cấp bộ nhớ

| Bộ nhớ | Phạm vi | Vị trí | Độ trễ (bậc độ lớn) | Ghi chú trong mã |
| :--- | :--- | :--- | :--- | :--- |
| Register | Mỗi luồng | Trên chip | ~1 chu kỳ | `visited_mask[4]`, `best_key`, `curr_load`, `curr_time` |
| Local memory | Mỗi luồng | VRAM (cache qua L1) | Hàng trăm chu kỳ nếu cache trượt | `local_perm[MAX_CUST]`, `local_mode[MAX_CUST]` (mỗi mảng 512 B) |
| Shared memory | Mỗi block | Trên chip | Vài chục chu kỳ | `s_perm`, `s_mode`, `s_cand_perm`, `s_ruin_*` |
| L1 / L2 cache | Dùng chung | Trên chip | Chục đến trăm chu kỳ | Tự động |
| Global memory (VRAM) | Toàn GPU | Ngoài chip | Hàng trăm chu kỳ | `d_dist`, `d_tau`, `d_all_perms` |
| Host memory (RAM) | CPU | Qua PCIe | Rất chậm so với VRAM | Chỉ dùng cho điều khiển và kết quả cuối |

Các con số độ trễ chỉ mang tính bậc độ lớn để so sánh.

### 5. Các khái niệm hiệu năng dùng trong mã

- **Coalescing (gộp truy cập bộ nhớ):** Khi 32 luồng của warp đọc các địa chỉ liên tiếp, phần cứng gom thành một giao dịch. Vì vậy, tensor $\tau$ được làm phẳng thành mảng 1D với chỉ số tuyến tính (mục II.2).
- **Occupancy:** Tỉ lệ warp đang thường trú so với tối đa. Occupancy cao giúp che độ trễ vì khi một warp chờ bộ nhớ, warp khác được chạy.
- **Warp shuffle (`__shfl_xor_sync`):** Hai lane trao đổi trực tiếp giá trị trong thanh ghi, không dùng shared memory và không cần `__syncthreads()`. Mã dùng bước xor-butterfly với offset $16, 8, 4, 2, 1$ để rút gọn 32 giá trị thành 1 (mục II.4).
- **Atomic (`atomicAdd`):** Cộng nguyên tử vào global memory. Nhiều luồng cùng ghi một địa chỉ sẽ bị tuần tự hóa. Mã dùng atomic khi lắng đọng mùi (mục II.8).
- **`__syncwarp()`:** Đồng bộ các lane trong một warp, rẻ hơn `__syncthreads()`.
- **cuRAND device API:** Mỗi luồng có trạng thái RNG riêng nằm trong VRAM và sinh số ngẫu nhiên ngay trên GPU. Mã dùng kiểu `curandState` (bộ sinh XORWOW mặc định), khởi tạo bằng `curand_init(seed, tid, 0, ...)`.
- **Host–device:** `cudaMalloc` cấp phát VRAM, `cudaMemcpy` truyền dữ liệu qua PCIe. Mọi bộ đệm được cấp phát một lần đầu mỗi lần giải, nên truyền dữ liệu chỉ diễn ra ở đầu và cuối, cùng vài thông tin điều khiển nhỏ trong vòng lặp (mục II.10).

---

## II. Các biến đổi để tính toán trên GPU

ACO chuẩn là một vòng lặp tuần tự: kiến dựng lộ trình từng bước, mỗi bước tính xác suất, chọn bước, cập nhật trạng thái, rồi bay hơi và lắng đọng mùi. Để chạy trên GPU, mỗi bước phải được viết lại thành dạng **song song theo ứng viên** và **không cần đồng bộ toàn cục**. Dưới đây là 10 biến đổi, mỗi biến đổi gắn với một đoạn mã cụ thể.

### Bảng tổng hợp: thành phần nào chạy ở đâu

| Thành phần | ACO tuần tự trên CPU | Phiên bản GPU 3D-PACO | Nơi thực thi |
| :--- | :--- | :--- | :--- |
| Lưu lời giải | Mảng động, con trỏ | Hai mảng phẳng `perm[n]`, `mode[n]` cho mỗi kiến | GPU (VRAM) |
| Mùi | Ma trận 2D hoặc dict | Mảng 1D `tau`, chỉ số tuyến tính | GPU (VRAM) |
| Roulette wheel | Tổng, tiền tố, tìm nhị phân | Khóa Gumbel-race và rút gọn argmax | GPU (thanh ghi) |
| Chọn bước | Vòng lặp tuần tự | 32 lane chia ứng viên, rút gọn bằng shuffle | GPU (warp) |
| Tìm kiếm cục bộ | Vòng lặp 2-opt, Or-opt | 32 nước đi thử song song mỗi bước | GPU (warp) |
| Bay hơi | Duyệt tuần tự | Kernel từng phần tử | GPU |
| Lắng đọng | Cộng tuần tự | `atomicAdd` | GPU |
| Sắp xếp top-$t$ | `std::sort` | `std::partial_sort` trên $N_{ants}$ số | **CPU** (nhẹ) |
| Biến điều khiển $I_{st}$, $\rho$ | Biến vô hướng | Biến vô hướng | **CPU** (nhẹ) |
| Kiểm tra cuối | Hàm đánh giá | `evaluate_solution_host` (một lần) | **CPU** (một lần) |

---

### T1. Biểu diễn lời giải phẳng và bộ tách tuyến (giant tour)

**Vấn đề.** Lời giải VRPPL gồm nhiều tuyến xe, mỗi tuyến là một danh sách có độ dài thay đổi. Cấu trúc đó khó chia cho các luồng và khó nằm gọn trong bộ nhớ đồng bộ.

**Biến đổi.** Lưu lời giải dưới dạng một **giant tour** phẳng $(\pi, o)$ gồm $n$ phần tử cố định. Việc chia thành tuyến được thực hiện bởi một bộ giải mã tham lam, chạy trong lúc đánh giá. Bộ giải mã mở xe mới khi vi phạm một trong ba ràng buộc: tải trọng, cửa sổ thời gian, hoặc ghé lại locker đã ghé.

$$
\text{viol}(k) =
\mathbf{1}\!\left[L + q_{\pi_k} > C\right]
\;\lor\;
\mathbf{1}\!\left[T + d_{u v} > l_v + 10^{-3}\right]
\;\lor\;
\mathbf{1}\!\left[v \in V_L \,\land\, v \in \text{Seen} \,\land\, v \neq u_{last}\right]
\tag{II.1}
$$

Trong đó $u$ là nút hiện tại, $v = \text{Node}(\pi_k, o_k)$, $T$ là thời điểm hiện tại, $L$ là tải hiện tại, $\text{Seen}$ là tập locker đã ghé trên tuyến hiện tại.

Khi xảy ra vi phạm, xe quay về kho, bắt đầu tuyến mới tại $t = 0$, và bộ giải mã thử lại đúng khách $\pi_k$ từ kho. Thời gian phục vụ được tính một lần cho mỗi lần dừng:

$$
T_v \leftarrow \max\!\left(T_u + d_{uv},\; e_v\right) + s_v \cdot \mathbf{1}\!\left[\neg(v \in V_L \land v = u)\right]
\tag{II.2}
$$

**Ý nghĩa cho GPU.** Bộ giải mã là một hàm tuần tự theo $n$, nhưng nó không cần bộ nhớ động và chỉ đọc mảng phẳng. Mỗi lane có thể chạy một bản sao độc lập, nên nhiều lời giải được đánh giá cùng lúc.

**Giới hạn đã biết.** Mã không kiểm tra giới hạn số xe $K$ và không cộng chi phí cố định xe $F_{fix}$. Số xe chỉ là kết quả phụ của bộ giải mã. Mã cũng giả định mọi khách hàng đều khả thi khi đi riêng từ kho. Nếu giả định này sai, vòng lặp thử lại sẽ không bao giờ kết thúc.

---

### T2. Mùi 3 chiều được phẳng hóa, kèm mặt nạ ràng buộc

**Vấn đề.** Mùi cần biểu diễn cạnh $(i \to j)$ kèm phương thức giao hàng $o$ của khách $j$. Nếu lưu dạng đối tượng hoặc ma trận lồng nhau, truy cập sẽ không gộp được.

**Biến đổi.** Định nghĩa tensor mùi

$$
\tau \in \mathbb{R}^{(n+1) \times (n+1) \times 2}, \qquad \tau_{ijo}: \text{mùi của bước đến khách } j \text{ với phương thức } o
$$

và lưu phẳng thành mảng 1D:

$$
\text{idx}(i, j, o) = i \cdot 2(n+1) + 2j + o
\tag{II.3}
$$

Mặt nạ khả thi theo loại khách (Home-only không được gửi locker, Locker-only không được giao tại nhà):

$$
M_{ijo} = \mathbf{1}[i \neq j] \cdot \mathbf{1}\big[\neg(j \in V_{C2} \land o = 0)\big] \cdot \mathbf{1}\big[\neg(j \in V_{C1} \land o = 1)\big]
\tag{II.4}
$$

Khởi tạo: $\tau_{ijo}^{(0)} = M_{ijo}$. Mọi phần tử bị cấm đều bằng 0 và không bao giờ có trọng số dương.

**Ý nghĩa cho GPU.** Chỉ số tuyến tính làm các luồng liên tiếp đọc các phần tử liên tiếp của $\tau$ (gộp bộ nhớ). Ràng buộc loại khách được mã hóa thành số 0 trong mảng, nên kernel không cần rẽ nhánh theo loại khách khi chọn ứng viên. Mã chỉ kiểm tra `t_val <= 0` để bỏ qua.

**Lưu ý.** $i$ là **khách hàng trước đó trong giant tour**, không phải nút vật lý. Giá trị `prev_cust` không được đặt lại khi xe quay về kho, nên mùi vẫn học được quan hệ thứ tự xuyên tuyến.

---

### T3. Từ roulette wheel sang cuộc đua mũ (Gumbel/Exponential race)

**Vấn đề.** Cách ACO chuẩn chọn bước là chuẩn hóa tổng trọng số rồi tìm nhị phân trên tổng tích lũy. Trên GPU, tổng và tiền tố đòi hỏi đồng bộ toàn block, gây nghẽn và phân kỳ.

**Biến đổi.** Với trọng số $w_{jo} = \tau_{ijo}^{\alpha}\,\eta_{ijo}^{\beta}$, lấy $U_{jo} \sim \text{Uniform}(0,1]$ và đặt $E_{jo} = -\ln U_{jo}$. Khi đó $E_{jo} \sim \text{Exp}(1)$, và khóa

$$
K_{jo} = \frac{w_{jo}}{E_{jo}} = \frac{w_{jo}}{-\ln U_{jo}}
\tag{II.5}
$$

được dùng để chọn bước theo quy tắc $(j^*, o^*) = \arg\max K_{jo}$.

**Chứng minh xác suất.** Ta có $\arg\max_k K_k = \arg\min_k E_k / w_k$. Mỗi $X_k = E_k / w_k$ có phân phối $\text{Exp}(\lambda_k)$ với tốc độ $\lambda_k = w_k$. Với các biến mũ độc lập, xác suất để biến thứ $j$ nhỏ nhất là

$$
P(\text{chọn } j) = P\!\left(X_j = \min_k X_k\right) = \frac{\lambda_j}{\sum_k \lambda_k} = \frac{w_j}{\sum_k w_k}
$$

Vậy quy tắc argmax của (II.5) cho đúng phân phối roulette của ACO, không cần chuẩn hóa hay tìm nhị phân.

**Ý nghĩa cho GPU.** Mỗi ứng viên được tính độc lập trong thanh ghi. Phép chọn được rút gọn thành một phép argmax, và argmax có thể làm bằng xor-shuffle (T4). Mã đặt `U` có cận dưới $10^{-7}$ để tránh chia cho 0 khi $U \to 0$.

**Lưu ý.** `curand_uniform` trả về giá trị trong $(0, 1]$, nên về lý thuyết $U = 1$ là có thể, khi đó $-\ln U = 0$ và $K = +\infty$. Mã hiện chỉ kẹp cận dưới. Xác suất xảy ra rất nhỏ, nhưng nên kẹp cả hai đầu.

---

### T4. Song song hóa theo lane và rút gọn bằng xor-shuffle

**Ánh xạ.** Một warp (32 luồng) phục vụ một kiến. Lane $\ell \in \{0, \dots, 31\}$ xét tập ứng viên

$$
J_\ell = \{\, \ell + 1 + 32k \;:\; k \ge 0,\; \ell + 1 + 32k \le n \,\}
$$

Với mỗi $j \in J_\ell$ chưa thăm, lane tính $K_{jo}$ cho mọi phương thức khả dĩ rồi giữ cực đại cục bộ $(K^{\ell}, j^{\ell}, o^{\ell})$.

**Rút gọn.** Sau đó 32 lane rút gọn bằng bướm xor với bốn bước, cộng thêm một bước cuối:

$$
\underbrace{\text{offset} \in \{16, 8, 4, 2, 1\}}_{\log_2 32 = 5 \text{ bước}}, \qquad
(K, j, o) \leftarrow \max\big((K, j, o),\; \text{shfl\_xor}(K, j, o, \text{offset})\big)
$$

Hòa được phá theo chỉ số $j$ nhỏ hơn. Sau 5 bước, **mọi lane đều giữ cùng một kết quả**.

**Trạng thái sao chép.** Vì mọi lane có cùng kết quả, các lane cập nhật đồng nhất các biến `visited_mask`, `curr_load`, `curr_time`, `curr_phys_node`, `visited_lockers_mask` mà không cần phát sóng (broadcast) hay đồng bộ bằng shared memory. Lane 0 chỉ ghi kết quả vào `my_perm[step]`.

**Chi phí.** Mỗi bước chọn tốn khoảng $\lceil n/32 \rceil$ ứng viên mỗi lane (nhân tối đa 2 phương thức) cộng 5 bước shuffle. Vì vậy một kiến tốn khoảng $\mathcal{O}(n^2/32)$ thao tác đánh giá ứng viên. Đây là số bước shuffle, **không phải** "5 chu kỳ xung nhịp": mỗi shuffle tốn nhiều hơn một chu kỳ.

---

### T5. Heuristic $\eta$ với hằng số thực trong mã

Gọi $u$ là nút hiện tại, $v = \text{Node}(j,o)$, $T$ là thời điểm hiện tại và $\text{arr} = T + d_{uv}$.

**Bước khả thi** (không vi phạm tải hay cửa sổ, không ghé lại locker):

$$
\eta_{ijo} = \begin{cases}
\kappa_{merge} = 50, & \text{nếu } v \in V_L \land v = u, \\[8pt]
\dfrac{1}{\,\mu_{uv} + 1\,}, & \text{ngược lại,}
\end{cases}
\qquad
\mu_{uv} = \big(d_{uv} + 0.7\,W_{v} + 0.12\,\text{Urg}_v\big)\cdot \gamma_v
\tag{II.6}
$$

với

$$
W_v = \max(0,\; e_v - \text{arr}), \qquad
\text{Urg}_v = l_v - \max(\text{arr},\, e_v), \qquad
\gamma_v = \begin{cases} 0.65, & v \in V_L \\ 1, & \text{ngược lại} \end{cases}
$$

**Bước vi phạm** (xe mới được mở ngay từ kho):

$$
\eta_{ijo} = \frac{10^{-4}}{\,d_{0v} + 0.5\,\max(0,\, e_v - d_{0v}) + 0.12\,l_v + 1\,}
\tag{II.7}
$$

Bước vi phạm không bị cấm hoàn toàn. Nó chỉ có trọng số rất nhỏ, nên hoạt động như một lựa chọn "mở xe mới" bị phạt nặng. Trường hợp $\eta$ rất nhỏ nhưng mọi ứng viên khả thi đều có $\tau = 0$ thì bước vi phạm sẽ được chọn.

**Ý nghĩa cho GPU.** Các nhánh được viết thành biểu thức chọn lựa đơn giản, và mỗi lane tự tính $\eta$ của ứng viên mình. Phân kỳ chỉ xảy ra giữa các lane trong một warp, không đồng bộ toàn block.

---

### T6. Tìm kiếm cục bộ song song: 32 nước đi mỗi bước

**Ý tưởng.** Thay vì thử lần lượt từng nước đi, mỗi lane áp dụng **một** toán tử lên bản sao riêng của lộ trình và đánh giá toàn bộ $\mathcal{F}$. Sau đó warp lấy nước đi có chi phí nhỏ nhất bằng cùng phép xor-rút gọn ở T4:

$$
\ell^* = \arg\min_{\ell \in \{0,\dots,31\}} \mathcal{F}(\pi^{(\ell)}, o^{(\ell)}), \qquad
\text{nếu } \mathcal{F}(\pi^{(\ell^*)}) < \mathcal{F}_{cur}\text{ thì chấp nhận}
\tag{II.8}
$$

Mỗi bước LS là một bước leo đồi (hill climbing) với lựa chọn tốt nhất trong 32 nước đi ngẫu nhiên. Không có tiêu chí Metropolis, nên không nhận nước đi xấu hơn.

**Toán tử (chế độ Lever 3, `op = lane & 3`):**

| `op` | Toán tử | Mô tả ngắn |
| :---: | :--- | :--- |
| 0 | 2-opt và hoán đổi đoạn | Lane có `lane & 4 == 0`: đảo đoạn $[i_1, i_2]$ (2-opt). Lane còn lại: hoán đổi hai đoạn liền kề cùng độ dài |
| 1 | Relocate liên tuyến | Rút $\pi_{i_1}$ và chèn vào vị trí $i_2$. Khách flex có thể đổi $o$ với xác suất 1/2 |
| 2 | Swap liên tuyến | Hoán đổi $\pi_{i_1}$ và $\pi_{i_2}$. Khách flex có thể đổi $o$ với xác suất 1/2 |
| 3 | Gom cụm locker | Đặt $o_{i_1} = 1$ cho khách locker/flex, tìm khách khác cùng tủ và kéo về sát $i_1$ |

Chế độ cơ bản (không có Lever 3) dùng: 2-opt, swap, Or-opt với đoạn dài 1 đến 4, và lật mode kết hợp gom locker.

**Chi phí.** Mỗi bước LS thử 32 nước đi, mỗi nước một lần giải mã đầy đủ $O(n)$. Tổng số lần đánh giá của một kiến là $32 \cdot S_{ls}$ (nhân 1.5 khi bật Lever 1). Song song hóa ở đây là **song song giữa các nước đi**. Bản thân một lần giải mã vẫn tuần tự theo $n$.

---

### T7. Kế thừa tinh hoa và Phá hủy & Tái tạo (Lever 2)

**Kế thừa.** Từ vòng $\text{iter} \ge 1$, $T_{elite}$ kiến đầu tiên không xây lại lộ trình mà kế thừa lời giải tinh hoa của thế hệ trước. Các kiến còn lại (từ $T_{elite}$ trở đi) vẫn xây dựng mới bằng T4.

**Lever 2: Ruin & Recreate định hướng mùi.** Chỉ áp dụng cho các kiến tinh hoa có `ant_id ≥ ⌊T_elite/4⌋`.

1. **Ruin:** Lane 0 chọn một khoảng liên tiếp $[r, r+q)$ của giant tour và gỡ bỏ, với $q$ được tính như sau:

$$
q = \min\!\Big(\,\underbrace{q_0}_{10 \text{ nếu } n \ge 100,\; 6 \text{ nếu } n \ge 50,\; 4 \text{ ngược lại}},\; \lfloor n/3 \rfloor,\; 14\Big)
$$

2. **Recreate:** Với từng khách $c$ bị gỡ (theo thứ tự đã lưu), mọi lane cùng tính điểm cho các vị trí chèn $p \in \{0, \dots, \text{len}\}$:

$$
\text{Score}(p) = \frac{\tau_{\pi_{p-1},\, c,\, m_c} + 0.1}{\max\!\big(\Delta C_p,\; 0.1\big) + 1}, \qquad
\Delta C_p = d_{a c} + d_{c b} - d_{a b}
\tag{II.9}
$$

với $a$ và $b$ là nút vật lý hai bên vị trí $p$. Điểm cao nhất được chọn bằng cùng phép rút gọn xor ở T4, và $c$ được chèn với phương thức $m_c$ được giữ nguyên.

Với kiến tinh hoa không được chọn Lever 2, mã dùng hoán đổi hai vị trí ngẫu nhiên cho các kiến có `ant_id ≥ ⌊T_elite/2⌋`.

**Ý nghĩa.** Ruin & Recreate tạo ra các nước đi lớn hơn một bước LS, nhưng khác với ruin ngẫu nhiên thuần túy, bước tái tạo vẫn dùng mùi tích lũy của bầy kiến.

**Lưu ý.** Ruin chỉ gỡ khoảng liên tiếp. Tài liệu methodology có nhắc đến việc gỡ cả cụm locker. Mã hiện **chưa** thực hiện điều đó.

---

### T8. Cập nhật mùi: bay hơi kẹp ngưỡng rồi lắng đọng nguyên tử

**Bay hơi và kẹp** (kernel từng phần tử, cho mọi phần tử của $\tau$):

$$
\tau_{ijo} \leftarrow \begin{cases}
\text{clip}\!\big((1-\rho)\,\tau_{ijo},\; \tau_{min},\; \tau_{max}\big), & M_{ijo} = 1, \\
0, & M_{ijo} = 0
\end{cases}
\tag{II.10}
$$

**Lắng đọng** (kernel theo lưới $T_{elite} \times (n+1)$, mỗi luồng ứng với một cạnh của giant tour):

$$
\tau_{uvd} \;\xleftarrow{\text{atomicAdd}}\; \frac{Q}{\mathcal{F}_r}\cdot M_{uvd}, \qquad r = 1, \dots, T_{elite}
\tag{II.11}
$$

Giant tour có $n+1$ cạnh: $(0, \pi_1)$, $(\pi_k, \pi_{k+1})$ với $k = 1, \dots, n-1$, và $(\pi_n, 0)$. Cạnh $(u,v)$ nhận mode $d$ của nút đích $v$, khớp với cách tra cứu mùi trong T4. Giá trị $Q = 2.0$ và $\mathcal{F}_r \le 0$ bị bỏ qua.

**Thứ tự.** Mã **bay hơi và kẹp trước**, rồi mới lắng đọng. Vì vậy sau một lần lắng đọng, $\tau$ có thể vượt $\tau_{max}$ cho đến lần kẹp tiếp theo.

**Vì sao không có tranh chấp nghiêm trọng.** Nhiều luồng có thể cộng vào cùng một ô $\tau$ (ví dụ cùng cạnh xuất hiện trong nhiều tinh hoa). Phần cứng xử lý bằng `atomicAdd`, nên kết quả đúng nhưng các cộng này bị tuần tự hóa ở mức ô.

**Cập nhật bay hơi thích nghi (sigmoid).** Hệ số bay hơi chỉ được cập nhật khi vòng hiện tại **không** cải thiện:

$$
\rho(I_{st}) = \rho_{ini} + \big(\rho_{max} - \rho_{ini}\big)\cdot \sigma\!\left(10\left(\frac{I_{st}}{I_{max}} - 0.5\right)\right), \qquad \rho_{ini} = 0.3,\; \rho_{max} = 0.7
\tag{II.12}
$$

Khi vòng cải thiện, $I_{st}$ được đặt về 0 nhưng $\rho$ **giữ nguyên** giá trị cũ. Chỉ khi không cải thiện thì $\rho$ mới được tính lại theo (II.12).

---

### T9. Sinh số ngẫu nhiên hoàn toàn trên thiết bị

Mỗi luồng có một `curandState` riêng, được khởi tạo một lần bằng `curand_init(seed, tid, 0, ...)`. Trạng thái được ghi lại vào VRAM khi kết thúc kernel. Mọi số ngẫu nhiên (khóa Gumbel, chọn $i_1, i_2$, lật mode, chọn $r$ trong Ruin) đều lấy từ trạng thái này, không cần truyền từ host.

Một lưu ý về tính đúng đắn: `curandState` mặc định là XORWOW, **không** phải MRG32k3a như ghi trong một số tài liệu cũ.

---

### T10. Vòng lặp kín: dữ liệu ở lại GPU, điều khiển ở CPU

**Phát biểu chính xác.** Toàn bộ **tính toán nặng** (xây dựng, LS, Ruin & Recreate, bay hơi, lắng đọng) chạy trên GPU. Dữ liệu lời giải và mùi **không rời VRAM** trong vòng lặp. Host chỉ là một bộ điều khiển nhẹ, và mỗi vòng lặp trao đổi với GPU một lượng dữ liệu tỉ lệ với $N_{ants}$:

| Việc trong mỗi vòng lặp | Chiều truyền | Kích thước |
| :--- | :--- | :--- |
| Đọc chi phí của mọi kiến để sắp xếp | D → H | $N_{ants}$ số float |
| Sắp xếp partial để chọn top-$T_{elite}$ | CPU | $N_{ants}$ phần tử |
| Tính $\rho$ và cập nhật $I_{st}$ | CPU | vô hướng |
| Ghi chỉ số tinh hoa | H → D | $T_{elite}$ số nguyên |
| Khi có cải thiện: đọc lời giải tốt nhất | D → H | $n$ + $n$ số nguyên |

Lời giải tốt nhất toàn cục được kiểm chứng lại bằng `evaluate_solution_host` một lần khi kết thúc.

**Khi bật `--ig`:** Mã **tắt** LS trong kernel (`g_use_ig ? 0 : ls_steps`) và chuyển $T_{elite}$ kiến hàng đầu (tối đa $k = 4$) sang CPU để tinh chỉnh bằng RVND và Iterated Greedy của tác giả. Mỗi vòng lặp khi đó sao chép $N_{ants} \cdot n$ số nguyên (`perm` và `mode`) mỗi chiều. Đây là một **biến thể lai CPU-GPU**, không phải pipeline thuần GPU. Chi phí cuối cùng được lấy từ IG, không kiểm chứng lại bằng bộ giải mã GPU.

> **Lưu ý khi trình bày:** Pipeline thuần GPU trong tài liệu này là cấu hình chính. Cờ `--ig` là một biến thể khác, đưa LS trở lại CPU.

---

### Giới hạn kiến trúc đã biết

| Giới hạn | Nguyên nhân trong mã | Hệ quả |
| :--- | :--- | :--- |
| $n \le 127$ | `MAX_CUST = 128`, mặt nạ thăm `visited_mask[4]` (4 × 32 bit) | Không chạy được với $n > 127$ |
| $m \le 32$ locker | `visited_lockers_mask` là `uint32` | Mỗi tuyến chỉ theo dõi tối đa 32 locker |
| Không có $K$, $F_{fix}$ | Hàm đánh giá chỉ tính quãng đường | Số xe là kết quả phụ |
| Bộ nhớ local cho LS | `local_perm`, `local_mode` mỗi thứ 512 B, chỉ số động | Truy cập chậm hơn thanh ghi |
| Hằng số cứng | $\rho_{max} = 0.7$, $\tau \in [0.01, 20]$, $Q = 2$, $\alpha, \beta$ | Chưa được điều chỉnh theo quy mô |

---

## III. Luồng thuật toán tổng thể (mã giả)

### Sơ đồ khối

```
 ┌─────────────────────── HOST (CPU) ───────────────────────┐
 │  Khởi tạo: đọc instance, tính mask M, τ ← M,  seed      │
 │  Cấp phát VRAM một lần, copy instance lên GPU            │
 └──────────────────────────────┬───────────────────────────┘
                                ▼
 ┌────────────── vòng lặp cho đến khi I_st ≥ I_max ─────────┐
 │                                                           │
 │  [GPU] construct_ls<<<N_ants/4, 128>>>  (Alg. 2)          │
 │        ├─ kiến ant < T_elite, iter>0: kế thừa + R&R      │
 │        ├─ còn lại: construct (Alg. 3, Gumbel + shuffle)  │
 │        └─ LS S_ls bước, 32 nước đi/bước (Alg. 5)         │
 │                       │                                  │
 │  [CPU] D2H costs[N_ants]; partial_sort top-T_elite       │
 │        cập nhật F*, I_st, ρ (sigmoid, II.12)              │
 │                       │                                  │
 │  [GPU] copy_elites<<<T_elite, n>>>                       │
 │  [GPU] evaporate_and_clamp (II.10)                       │
 │  [GPU] deposit<<<T_elite, n+1>>> atomicAdd (II.11)       │
 └──────────────────────────────┬───────────────────────────┘
                                ▼
 ┌─────────────────────── HOST (CPU) ───────────────────────┐
 │  Kiểm chứng S* bằng evaluate_solution_host (Alg. 6)      │
 └──────────────────────────────────────────────────────────┘
```

### Thuật toán 1: Vòng lặp điều khiển trên host

```text
ALGORITHM 1: GPU_3D_PACO_HOST(instance, N_ants, T_elite, I_max, S_ls, α, β, ρ_ini, Q)
INPUT : instance (n, m, C, d, e, l, s, q, cust_type, cust_locker)
OUTPUT: lời giải đã kiểm chứng (S*, F*)

 1  cấp phát VRAM; copy instance lên GPU                        ▷ một lần
 2  h_mask ← M_{ijo} theo (II.4);  h_tau ← 1 nơi M=1, 0 nơi M=0
 3  copy h_tau, h_mask lên GPU
 4  KERNEL init_rng<<<⌈32·N_ants / 128⌉, 128>>>(seed)
 5  F* ← +∞;  I_st ← 0;  iter ← 0;  ρ ← ρ_ini
 6  while I_st < I_max do
 7      KERNEL construct_ls<<<N_ants/4, 128>>>(iter, ...)        ▷ Alg. 2
 8      D2H  costs[0..N_ants−1]                                    ▷ N_ants floats
 9      idx ← partial_sort(0..N_ants−1 by costs), lấy T_elite đầu
10      if BẬT --ig then                                           ▷ biến thể lai
11          ig_refine(idx[0..min(k,T_elite)−1])                  ▷ Alg. 7
12      end if
13      if costs[idx_0] + 10⁻⁴ < F* then
14          F* ← costs[idx_0];  D2H S* ← perm/mode của kiến idx_0
15          I_st ← 0                                              ▷ ρ giữ nguyên
16      else
17          I_st ← I_st + 1
18          ρ ← ρ_ini + (ρ_max − ρ_ini) · σ(10 (I_st / I_max − 0.5))  ▷ (II.12)
19      end if
20      H2D elite_idx ← idx[0..T_elite−1]
21      KERNEL copy_elites<<<T_elite, n>>>(elite_idx)               ▷ Alg. 5a
22      KERNEL evaporate<<<⌈2(n+1)²/256⌉, 256>>>(ρ)                 ▷ Alg. 5b, (II.10)
23      KERNEL deposit<<<T_elite, n+1>>>(Q)                         ▷ Alg. 5c, (II.11)
24      iter ← iter + 1
25  end while
26  (S*, F*_verified) ← evaluate_solution_host(S*)                 ▷ Alg. 6
27  return (S*, F*_verified)
```

Ghi chú: Dòng 13 trong mã thật so sánh với $\mathcal{F}^*$ bằng ngưỡng $10^{-4}$. Trong biến thể không `--ig`, cập nhật toàn cục chỉ xảy ra khi `iter_best + 1e-4 < global_best`.

### Thuật toán 2: Kernel xây dựng và tìm kiếm cục bộ (một warp = một kiến)

```text
ALGORITHM 2: CONSTRUCT_LS_KERNEL(iter, ...)  ▷ g = global thread id, ant = ⌊g/32⌋, ℓ = g mod 32
 1  if ant ≥ N_ants then return
 2  rng ← rng_states[g]
 3  if iter > 0 and ant < T_elite then                        ▷ kiến tinh hoa
 4      (π, o) ← copy elite_perm[ant], elite_mode[ant]         ▷ n phần tử, mỗi lane một phần
 5      if BẬT Lever 2 then
 6          if ant ≥ ⌊T_elite/4⌋ then RUIN_RECREATE(π, o)      ▷ Alg. 4
 7      else if ant ≥ ⌊T_elite/2⌋ and ℓ = 0 then
 8          hoán đổi ngẫu nhiên hai vị trí trong (π, o)
 9      end if
10  else                                                        ▷ kiến mới
11      CONSTRUCT(π, o)                                         ▷ Alg. 3
12  end if
13  F_cur ← DECODE(π, o)                                       ▷ Alg. 6 (mỗi lane tính lại, đồng nhất)
14  S_rounds ← S_ls;  if BẬT Lever 1 then S_rounds ← S_ls + ⌊S_ls/2⌋ end if
15  for s = 1 to S_rounds do                                    ▷ Tìm kiếm cục bộ, Alg. 5
16      (π', o') ← copy (π, o)                                  ▷ bản sao riêng của lane ℓ
17      i₁, i₂ ← U{0..n−1}  (ngẫu nhiên trên thiết bị)
18      LOCAL_MOVE(ℓ & 3, i₁, i₂, π', o')                        ▷ bảng toán tử ở mục T6
19      F_ℓ ← DECODE(π', o')
20      (F_best, ℓ_best) ← WARP_ARGMIN(F_ℓ, ℓ)                  ▷ 5 bước xor-shuffle
21      if F_best < F_cur then
22          F_cur ← F_best
23          if ℓ = ℓ_best then ghi (π', o') vào bộ đệm chung
24          __syncwarp();  mọi lane sao chép bộ đệm về (π, o)
25      end if
26  end for
27  ghi (π, o) vào d_all_perms[ant], d_all_modes[ant]; ℓ=0 ghi d_all_costs[ant] ← F_cur
28  rng_states[g] ← rng
```

### Thuật toán 3: Xây dựng lộ trình hợp tác trong warp (CONSTRUCT)

```text
ALGORITHM 3: CONSTRUCT(π, o)  ▷ chạy trong warp, tất cả lane giữ cùng trạng thái sau mỗi bước
 1  mask[0..3] ← 0;  prev ← 0;  node ← 0;  load ← 0;  T ← 0;  seen ← ∅
 2  for step = 1 to n do
 3      K* ← −1;  j* ← −1;  o* ← 0
 4      for j ∈ J_ℓ = {ℓ+1, ℓ+33, ℓ+65, ...}, j ≤ n do           ▷ ứng viên của lane ℓ
 5          if j ∉ mask then
 6              for o ∈ MODES(type(j)) do                       ▷ Home:{0}, Locker:{1}, Flex:{0,1}
 7                  τ_val ← tau[prev, j, o]
 8                  if τ_val ≤ 0 then continue                  ▷ bị cấm theo M
 9                  v ← Node(j, o);  arr ← T + d[node, v]
10                  viol ← (load + q_j > C) ∨ (arr > l_v + 10⁻³) ∨ DUP(v, seen, node)
11                  η ← ETA(node, v, arr, viol)                  ▷ (II.6) hoặc (II.7)
12                  w ← τ_val^α · η^β
13                  U ← curand_uniform(rng);  U ← max(U, 10⁻⁷)
14                  K ← w / (−ln U)                              ▷ (II.5)
15                  if K > K* then (K*, j*, o*) ← (K, j, o)
16              end for
17          end if
18      end for
19      (K*, j*, o*) ← WARP_ARGMAX(K*, j*, o*)                    ▷ 5 bước xor-shuffle, hòa ưu tiên j nhỏ
20      if j* ≤ 0 then break
21      π[step] ← j*;  o[step] ← o*                              ▷ lane 0 ghi
22      mask ← mask ∪ {j*}
23      CẬP NHẬT (load, T, node, seen, prev) theo T1 — đồng nhất trên mọi lane
24  end for
```

Lưu ý: Dòng 23 gồm cùng logic với bộ giải mã (II.1)–(II.2), trong đó việc đặt lại tuyến khi vi phạm dùng `arr = d[0, v]`.

### Thuật toán 4: Ruin & Recreate định hướng mùi (Lever 2)

```text
ALGORITHM 4: RUIN_RECREATE(π, o)  ▷ chỉ kiến tinh hoa, Lever 2 bật
 1  q ← min(q₀(n), ⌊n/3⌋, 14)                                  ▷ q₀ = 10 / 6 / 4
 2  ▷ Ruin (lane 0)
 3  r ← U{0, …, n−q}
 4  R ← (π[r..r+q−1], o[r..r+q−1]);  dịch đuôi trái q vị trí
 5  ▷ Recreate
 6  for t = 1 to q do
 7      (c, m) ← R[t];  len ← n − q + t − 1
 8      for p ∈ {ℓ, ℓ+32, ...}, p ≤ len do                       ▷ vị trí chèn 0..len
 9          a ← (p = 0 ? 0 : π[p−1]);  b ← (p = len ? 0 : π[p])
10          Δ ← d[Node(a)][Node(c,m)] + d[Node(c,m)][Node(b)] − d[Node(a)][Node(b)]
11          score_p ← (tau[a, c, m] + 0.1) / (max(Δ, 0.1) + 1)     ▷ (II.9)
12      end for
13      (score*, p*) ← WARP_ARGMAX(score, p)
14      lane 0 chèn (c, m) vào vị trí p* (dịch phải)
15  end for
```

### Thuật toán 5: Kernel bay hơi và lắng đọng

```text
ALGORITHM 5a: COPY_ELITES<<<T_elite, n>>>(elite_idx)
 1  r ← blockIdx.x;  k ← threadIdx.x
 2  a ← elite_idx[r]
 3  elite_perm[r, k] ← d_all_perms[a, k];  elite_mode[r, k] ← d_all_modes[a, k]

ALGORITHM 5b: EVAPORATE_CLAMP<<<⌈2(n+1)²/256⌉, 256>>>(ρ)           ▷ theo (II.10)
 1  e ← blockIdx.x · blockDim.x + threadIdx.x
 2  if e ≥ 2(n+1)² then return
 3  if M[e] = 0 then τ[e] ← 0
 4  else τ[e] ← clip((1 − ρ) · τ[e], τ_min, τ_max)

ALGORITHM 5c: DEPOSIT<<<T_elite, n+1>>>(Q)                          ▷ theo (II.11)
 1  r ← blockIdx.x;  s ← threadIdx.x
 2  a ← elite_idx[r];  F ← d_all_costs[a]
 3  if F ≤ 0 then return
 4  if s = 0 then (u, v, o) ← (0, π_a[1], o_a[1])
 5  else if s < n then (u, v, o) ← (π_a[s], π_a[s+1], o_a[s+1])
 6  else (u, v, o) ← (π_a[n], 0, 0)
 7  e ← idx(u, v, o)                                                  ▷ (II.3)
 8  if M[e] = 1 then atomicAdd(&τ[e], Q / F)
```

Ghi chú về chỉ số: Mã thật dùng chỉ số 0-based. Ở đây dùng 1-based cho dễ đọc. Điều kiện `s = 0` ứng với cạnh $(0, \pi_1)$, `s = n` ứng với cạnh $(\pi_n, 0)$.

### Thuật toán 6: Bộ giải mã (DECODE), dùng chung trên device và host

```text
ALGORITHM 6: DECODE(π, o)  ▷ trả về (F, số xe)
 1  F ← 0;  veh ← 0;  load ← 0;  T ← 0;  u ← 0;  last ← 0;  seen ← ∅
 2  k ← 1
 3  while k ≤ n do
 4      c ← π[k];  v ← Node(c, o[k]);  q ← q_c
 5      arr ← T + d[u, v]
 6      dup ← (v ∈ V_L) ∧ (v ∈ seen) ∧ (v ≠ last)
 7      if load + q > C ∨ arr > l_v + 10⁻³ ∨ dup then               ▷ (II.1)
 8          if u ≠ 0 then F ← F + d[u, 0] end if
 9          veh ← veh + 1;  load ← 0;  u ← 0;  T ← 0;  seen ← ∅;  last ← 0
10          continue                                                ▷ thử lại cùng khách k từ kho
11      end if
12      F ← F + d[u, v]
13      T ← max(arr, e_v) + (v ∈ V_L ∧ v = u ? 0 : s_v)           ▷ (II.2)
14      load ← load + q;  u ← v;  last ← v
15      if v ∈ V_L then seen ← seen ∪ {v} end if
16      k ← k + 1
17  end while
18  if u ≠ 0 then F ← F + d[u, 0];  veh ← veh + 1 end if
19  return (F, veh)
```

Ghi chú: Bản thực thi trên device và trên host có cùng logic. Khác biệt duy nhất là host dùng kiểu `double` để cộng dồn quãng đường.

### Thuật toán 7: Tinh chỉnh bằng IG trên CPU (biến thể `--ig`)

```text
ALGORITHM 7: IG_REFINE(idx[0..k−1])  ▷ k = min(g_ig_k, T_elite) = 4 mặc định, chỉ khi bật --ig
 1  LS trong kernel bị tắt: S_ls ← 0
 2  D2H  perms[0..N_ants·n−1], modes[0..N_ants·n−1]              ▷ N_ants·n số nguyên mỗi chiều
 3  for q = 0 to k−1 do
 4      a ← idx[q]
 5      (π', o', F_IG) ← IG_RVND(perms[a], modes[a])               ▷ bộ IG của tác giả, qua ig_bridge
 6      F_eval ← DECODE_host(π', o')
 7      costs[a] ← min(F_IG, F_eval)
 8      H2D perm[a], mode[a]
 9      if F_IG < F* then F* ← F_IG;  S* ← (π', o')
 10 end for
11 H2D costs; sắp xếp lại idx
12 Khi --ig, F_verified ← F_IG (không kiểm chứng lại bằng GPU)
```

---

## Phụ lục: Kiểm tra nhanh trước khi trình bày

- Chỉ số tuyến tính $\text{idx}(i, j, o) = i\cdot 2(n+1) + 2j + o$ và kích thước $2(n+1)^2$ khớp `tau_elements` trong `run_gpu_3d_paco_single`.
- Số warp mỗi block là 4 (`threads_per_block = 128`), nên số block là $N_{ants}/4$.
- Các giá trị $\alpha=1.25$, $\beta=1.5$, $\rho_{ini}=0.3$, $Q=2.0$ khớp với lời gọi trong `run_dataset_benchmark`.
- Bộ kiểm chứng cuối cùng trên host không dùng cạnh $\text{ETA}$ hay mùi. Nó chỉ tính lại $\mathcal{F}$ từ $(\pi, o)$.
