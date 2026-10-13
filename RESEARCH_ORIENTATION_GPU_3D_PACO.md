# ĐỊNH HƯỚNG NGHIÊN CỨU & LỘ TRÌNH PHÁT TRIỂN: ĐƯA 3D-PACO LÊN GPU
**Dự án:** Vehicle Routing Problem with Parcel Lockers (VRPPL)  
**Tài liệu tham chiếu:** Bài báo *"Parallel ant colony optimization for vehicle routing with parcel lockers"*, *Swarm and Evolutionary Computation*, Tập 104 (2026) 102371.  
**Tác giả tham chiếu:** Phú An Châu, Loan T.T. Nguyễn, Witold Pedrycz, Bảy Võ.  
**Kho mã nguồn:** [routing-PL](https://github.com/congnghip/routing-PL)

---

## MỤC LỤC
1. [Bản chất thuật toán & Cơ chế Pheromone trong 3D-PACO](#1-bản-chất-thuật-toán--cơ-chế-pheromone-trong-3d-paco)
2. [Khoảng trống nghiên cứu & Tính khả thi cho bài báo mới (Q1)](#2-khoảng-trống-nghiên-cứu--tính-khả-thi-cho-bài-báo-mới-q1)
3. [Phân tích giảm thiểu chi phí (Cost Reduction Analysis)](#3-phân-tích-giảm-thiểu-chi-phí-cost-reduction-analysis)
4. [Tình trạng phần cứng hệ thống hiện tại](#4-tình-trạng-phần-cứng-hệ-thống-hiện-tại)
5. [Lộ trình triển khai từng bước (Step-by-Step Execution Roadmap)](#5-lộ-trình-triển-khai-từng-bước-step-by-step-execution-roadmap)
6. [Định hướng bài báo khoa học mục tiêu](#6-định-hướng-bài-báo-khoa-học-mục-tiêu)

---

## 1. Bản chất thuật toán & Cơ chế Pheromone trong 3D-PACO

### 1.1. Lần đầu tiên đi qua con đường đó, kiến có tiết pheromone không?
* **Câu trả lời:** **KHÔNG**.
* **Nguyên lý:**
  1. Trong ACO, lượng pheromone để lại phụ thuộc vào **chất lượng của toàn bộ hành trình**:
     $$\Delta \tau_{ijo}^{(k)} = \frac{Q}{L^{(k)}}$$
     *(với $L^{(k)}$ là tổng quãng đường hoàn thành của cả chuyến xe).*
  2. Khi kiến mới đi qua một chặng nhỏ, nó chưa thể biết toàn bộ chuyến đi sau đó sẽ dài hay ngắn, tối ưu hay vi phạm ràng buộc. Do đó, kiến phải **đi xong toàn bộ lộ trình**, quay về kho trung tâm và đánh giá tổng chi phí $L^{(k)}$ thì mới xác định được lượng pheromone để lại.
  3. **Cơ chế cập nhật tinh hoa (Elitist Update) trong 3D-PACO:** Thậm chí khi đã đi xong, **chỉ top $t$ con kiến có lộ trình ngắn nhất** trong chu kỳ mới được phép để lại pheromone. Những con kiến đi đường kém tối ưu hoàn toàn không được rải pheromone.
  4. **Ở vòng lặp đầu tiên:** Toàn bộ ma trận pheromone được gán sẵn một giá trị nền khởi tạo đồng nhất $\tau_0 > 0$. Kiến dựa vào $\tau_0$ và thông tin khoảng cách heuristic $\eta_{ij} = \frac{1}{d_{ij}}$ để ra quyết định chọn đường.

### 1.2. Pheromone có tương đương trên cả lộ trình A - B - C - D khi kiến đi xong không?
* **Lượng pheromone cộng thêm do chính con kiến đó để lại ($\Delta \tau$): 👉 BẰNG NHAU trên mọi chặng.**
  - Nếu kiến đi $A \to B \to C \to D$ có tổng chiều dài $L$, thì mỗi chặng ($A \to B, B \to C, C \to D$) đều được cộng thêm cùng một lượng $\Delta \tau = \frac{Q}{L}$.
  - Lý do: Thuật toán đánh giá sự thành công của **toàn bộ giải pháp tổng thể**. Một đoạn đường dù dài, nhưng nếu nằm trong một giải pháp tổng thể tối ưu ($L$ nhỏ), thì nó vẫn được thưởng lượng pheromone cao.
* **Tổng nồng độ pheromone tích lũy trên mỗi đoạn ($\tau_{mới}$): 👉 KHÔNG BẰNG NHAU.**
  - **Cộng dồn từ nhiều kiến:** Đoạn đường nào càng nằm trong nhiều lộ trình tốt của các con kiến khác nhau thì càng được cộng dồn nhiều lần $\Delta \tau$.
  - **Lịch sử bay hơi:** $\tau \leftarrow (1 - \rho)\tau + \sum \Delta \tau$. Các cung đường đã tốt từ các thế hệ trước sẽ có nồng độ nền cao hơn.

---

## 2. Khoảng trống nghiên cứu & Tính khả thi cho bài báo mới (Q1)

### 2.1. Tình trạng nghiên cứu thế giới hiện nay (State of the Art)
* **Về bài toán VRPPL (Vehicle Routing Problem with Parcel Lockers):**
  - Mới xuất hiện từ năm 2021–2022 (Yu et al. 2022 dùng Simulated Annealing; Châu et al. 2026 dùng 3D-PACO OpenMP CPU; Chen et al. 2024 dùng Multi-objective).
  - **100% các nghiên cứu VRPPL trên thế giới hiện nay đều chạy trên CPU.**
* **Về ACO trên GPU:**
  - Các công trình hiện tại (TensorACO, Skinderowicz 2020-2022, Cecilia 2018) chỉ giải quyết các bài toán kinh điển: TSP (Người bán hàng), CVRP thông thường với **ma trận pheromone 2 chiều $\tau_{ij}$**.
  - **CHƯA CÓ BẤT KỲ CÔNG TRÌNH NÀO đưa ma trận Pheromone 3 chiều ($\tau_{ijo}$ - 3D Pheromone Tensor) lên GPU.**

### 2.2. Đóng góp khoa học cốt lõi (Scientific Novelties) để công bố Q1
Để bài báo đạt chuẩn các tạp chí top đầu (*Swarm and Evolutionary Computation*, *Future Generation Computer Systems*, *IEEE TEVC*, *JPDC*), đề tài cần tập trung giải quyết các bài toán cấu trúc trên GPU:
1. **Mô hình hóa 3D Pheromone Tensor trên bộ nhớ GPU:**
   - Thay vì lưu mảng 3 chiều rời rạc trong RAM, thiết kế cấu trúc dữ liệu phẳng (Flattened Tensor) liên tục trên VRAM để đạt cơ chế **Coalesced Memory Access**.
   - Khai thác các phép nhân ma trận/ten-xơ tốc độ cao (Tensor Cores / CuBLAS) để tính ma trận xác suất chuyển trạng thái $P_{i \to j, o}$.
2. **Khắc phục hiện tượng Phân kỳ luồng (Warp Divergence):**
   - Ràng buộc VRPPL phức tạp (tải trọng xe, khung thời gian, tùy chọn nhận tại nhà hoặc tại tủ) khiến mỗi con kiến rẽ các nhánh điều kiện khác nhau.
   - Cần thiết kế kỹ thuật phân nhóm kiến hoặc cấu trúc tính toán phi phân nhánh (*branchless / warp-aligned computation*) để tối ưu hiệu năng SIMT của GPU.
3. **Massively Parallel Local Search trên GPU:**
   - Thiết kế các toán tử 2-opt, Swap, Insert chạy song song hoàn toàn trên hàng chục ngàn CUDA threads mà không bị nghẽn xung đột bộ nhớ (*race conditions*).
4. **Mở rộng quy mô đột phá (Super Large-Scale Instances):**
   - Bài báo 2026 mới chỉ dừng ở 100 khách hàng (Solomon benchmark).
   - Với GPU, mở rộng thử nghiệm lên **500 đến 2.000 khách hàng** — quy mô mà CPU đa lõi không thể giải quyết trong thời gian thực.

---

## 3. Phân tích giảm thiểu chi phí (Cost Reduction Analysis)

Đưa 3D-PACO lên GPU mang lại sự tối ưu trên cả 4 khía cạnh chi phí:

| Loại chi phí | Hiện trạng (CPU 64 luồng) | Kỳ vọng khi lên GPU (RTX 5060 Ti / CUDA) | Tác động thực tế |
| :--- | :--- | :--- | :--- |
| **Thời gian tính toán (Wall-clock Time)** | 267 giây trên tập 100 khách (Large). | Dự kiến **dưới 10 – 20 giây** (Speedup $15\times - 30\times$). | Cho phép ra quyết định điều phối xe gần như tức thời (*near real-time logistics*). |
| **Chi phí quãng đường (Solution Cost)** | Quần thể bị giới hạn: $m = 50 \div 400$ kiến do nghẽn CPU. | Huy động **$10.000 \div 50.000$ kiến** đồng thời trong cùng 1 chu kỳ. | Không gian tìm kiếm rộng hơn nhiều $\to$ tìm ra lộ trình ngắn hơn, giảm chi phí nhiên liệu đội xe. |
| **Năng lượng tiêu thụ (Energy Cost)** | CPU Xeon 64 luồng tiêu thụ ~150-200W liên tục trong hàng trăm giây. | GPU hoàn thành tác vụ cực nhanh, tổng điện năng $E = P \times t$ **giảm 30% – 70%**. | Hướng tới logistics xanh và tính toán bền vững (*Green Computing*). |
| **Hạ tầng máy chủ (Cloud Instance Cost)** | Thuê instance CPU cao cấp (64 vCPUs) giá đắt (~$2.5 – $3.5/h). | Thuê GPU tầm trung (L4/T4/RTX) giá rẻ hơn hoặc tương đương, thời gian chạy giảm mạnh. | **Chi phí trên mỗi lượt giải (Cost per solve) giảm đáng kể.** |

---

## 4. Tình trạng phần cứng hệ thống hiện tại

* **GPU:** NVIDIA GeForce RTX 5060 Ti
* **VRAM:** 16,311 MiB (16 GB GDDR6/GDDR7)
* **Driver:** 595.91.07
* **CUDA Driver Version hỗ trợ:** 13.2
* **Đánh giá:** Phần cứng cực kỳ mạnh mẽ, thế hệ mới nhất, 16GB VRAM thừa sức chứa cả ma trận pheromone 3D cho các bài toán quy mô hàng ngàn khách hàng.

---

## 5. Lộ trình triển khai từng bước (Step-by-Step Execution Roadmap)

### Giai đoạn 1: Chuẩn bị môi trường & Profiling mã nguồn
- [ ] Cài đặt CUDA Toolkit (`nvidia-cuda-toolkit`) để có trình biên dịch `nvcc`.
- [ ] Phân tích điểm nghẽn của `src/src/solvers/PACO.cpp`: hàm `paco_construct_solution` chiếm >80% thời gian chạy.
- [ ] Thiết lập thư mục làm việc mới `src/cuda/` để độc lập với mã nguồn CPU hiện có.

### Giai đoạn 2: Thiết kế cấu trúc dữ liệu phẳng & Kernel Tour Construction
- [ ] Chuyển đổi `tau[i][j][d]` thành mảng 1D liên tục trong Global Memory: `index = i * (N * 2) + j * 2 + d`.
- [ ] Đưa ma trận khoảng cách $d_{ij}$ và thông tin khách hàng vào GPU Memory (ưu tiên Constant/Shared Memory).
- [ ] Viết Kernel tạo đường đi song song bằng CUDA C++ kết hợp thư viện `cuRAND`.
- [ ] Triển khai song song reduction tính tổng xác suất chuyển trạng thái.

### Giai đoạn 3: Hiện thực hóa Local Search & Pheromone Update trên GPU
- [ ] Viết Kernel 2-opt song song cho từng khối kiến (mỗi kiến chạy tinh chỉnh trên VRAM).
- [ ] Viết Kernel cập nhật pheromone (Bay hơi $\tau = (1-\rho)\tau$ song song toàn ma trận; Tích tụ tinh hoa bằng `atomicAdd`).
- [ ] Khép kín toàn bộ chu trình lặp bên trong GPU, triệt tiêu chi phí truyền dữ liệu Host $\leftrightarrow$ Device qua PCIe.

### Giai đoạn 4: Thực nghiệm & Đối sánh Benchmarks
- [ ] Chạy kiểm thử trên 56 bộ dữ liệu chuẩn Solomon (25, 50, 100 khách hàng), đối chiếu trực tiếp với kết quả trong bài báo 2026.
- [ ] Xây dựng bộ dữ liệu mở rộng quy mô lớn (200, 500, 1.000 khách hàng).
- [ ] Đo đạc và vẽ biểu đồ 3 chỉ số then chốt: **Speedup ($S$)**, **Chất lượng nghiệm (BKS)**, và **Năng lượng tiêu thụ (Joules)** qua `nvidia-smi`.

### Giai đoạn 5: Viết bài báo khoa học & Nộp tạp chí
- [ ] Hoàn thiện bản thảo bài báo theo cấu trúc chuẩn IMRaD.
- [ ] Tạo repository mã nguồn mở trên GitHub phục vụ tính tái lập thực nghiệm (*Reproducibility*).
- [ ] Nộp bài vào các tạp chí Q1 phù hợp.

---

## 6. Định hướng bài báo khoa học mục tiêu

* **Tên bài báo đề xuất:**  
  *Massively Parallel 3D-Tensor Ant Colony Optimization on GPU for Large-Scale Vehicle Routing with Parcel Lockers*  
  *(Tối ưu hóa đàn kiến ten-xơ 3D song song ồ ạt trên GPU cho bài toán định tuyến phương tiện có tủ nhận bưu kiện quy mô lớn)*
* **Các tạp chí mục tiêu (Q1 / Scopus / WoS):**
  1. *Swarm and Evolutionary Computation* (Elsevier) - IF ~8.0
  2. *Future Generation Computer Systems* (Elsevier) - IF ~7.5
  3. *Journal of Parallel and Distributed Computing* (Elsevier) - IF ~3.8
  4. *Computers & Industrial Engineering* (Elsevier) - IF ~7.1
  5. *IEEE Transactions on Evolutionary Computation* (IEEE) - IF ~14.0
