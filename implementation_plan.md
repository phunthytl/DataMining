# Triển khai hệ thống: Cluster-based Association Rule Mining (Phân cụm + Khai phá luật)

Mục tiêu: Nâng cấp mô hình FP-Growth hiện tại bằng cách phân cụm người dùng dựa trên sở thích thể loại phim (Genres), sau đó khai phá các luật kết hợp chuyên sâu cho từng cụm (Cluster). Phương pháp này giúp khắc phục nhược điểm "trung bình hóa sở thích" của FP-Growth truyền thống.

## User Review Required

> [!IMPORTANT]
> Phương pháp này sẽ thay đổi luồng hoạt động của hệ thống gợi ý.
> Khi một user chọn 3 phim yêu thích ban đầu (seed), hệ thống sẽ phân tích thể loại của 3 phim đó để **xếp user vào một cụm (ví dụ: Cụm Fan Hành Động, Cụm Fan Hài hước)**. Sau đó, nó mới nạp bộ luật FP-Growth riêng của cụm đó ra để tính toán gợi ý.
> Bạn xem qua cấu trúc này xem có phù hợp với yêu cầu của đồ án Data Mining không nhé.

## Open Questions

1. **Số lượng cụm (k):** Hiện tại mình dự tính chia làm `k = 4` cụm người dùng. Nếu bạn muốn số khác (ví dụ 3 hoặc 5) thì báo mình nhé.
2. **File Training Mới:** Mình sẽ tạo một file script mới là `train_cluster_fpgrowth.py` thay vì ghi đè lên file cũ của bạn, để bạn có 2 phiên bản (Basic và Nâng cao) báo cáo cho thầy cô. Bạn đồng ý chứ?

## Proposed Changes

### `model/` (Hệ thống Huấn luyện)

#### [NEW] `model/train_cluster_fpgrowth.py`
- Sẽ đọc `ratings.csv` và `movies.csv`.
- **Bước 1 (Feature Extraction):** Xây dựng ma trận Người dùng - Thể loại (User-Genre Matrix). Tức là tính xem mỗi user xem bao nhiêu % phim Hành động, bao nhiêu % phim Hài...
- **Bước 2 (Clustering):** Chạy thuật toán `K-Means (k=4)` từ thư viện `scikit-learn` trên ma trận trên để chia người dùng làm 4 cụm. Lưu model K-Means thành `kmeans_model.joblib`.
- **Bước 3 (Khai phá luật):** Lặp qua 4 cụm. Với mỗi cụm, lấy ra tập ratings của user trong cụm đó và chạy `fpgrowth` + `association_rules`. Lưu 4 bộ luật thành `fpgrowth_cluster_0.joblib` đến `fpgrowth_cluster_3.joblib`.

### `recommender.py` (Hệ thống Dự đoán)

#### [MODIFY] `recommender.py`
- Viết thêm hàm `predict_cluster_from_seed(seed_movie_ids)`: Hàm này nhận vào các phim user thích, biến đổi thành vector thể loại (giống Bước 1) và đưa vào K-Means để đoán xem user thuộc Cụm nào.
- Sửa đổi hàm `recommend_from_seed`:
  - Đầu tiên, gọi K-Means để lấy số `cluster_id`.
  - Thứ hai, load `fpgrowth_cluster_{cluster_id}.joblib` thay vì load file chung.
  - Sau đó áp dụng cách tính score + penalty factor như bình thường.

## Verification Plan

### Phân tích kết quả (Viết báo cáo)
- Chạy file `train_cluster_fpgrowth.py`.
- Lấy thông tin thống kê của từng cụm (ví dụ: Cụm 0 có bao nhiêu user, thể loại thống trị là gì, khai phá ra bao nhiêu luật).
- Mình sẽ viết một file Báo cáo chi tiết (theo đúng yêu cầu "làm như nào, làm ra sao, kết quả thế nào" của bạn) trong thư mục `reports/cluster_report.md` để bạn dùng dán thẳng vào slide hoặc báo cáo môn học.
