# Data Mining Movie Recommender - Cluster-Based FP-Growth (Backend/CLI Version)

Project xây dựng hệ gợi ý phim nâng cao kết hợp giữa **Phân cụm (K-Means Clustering)** và **Luật kết hợp (FP-Growth)**. Đây là phiên bản Backend/CLI tập trung hoàn toàn vào xử lý mô hình, không chứa giao diện Web.

## Ý tưởng & Kiến trúc

- **Tiền xử lý (Data Cleaning) & Migration**: Làm sạch dữ liệu, xử lý missing values từ bộ dữ liệu thô và import toàn bộ vào CSDL **SQLite3** (`database.db`) để tối ưu hiệu năng và xử lý đa luồng.
- **Sinh dữ liệu & EDA**: Trực tiếp truy vấn CSDL để lấy danh sách đánh giá tốt (rating >= 4.0), sinh dữ liệu người dùng mô phỏng và tự động vẽ các biểu đồ phân tích (EDA).
- **Phân cụm người dùng (K-Means)**: Hệ thống gom nhóm người dùng thành 4 cụm (Clusters) dựa trên ma trận sở thích thể loại phim (Genres).
- **Khai phá luật (FP-Growth)**: Huấn luyện độc lập 4 mô hình FP-Growth cho 4 cụm. Các luật sinh ra (vd: `Thích phim A => Thích phim B`) sẽ mang tính đặc trưng riêng cho từng nhóm đối tượng.
- **Cơ chế Penalty Factor**: Áp dụng công thức phạt (Log-penalty) đối với các phim quá phổ biến (Popularity Bias) nhằm đa dạng hóa danh mục gợi ý.

## Cấu trúc File chính

```text
data_preprocessing.py           Làm sạch dữ liệu và khởi tạo CSDL SQLite3 (database.db)
data_generation.py              Sinh dữ liệu User vào DB và vẽ biểu đồ EDA
model/train_cluster_fpgrowth.py Huấn luyện cụm K-Means và các tập luật FP-Growth
model/evaluate_model.py         Kịch bản đo lường (Hit Rate, Precision, Recall, Coverage) & A/B Testing
recommender.py                  Core Engine: Dự đoán cụm, query luật, tính điểm & penalty
test_user.py                    CLI Tool để phân tích, debug quá trình gợi ý cho 1 User bất kỳ
```

## Hướng dẫn chạy

**Bước 1: Tiền xử lý & Sinh dữ liệu (Bắt buộc chạy lần đầu)**
```bash
python data_preprocessing.py
python data_generation.py
```

**Bước 2: Huấn luyện Mô hình**
```bash
python model/train_cluster_fpgrowth.py
```

**Bước 3: Đánh giá & Phân tích**
```bash
# Xem báo cáo độ chính xác tổng quan và A/B Testing Penalty
python model/evaluate_model.py

# Xem chi tiết mô phỏng gợi ý cho một User cụ thể
python test_user.py --user 1
```
