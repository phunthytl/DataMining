# Data Mining Movie Recommender - Cluster-Based FP-Growth

Project xây dựng hệ gợi ý phim kết hợp giữa **phân cụm người dùng (K-Means Clustering)** và **khai phá luật kết hợp (FP-Growth)**. Dự án có pipeline xử lý dữ liệu, huấn luyện/đánh giá mô hình, công cụ CLI để kiểm thử gợi ý và giao diện web Flask để demo người dùng.

## Ý tưởng & Kiến trúc

- **Tiền xử lý (Data Cleaning) & Migration**: Làm sạch dữ liệu, xử lý missing values từ bộ dữ liệu thô và import toàn bộ vào CSDL **SQLite3** (`data/database.db`).
- **Sinh dữ liệu & EDA**: Truy vấn CSDL để lấy danh sách đánh giá tốt (rating >= 4.0), sinh dữ liệu người dùng mô phỏng và tự động vẽ các biểu đồ phân tích.
- **Phân cụm người dùng (K-Means)**: Gom nhóm người dùng thành 4 cụm dựa trên ma trận sở thích thể loại phim.
- **Khai phá luật (FP-Growth)**: Huấn luyện độc lập các tập luật FP-Growth cho từng cụm người dùng.
- **Penalty Factor**: Áp dụng log-penalty với phim quá phổ biến để giảm popularity bias và tăng độ đa dạng gợi ý.
- **Flask Web Interface**: Cho phép đăng ký, đăng nhập, onboarding phim yêu thích, xem gợi ý, tìm kiếm phim và quản lý phim yêu thích.

## Cấu trúc file chính

```text
data_preprocessing.py             Làm sạch dữ liệu và khởi tạo SQLite database
data_generation.py                Sinh dữ liệu user và vẽ biểu đồ EDA
model/train_cluster_fpgrowth.py   Huấn luyện K-Means và FP-Growth theo cụm
model/evaluate_model.py           Đánh giá Hit Rate, Precision, Recall, Coverage
recommender.py                    Core engine dự đoán cụm, query luật và tính điểm gợi ý
test_user.py                      CLI phân tích quá trình gợi ý cho một user
flask_app/app.py                  Flask web app demo hệ gợi ý
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Hướng dẫn chạy pipeline

**Bước 1: Tiền xử lý & sinh dữ liệu**

```bash
python data_preprocessing.py
python data_generation.py
```

**Bước 2: Huấn luyện mô hình**

```bash
python model/train_cluster_fpgrowth.py
```

**Bước 3: Đánh giá & phân tích**

```bash
python model/evaluate_model.py
python test_user.py --user 1
```

## Chạy giao diện web

```bash
python flask_app/app.py
```

Mở trình duyệt tại `http://localhost:5000`.

Có thể bật debug khi chạy local bằng biến môi trường:

```bash
FLASK_DEBUG=1 python flask_app/app.py
```

Nếu muốn cấu hình secret key riêng cho session Flask:

```bash
SECRET_KEY="your-secret-key" python flask_app/app.py
```
