# CineRec — Flask Web Interface

Giao diện web cho hệ thống gợi ý phim Cluster-Based FP-Growth.

## Cấu trúc thư mục

```text
DataMining/
├── data/                        Dữ liệu và SQLite database
├── model/                       Model đã train
├── reports/                     Báo cáo EDA/đánh giá
├── recommender.py               Core engine
├── requirements.txt             Dependencies toàn dự án
└── flask_app/
    ├── app.py
    └── templates/
        ├── base.html
        ├── login.html
        ├── register.html
        ├── onboarding.html
        ├── home.html
        ├── favorites.html
        └── movie_detail.html
```

## Cài đặt & chạy

```bash
# 1. Cài dependencies từ project root
pip install -r requirements.txt

# 2. Đảm bảo đã chạy pipeline trước
python data_preprocessing.py
python data_generation.py
python model/train_cluster_fpgrowth.py

# 3. Chạy Flask từ project root
python flask_app/app.py
```

Mở trình duyệt: `http://localhost:5000`

## Tính năng

### Auth

- **Đăng ký**: Tạo tài khoản mới trong bảng `users` của `data/database.db`.
- **Đăng nhập**: Xác thực bằng account + password từ SQLite.
- **Đăng xuất**: Xóa session hiện tại.
- Password của tài khoản mới được lưu dạng hash; dữ liệu user cũ dạng plain text vẫn được hỗ trợ để tương thích demo.

### Onboarding

- Chọn phim yêu thích bằng multi-select.
- Có thể bỏ qua để vào thẳng trang chủ.
- Phim đã chọn được lưu vào cột `favorite_movies` của user.

### Trang chủ

- Navbar có logo, live search, tên user và nút đăng xuất.
- Hero banner hiển thị trạng thái gợi ý cá nhân hóa hoặc phổ biến.
- Cluster badge cho biết user thuộc cụm nào khi có seed gợi ý.
- Genre filter bar, phân trang danh sách phim và movie grid responsive.
- Modal chi tiết phim với poster, genres, rating, score và overview.
- Fallback logic:
  1. Nếu có lịch sử liked hoặc onboarding seed → dùng `recommend_from_seed()`.
  2. Nếu chưa có seed → hiển thị `popular_movies()`.
  3. Nếu model chưa train hoặc lỗi recommender → fallback danh sách phim theo `avg_rating`.

### Search

- Live search gọi `/api/search?q=...`.
- Kết quả trả về JSON gồm movie id, title, genres, poster và rating.
