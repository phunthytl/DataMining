# CineRec — Flask Web Interface

Giao diện web cho hệ thống gợi ý phim Cluster-Based FP-Growth.

## Cấu trúc thư mục (đặt trong project root)

```
DataMining/
├── data/                        ← dữ liệu gốc
├── model/                       ← model đã train
├── reports/                     ← báo cáo EDA
├── recommender.py               ← core engine 
├── flask_app/                  
│   ├── app.py
│   ├── requirements.txt
│   └── templates/
│       ├── base.html
│       ├── login.html
│       ├── register.html
│       ├── onboarding.html
│       └── home.html
```

## Cài đặt & Chạy

```bash
# 1. Cài dependencies (nếu chưa có flask)
pip install flask

# 2. Đảm bảo đã chạy pipeline trước:
python data_preprocessing.py
python data_generation.py
python model/train_cluster_fpgrowth.py

# 3. Chạy Flask (từ thư mục flask_app/)
cd flask_app
python app.py
```

Mở trình duyệt: http://localhost:5000

## Tính năng

### Auth
- **Đăng ký**: Tạo tài khoản mới, lưu vào `data/users.csv`
- **Đăng nhập**: Xác thực bằng account + password từ `users.csv`
- **Đăng xuất**: Xóa session

### Onboarding (xuất hiện sau khi đăng ký)
- Chọn thể loại phim yêu thích (multi-select chips)
- Hệ thống lưu seed movies tương ứng với thể loại đã chọn
- Có thể **bỏ qua (Skip)** → vào thẳng trang chủ

### Trang chủ
- **Navbar**: Logo, thanh tìm kiếm live search, tên user, nút đăng xuất
- **Hero banner**: Hiển thị trạng thái gợi ý (cá nhân hóa hay phổ biến)
- **Cluster badge**: Cho biết user đang ở nhóm nào (Cluster #N)
- **Genre filter bar**: Lọc phim theo thể loại, scroll ngang
- **Movie grid**: Responsive, 24 phim, có poster/placeholder, rating, thể loại
- **Score bar**: Thanh điểm gợi ý (chỉ hiện với phim được recommend)
- **Modal popup**: Click vào phim → popup chi tiết (poster, genres, rating, điểm, overview)
- **Fallback logic**:
  1. Nếu có lịch sử liked → dùng `recommend_from_seed()`
  2. Nếu có onboarding genres → dùng seed từ thể loại đã chọn
  3. Nếu chưa có gì → hiển thị `popular_movies()`
  4. Nếu model chưa train → fallback `movies.sort_values('avg_rating')`

### Search
- Live search (debounce 280ms) → gọi `/api/search?q=...`
- Hiển thị dropdown, click vào mở modal chi tiết phim