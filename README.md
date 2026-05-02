# Data Mining Movie Recommender - FP-Growth

Project xây dựng hệ gợi ý phim bằng **luật khai phá kết hợp** theo hướng FP-Growth/frequent itemset mining.

## Ý tưởng

- Mỗi user là một giao dịch gồm các phim user đã đánh giá cao (`rating >= 4`).
- Khai phá các tập phim thường xuất hiện cùng nhau.
- Sinh luật kết hợp dạng:

```text
Nếu thích phim A => có khả năng thích phim B
```

- Gợi ý phim dựa trên `confidence`, `lift` và `support` của luật.

## Luồng web

- User đăng nhập bằng User ID dataset hoặc tạo user mới.
- User mới có thể chọn tối đa 3 phim yêu thích nhất.
- Có thể skip nếu chưa muốn chọn.
- Nếu có seed movies, hệ thống dùng luật kết hợp để gợi ý.
- Nếu skip, hệ thống hiển thị phim phổ biến làm cold-start fallback.

## Chạy project

```bash
cd D:\Web\Data_Mining
python model\train_fpgrowth.py
python app.py
```

Mở:

```text
http://127.0.0.1:5000
```

## File chính

```text
model/train_fpgrowth.py   Train luật FP-Growth/association rules
recommender.py            Logic gợi ý từ seed movies
app.py                    Flask web demo
templates/                Giao diện web
static/style.css          CSS
```
