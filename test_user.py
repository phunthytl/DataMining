import argparse
import sys
from pathlib import Path
import random

import pandas as pd

# Đảm bảo hiển thị Tiếng Việt tốt trên terminal Windows
sys.stdout.reconfigure(encoding='utf-8')

from recommender import (
    user_history_seed, 
    predict_cluster_from_seed, 
    recommend_from_seed, 
    load_movies,
    load_liked_ratings
)

def test_single_user(user_id=None):
    print("Đang tải dữ liệu...")
    liked = load_liked_ratings()
    
    if user_id is None:
        user_counts = liked['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 10].index.tolist()
        user_id = random.choice(valid_users)
        print(f"[*] Bạn không truyền User ID. Chọn ngẫu nhiên User: {user_id}")
    else:
        print(f"[*] Đang phân tích User: {user_id}")
        if user_id not in liked['user_id'].values:
            print("Lỗi: Không tìm thấy người dùng này trong cơ sở dữ liệu (Database).")
            return

    seed_movie_ids = user_history_seed(user_id, limit=10)
    
    if not seed_movie_ids:
        print("Người dùng này chưa có lịch sử xem phim (hoặc không hợp lệ).")
        return

    # Dự đoán cụm (Cluster)
    cluster_id = predict_cluster_from_seed(seed_movie_ids)
    
    # Load thông tin phim
    movies = load_movies()
    # Lấy thông tin chi tiết lịch sử để in ra
    # Giữ nguyên thứ tự theo seed_movie_ids
    seed_movies = movies[movies['movie_id'].isin(seed_movie_ids)].set_index('movie_id').loc[seed_movie_ids].reset_index()
    
    print("\n" + "="*80)
    print(f"🎥 LỊCH SỬ XEM PHIM NỔI BẬT (SEED) CỦA USER {user_id}:")
    print("="*80)
    for idx, row in seed_movies.iterrows():
        print(f" - [ID: {row['movie_id']:<6}] {row['title']} (Thể loại: {row['genres']})")
        
    print("\n" + "="*80)
    print(f"🔮 KẾT QUẢ PHÂN CỤM (K-MEANS): USER NÀY THUỘC VỀ >> CỤM SỐ {cluster_id} <<")
    print("="*80)
    
    # Lấy gợi ý
    print("\nĐang quét tập luật FP-Growth của Cụm tương ứng...")
    recs = recommend_from_seed(seed_movie_ids, top_k=10, use_penalty=True)
    
    if recs.empty:
        print("Không tìm thấy luật kết hợp phù hợp. Fallback sang danh sách phim phổ biến.")
    else:
        print("\n" + "="*80)
        print(f"🌟 TOP 10 GỢI Ý DÀNH CHO USER {user_id} (Dựa trên Cluster {cluster_id}):")
        print("="*80)
        for idx, row in recs.iterrows():
            print(f"{idx+1:02d}. [ID: {row['movie_id']:<6}] {row['title']}")
            print(f"    -> Thể loại: {row['genres']}")
            print(f"    -> Mức độ phù hợp (Norm Score): {row.get('normalized_score', 0):.4f} | Độ tin cậy: {row.get('confidence', 0)*100:.1f}%")
            print(f"    -> Gợi ý từ (các) phim đã xem: {row.get('matched_from', [])}")
            print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Recommendation Logic for a Single User")
    parser.add_argument("--user", type=int, help="ID của người dùng cần kiểm tra", default=None)
    args = parser.parse_args()
    
    test_single_user(args.user)
