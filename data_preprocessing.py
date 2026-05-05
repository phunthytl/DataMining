import os
import sys
import sqlite3
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "database.db"

def preprocess_datasets():
    print("Bắt đầu khởi tạo và xử lý dữ liệu vào SQLite...")
    
    # Kết nối DB
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Xử lý movies.csv
    print("Đang xử lý movies.csv...")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    movies.drop_duplicates(subset=["movie_id"], keep="first", inplace=True)
    movies["poster_url"] = movies["poster_url"].fillna("")
    movies["overview"] = movies["overview"].fillna("Không có thông tin mô tả.")
    
    # 2. Xử lý ratings.csv
    print("Đang xử lý ratings.csv...")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    ratings.drop_duplicates(inplace=True)
    ratings = ratings[(ratings["rating"] >= 1.0) & (ratings["rating"] <= 5.0)]
    
    # Tính toán lại num_ratings và avg_rating từ data thực tế để đảm bảo chính xác
    stats = ratings.groupby("movie_id").agg(
        calc_num_ratings=("rating", "count"),
        calc_avg_rating=("rating", "mean")
    ).reset_index()
    
    movies = movies.merge(stats, on="movie_id", how="left")
    movies["num_ratings"] = movies["calc_num_ratings"].fillna(0).astype(int)
    movies["avg_rating"] = movies["calc_avg_rating"].fillna(0.0).round(2)
    movies.drop(columns=["calc_num_ratings", "calc_avg_rating"], inplace=True)
    movies["tmdb_rating"] = movies["tmdb_rating"].fillna(movies["avg_rating"])
    
    # Đảm bảo tính toàn vẹn dữ liệu
    valid_movie_ids = set(movies["movie_id"])
    ratings = ratings[ratings["movie_id"].isin(valid_movie_ids)]
    
    # 3. Xử lý users.csv
    print("Đang xử lý users.csv...")
    if (DATA_DIR / "users.csv").exists():
        users = pd.read_csv(DATA_DIR / "users.csv")
    else:
        users = pd.DataFrame(columns=["user_id", "account", "password", "birth_year", "gender", "favorite_movies"])

    # 4. Lưu vào SQLite
    print("Đang lưu dữ liệu vào bảng movies...")
    movies.to_sql("movies", conn, if_exists="replace", index=False)
    
    print("Đang lưu dữ liệu vào bảng ratings...")
    ratings.to_sql("ratings", conn, if_exists="replace", index=False)
    
    print("Đang lưu dữ liệu vào bảng users...")
    users.to_sql("users", conn, if_exists="replace", index=False)
    
    conn.commit()
    conn.close()
    
    print(f"Hoàn thành khởi tạo DB tại {DB_PATH}!")

if __name__ == "__main__":
    preprocess_datasets()
