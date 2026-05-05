import os
import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

def preprocess_datasets():
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    
    # 1. Xử lý ratings.csv
    print(f"Xử lý ratings.csv (Ban đầu: {len(ratings)} dòng)...")
    ratings.drop_duplicates(inplace=True)
    ratings = ratings[(ratings["rating"] >= 1.0) & (ratings["rating"] <= 5.0)]
    
    # 2. Xử lý movies.csv
    movies.drop_duplicates(subset=["movie_id"], keep="first", inplace=True)
    
    # Xử lý missing values
    movies["poster_url"] = movies["poster_url"].fillna("")
    movies["overview"] = movies["overview"].fillna("Không có thông tin mô tả.")
    
    # Tính toán lại num_ratings và avg_rating từ data thực tế để đảm bảo chính xác
    stats = ratings.groupby("movie_id").agg(
        calc_num_ratings=("rating", "count"),
        calc_avg_rating=("rating", "mean")
    ).reset_index()
    
    movies = movies.merge(stats, on="movie_id", how="left")
    movies["num_ratings"] = movies["calc_num_ratings"].fillna(0).astype(int)
    movies["avg_rating"] = movies["calc_avg_rating"].fillna(0.0).round(2)
    movies.drop(columns=["calc_num_ratings", "calc_avg_rating"], inplace=True)
    
    # Missing tmdb_rating lấy bằng avg_rating
    movies["tmdb_rating"] = movies["tmdb_rating"].fillna(movies["avg_rating"])
    
    # 3. Đảm bảo tính toàn vẹn dữ liệu
    valid_movie_ids = set(movies["movie_id"])
    ratings = ratings[ratings["movie_id"].isin(valid_movie_ids)]
    
    movies_clean_path = DATA_DIR / "movies_clean.csv"
    ratings_clean_path = DATA_DIR / "ratings_clean.csv"
    
    movies.to_csv(movies_clean_path, index=False)
    ratings.to_csv(ratings_clean_path, index=False)
    
    print(f"Hoàn thành tiền xử lý!")
    
    return ratings, movies

if __name__ == "__main__":
    preprocess_datasets()
