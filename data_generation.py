import os
import sys
import random
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def generate_password(length=6):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def generate_users_and_liked(ratings: pd.DataFrame, movies: pd.DataFrame):
    print("--- Sinh dữ liệu User và File Liked ---")
    user_ids = ratings["user_id"].unique()
    num_users = len(user_ids)
    
    print(f"Đang tạo users.csv cho {num_users} users...")
    
    # Generate Gender (50% Male, 40% Female, 10% Other)
    genders = np.random.choice(["Male", "Female", "Other"], size=num_users, p=[0.5, 0.4, 0.1])
    
    # Generate Birth Year
    birth_years = np.random.randint(1950, 2010, size=num_users)
    
    # Generate Accounts & Passwords
    accounts = [f"user_{uid:06d}" for uid in user_ids]
    passwords = [generate_password() for _ in range(num_users)]
    
    # Lọc file liked (rating >= 4.0) và lưu
    liked_ratings = ratings[ratings["rating"] >= 4.0]
    liked_path = DATA_DIR / "liked.csv"
    liked_ratings.to_csv(liked_path, index=False)
    print(f"Đã lưu liked.csv với {len(liked_ratings)} lượt đánh giá >= 4.0.")
    
    # Tạo danh sách phim yêu thích ngẫu nhiên (tối đa 3) từ danh sách liked của chính user đó
    user_liked_movies = liked_ratings.groupby("user_id")["movie_id"].apply(list).to_dict()
    favorite_movies_list = []
    
    for uid in user_ids:
        liked_movies = user_liked_movies.get(uid, [])
        if len(liked_movies) >= 3:
            favs = random.sample(liked_movies, 3)
            favorite_movies_list.append("|".join(map(str, favs)))
        else:
            favorite_movies_list.append("")
            
    users_df = pd.DataFrame({
        "user_id": user_ids,
        "account": accounts,
        "password": passwords,
        "birth_year": birth_years,
        "gender": genders,
        "favorite_movies": favorite_movies_list
    })
    
    users_path = DATA_DIR / "users.csv"
    users_df.to_csv(users_path, index=False)
    print(f"Đã lưu thành công users.csv.")
    
    return users_df

def draw_descriptive_stats(users, ratings, movies):
    print("--- Vẽ biểu đồ (EDA) ---")
    sns.set_theme(style="whitegrid")
    
    # 1. Gender Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=users, x="gender", order=["Male", "Female", "Other"])
    plt.title("Phân bố Giới tính Người dùng")
    plt.xlabel("Giới tính")
    plt.ylabel("Số lượng")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    plt.savefig(REPORT_DIR / "gender_distribution.png")
    plt.close()
    
    # 2. Rating Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=ratings, x="rating")
    plt.title("Phân bố Điểm đánh giá (Ratings)")
    plt.xlabel("Điểm đánh giá")
    plt.ylabel("Số lượng")
    plt.savefig(REPORT_DIR / "rating_distribution.png")
    plt.close()
    
    # 3. Top 10 Genres
    genres_exploded = movies["genres"].fillna("").str.split("|").explode()
    genres_exploded = genres_exploded[genres_exploded != ""]
    top_genres = genres_exploded.value_counts().head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_genres.values, y=top_genres.index)
    plt.title("Top 10 Thể loại phim phổ biến nhất")
    plt.xlabel("Số lượng phim")
    plt.ylabel("Thể loại")
    plt.savefig(REPORT_DIR / "top_genres.png")
    plt.close()
    
    # 4. User Age
    users["age"] = 2026 - users["birth_year"]
    plt.figure(figsize=(10, 6))
    sns.histplot(users["age"], bins=20, kde=True, color="coral")
    plt.title("Phân bố Độ tuổi Người dùng")
    plt.xlabel("Độ tuổi")
    plt.ylabel("Số lượng")
    plt.savefig(REPORT_DIR / "age_distribution.png")
    plt.close()
    
    print(f"Đã lưu các biểu đồ tại {REPORT_DIR}")

if __name__ == "__main__":
    print("Đang đọc dữ liệu đã tiền xử lý...")
    try:
        # Ưu tiên đọc file clean nếu có, không thì đọc file gốc
        if (DATA_DIR / "ratings_clean.csv").exists():
            ratings_df = pd.read_csv(DATA_DIR / "ratings_clean.csv")
            movies_df = pd.read_csv(DATA_DIR / "movies_clean.csv")
        else:
            ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")
            movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu (ratings.csv hoặc movies.csv).")
        exit(1)
        
    users_df = generate_users_and_liked(ratings_df, movies_df)
    draw_descriptive_stats(users_df, ratings_df, movies_df)
