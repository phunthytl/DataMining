import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def build_user_genre_matrix() -> pd.DataFrame:
    print("Loading data...")
    liked = pd.read_csv(DATA_DIR / "liked.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")

    # Merge to get genres
    df = liked.merge(movies[["movie_id", "genres"]], on="movie_id")

    print("Building user genre profiles...")
    # Explode genres (e.g., "Action|Comedy" -> ["Action", "Comedy"])
    df["genre_list"] = df["genres"].fillna("").str.split("|")
    df_exploded = df.explode("genre_list")
    df_exploded = df_exploded[df_exploded["genre_list"] != ""]

    # Count genres per user
    user_genre_counts = df_exploded.groupby(["user_id", "genre_list"]).size().unstack(fill_value=0)

    # Normalize counts to get percentages (profile)
    user_genre_profile = user_genre_counts.div(user_genre_counts.sum(axis=1), axis=0)

    return user_genre_profile

def find_optimal_k():
    user_genre_profile = build_user_genre_matrix()
    
    print("Running KMeans from K=1 to 10...")
    wcss = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(user_genre_profile)
        wcss.append(kmeans.inertia_)
        print(f"K = {k}: WCSS = {kmeans.inertia_:.2f}")

    # Plot
    print("Saving elbow plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
    plt.title('Phương pháp Elbow để tìm số cụm (K) tối ưu')
    plt.xlabel('Số cụm (K)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.xticks(k_range)
    plt.grid(True)
    
    plot_path = REPORT_DIR / "elbow_plot.png"
    plt.savefig(plot_path)
    print(f"Đã lưu đồ thị tại: {plot_path}")

if __name__ == "__main__":
    find_optimal_k()
