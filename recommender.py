from collections import defaultdict
from pathlib import Path

import math
import joblib
import pandas as pd
import numpy as np

import sqlite3

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DB_PATH = DATA_DIR / "database.db"

def get_db_conn():
    return sqlite3.connect(DB_PATH)

def load_movies() -> pd.DataFrame:
    conn = get_db_conn()
    movies = pd.read_sql("SELECT * FROM movies", conn)
    conn.close()
    movies["poster_url"] = movies.get("poster_url", "").fillna("")
    movies["overview"] = movies.get("overview", "").fillna("")
    movies["avg_rating"]  = movies.get("avg_rating",   pd.Series(dtype=float)).fillna(0.0)
    movies["num_ratings"] = movies.get("num_ratings",  pd.Series(dtype=int)).fillna(0)
    return movies


def load_liked_ratings() -> pd.DataFrame:
    conn = get_db_conn()
    liked = pd.read_sql("SELECT * FROM ratings WHERE rating >= 4.0", conn)
    conn.close()
    return liked


def load_users() -> pd.DataFrame:
    conn = get_db_conn()
    users = pd.read_sql("SELECT * FROM users", conn)
    conn.close()
    return users


def load_kmeans_and_genres() -> tuple:
    kmeans = joblib.load(MODEL_DIR / "kmeans_model.joblib")
    genre_columns = joblib.load(MODEL_DIR / "genre_columns.joblib")
    return kmeans, genre_columns


def load_cluster_model(cluster_id: int) -> dict:
    try:
        return joblib.load(MODEL_DIR / f"fpgrowth_cluster_{cluster_id}.joblib")
    except FileNotFoundError:
        return joblib.load(MODEL_DIR / "fpgrowth_rules.joblib")


def predict_cluster_from_seed(seed_movie_ids: list[int]) -> int:
    if not seed_movie_ids:
        return 0
        
    try:
        kmeans, genre_columns = load_kmeans_and_genres()
    except FileNotFoundError:
        return 0 # Fallback
        
    movies = load_movies()
    seed_movies = movies[movies["movie_id"].astype(int).isin(seed_movie_ids)]
    
    genres = seed_movies["genres"].fillna("").str.split("|").explode()
    genres = genres[genres != ""]
    
    if genres.empty:
        return 0
        
    counts = genres.value_counts()
    profile = counts / counts.sum()
    
    vector = []
    for g in genre_columns:
        vector.append(profile.get(g, 0.0))
        
    X = np.array([vector])
    cluster_id = kmeans.predict(X)[0]
    return int(cluster_id)


def recommend_from_seed(seed_movie_ids: list[int], top_k: int = 12, use_penalty: bool = True) -> pd.DataFrame:
    cluster_id = predict_cluster_from_seed(seed_movie_ids)
    movies = load_movies()
    seen = set(map(int, seed_movie_ids))
    
    def get_scores(idx_dict):
        scores_dict = defaultdict(lambda: {"score": 0.0, "confidence": 0.0, "lift": 0.0, "support_count": 0, "matched_from": []})
        for m_id in seen:
            for row in idx_dict.get(m_id, []):
                target = int(row["target"])
                if target in seen:
                    continue
                contribution = float(row["confidence"]) * float(row["lift"])
                scores_dict[target]["score"] += contribution
                scores_dict[target]["confidence"] = max(scores_dict[target]["confidence"], float(row["confidence"]))
                scores_dict[target]["lift"] = max(scores_dict[target]["lift"], float(row["lift"]))
                scores_dict[target]["support_count"] = max(scores_dict[target]["support_count"], int(row["support_count"]))
                scores_dict[target]["matched_from"].append(m_id)
        return scores_dict

    artifact = load_cluster_model(cluster_id)
    scores = get_scores(artifact.get("recommendation_index", {}))

    # Fallback 1: Nếu cụm không có dữ liệu, dùng mô hình chung (cluster 0)
    if not scores and cluster_id != 0:
        artifact_0 = load_cluster_model(0)
        scores = get_scores(artifact_0.get("recommendation_index", {}))

    if not scores:
        return pd.DataFrame(columns=["movie_id", "score", "confidence", "lift", "support_count", "matched_from"])

    rows = []
    for movie_id, info in scores.items():
        penalty_factor = math.log10(info["support_count"] + 10) if use_penalty else 1.0
        normalized_score = info["score"] / penalty_factor
        rows.append({"movie_id": movie_id, **info, "normalized_score": normalized_score, "matched_from": sorted(set(info["matched_from"]))})
    
    score_df = pd.DataFrame(rows).sort_values(["normalized_score", "score", "lift", "confidence", "support_count"], ascending=False)
    recs = score_df.head(top_k).merge(movies, on="movie_id", how="left")
    return recs


def popular_movies(top_k: int = 12, exclude: set[int] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    movies = load_movies()
    artifact = load_cluster_model(0)
    pop = pd.DataFrame(artifact["popular_movies"])
    pop = pop[~pop["movie_id"].astype(int).isin(exclude)]
    pop["score"] = pop["liked_count"] * pop["avg_rating"]
    pop["normalized_score"] = pop["score"]
    pop["confidence"] = 0.0
    pop["lift"] = 0.0
    pop["support_count"] = pop["liked_count"]
    pop["matched_from"] = [[] for _ in range(len(pop))]
    
    overlap = set(pop.columns).intersection(set(movies.columns)) - {"movie_id"}
    pop = pop.drop(columns=list(overlap))
    
    return pop.head(top_k).merge(movies, on="movie_id", how="left")


def user_history_seed(user_id: int, limit: int | None = 10) -> list[int]:
    seed_ids = []
    
    liked = load_liked_ratings()
    if not liked.empty and "user_id" in liked.columns:
        rows = liked[liked["user_id"] == user_id].sort_values("rating", ascending=False)
        seed_ids.extend(rows["movie_id"].astype(int).tolist())
        
    users = load_users()
    if not users.empty and "user_id" in users.columns:
        user_row = users[users["user_id"] == user_id]
        if not user_row.empty:
            favs_str = str(user_row.iloc[0].get("favorite_movies", ""))
            if favs_str and favs_str != "nan":
                fav_ids = [int(x) for x in favs_str.split("|") if x.strip().isdigit()]
                seed_ids.extend(fav_ids)
                
    unique_seeds = []
    for sid in seed_ids:
        if sid not in unique_seeds:
            unique_seeds.append(sid)
            
    if limit is not None:
        unique_seeds = unique_seeds[:limit]
        
    return unique_seeds
