from collections import defaultdict
from pathlib import Path

import math

import joblib
import pandas as pd

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"


def load_movies() -> pd.DataFrame:
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    movies["poster_url"] = movies.get("poster_url", "").fillna("")
    movies["overview"] = movies.get("overview", "").fillna("")
    return movies


def load_liked_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "liked.csv")


def load_kmeans_and_genres() -> tuple:
    kmeans = joblib.load(MODEL_DIR / "kmeans_model.joblib")
    genre_columns = joblib.load(MODEL_DIR / "genre_columns.joblib")
    return kmeans, genre_columns


def load_cluster_model(cluster_id: int) -> dict:
    try:
        return joblib.load(MODEL_DIR / f"fpgrowth_cluster_{cluster_id}.joblib")
    except FileNotFoundError:
        # Fallback to general model if not found or using old setup
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
    print(f"DEBUG: Predicted Cluster -> {cluster_id}")
    
    movies = load_movies()
    artifact = load_cluster_model(cluster_id)
    index = artifact.get("recommendation_index", {})
    seen = set(map(int, seed_movie_ids))
    scores: dict[int, dict] = defaultdict(lambda: {"score": 0.0, "confidence": 0.0, "lift": 0.0, "support_count": 0, "matched_from": []})

    for movie_id in seen:
        for row in index.get(movie_id, []):
            target = int(row["target"])
            if target in seen:
                continue
            contribution = float(row["confidence"]) * float(row["lift"])
            scores[target]["score"] += contribution
            scores[target]["confidence"] = max(scores[target]["confidence"], float(row["confidence"]))
            scores[target]["lift"] = max(scores[target]["lift"], float(row["lift"]))
            scores[target]["support_count"] = max(scores[target]["support_count"], int(row["support_count"]))
            scores[target]["matched_from"].append(movie_id)

    if not scores:
        return pd.DataFrame(columns=["movie_id", "score", "confidence", "lift", "support_count", "matched_from"])

    rows = []
    for movie_id, info in scores.items():
        # Áp dụng Penalty Factor (Hệ số phạt Logarit) chống thiên vị phim phổ biến
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
    return pop.head(top_k).merge(movies, on="movie_id", how="left")


def user_history_seed(user_id: int, limit: int | None = 10) -> list[int]:
    liked = load_liked_ratings()
    rows = liked[liked["user_id"] == user_id].sort_values("rating", ascending=False)
    if limit is not None:
        rows = rows.head(limit)
    return rows["movie_id"].astype(int).tolist()
