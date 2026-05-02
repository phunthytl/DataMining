import json
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"

MIN_RATING = 4.0
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.20
MAX_RULES = 15000
K_CLUSTERS = 4  # Dựa trên phương pháp Elbow

def extract_user_genre_profile(liked: pd.DataFrame, movies: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = liked.merge(movies[["movie_id", "genres"]], on="movie_id")
    df["genre_list"] = df["genres"].fillna("").str.split("|")
    df_exploded = df.explode("genre_list")
    df_exploded = df_exploded[df_exploded["genre_list"] != ""]
    
    user_genre_counts = df_exploded.groupby(["user_id", "genre_list"]).size().unstack(fill_value=0)
    user_genre_profile = user_genre_counts.div(user_genre_counts.sum(axis=1), axis=0).fillna(0)
    
    return user_genre_profile, list(user_genre_profile.columns)

def build_transaction_matrix(liked: pd.DataFrame) -> pd.DataFrame:
    top_items = liked["movie_id"].astype(int).value_counts().head(1200).index.astype(int).tolist()
    liked = liked[liked["movie_id"].astype(int).isin(top_items)]
    baskets = liked.groupby("user_id")["movie_id"].apply(lambda s: [int(x) for x in s.tolist() if pd.notna(x)]).tolist()
    baskets = [b for b in baskets if len(b) >= 2]
    
    te = TransactionEncoder()
    arr = te.fit(baskets).transform(baskets, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(arr, columns=[str(c) for c in te.columns_])
    return df

def train_fpgrowth_for_cluster(cluster_id: int, liked_subset: pd.DataFrame, popularity: pd.DataFrame):
    print(f"Training FP-Growth cho Cluster {cluster_id} ({len(liked_subset['user_id'].unique())} users)...")
    txn = build_transaction_matrix(liked_subset)
    if len(txn) < 10:
        return
        
    frequent_itemsets = fpgrowth(txn, min_support=MIN_SUPPORT, use_colnames=True, max_len=2)
    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(lambda s: frozenset(int(x) for x in s))
    
    rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules_df = rules_df.sort_values(["lift", "confidence", "support"], ascending=False).head(MAX_RULES).copy()

    index: dict[int, list[dict]] = defaultdict(list)
    for _, row in rules_df.iterrows():
        antecedent = sorted(int(x) for x in row["antecedents"])
        consequent = sorted(int(x) for x in row["consequents"])
        support = float(row["support"])
        confidence = float(row["confidence"])
        lift = float(row["lift"])
        
        target = consequent[0]
        support_count = int(round(support * len(txn)))
        
        for movie_id in antecedent:
            index[movie_id].append({
                "target": target,
                "confidence": confidence,
                "lift": lift,
                "support_count": support_count,
            })

    for movie_id, rows in index.items():
        rows.sort(key=lambda x: (x["lift"], x["confidence"], x["support_count"]), reverse=True)
        index[movie_id] = rows[:150]

    artifact = {
        "cluster_id": cluster_id,
        "n_transactions": len(txn),
        "rules_count": len(rules_df),
        "recommendation_index": index,
        "popular_movies": popularity.to_dict("records"),
    }
    joblib.dump(artifact, MODEL_DIR / f"fpgrowth_cluster_{cluster_id}.joblib", compress=3)
    return len(rules_df)

def train_cluster_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    liked = pd.read_csv(DATA_DIR / "liked.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    
    # 1. Feature Extraction
    print("Trích xuất đặc trưng User-Genre...")
    user_genre_profile, genre_columns = extract_user_genre_profile(liked, movies)
    joblib.dump(genre_columns, MODEL_DIR / "genre_columns.joblib")
    
    # 2. Clustering
    print(f"Chạy K-Means phân thành {K_CLUSTERS} cụm...")
    kmeans = KMeans(n_clusters=K_CLUSTERS, init='k-means++', max_iter=300, n_init=10, random_state=42)
    user_clusters = kmeans.fit_predict(user_genre_profile)
    joblib.dump(kmeans, MODEL_DIR / "kmeans_model.joblib")
    
    user_cluster_map = dict(zip(user_genre_profile.index, user_clusters))
    liked["cluster"] = liked["user_id"].map(user_cluster_map)
    
    # Popularity (chung cho tất cả, fallback)
    popularity = (
        liked
        .groupby("movie_id")
        .agg(liked_count=("rating", "size"), avg_rating=("rating", "mean"))
        .reset_index()
        .sort_values(["liked_count", "avg_rating"], ascending=False)
        .head(300)
    )
    
    # 3. Train FP-Growth cho từng cụm
    cluster_stats = []
    for cluster_id in range(K_CLUSTERS):
        subset = liked[liked["cluster"] == cluster_id]
        if subset.empty:
            continue
            
        rules_count = train_fpgrowth_for_cluster(cluster_id, subset, popularity)
        cluster_stats.append({
            "cluster_id": cluster_id,
            "user_count": len(subset["user_id"].unique()),
            "rules_generated": rules_count
        })
        
    print("\n--- Báo cáo kết quả Phân cụm ---")
    for stat in cluster_stats:
        print(f"Cụm {stat['cluster_id']}: {stat['user_count']} users, {stat['rules_generated']} luật kết hợp.")
    
    (REPORT_DIR / "cluster_fpgrowth_metrics.json").write_text(json.dumps(cluster_stats, indent=2))

if __name__ == "__main__":
    train_cluster_model()
