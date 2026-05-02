from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"

MIN_RATING = 4.0
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.12
MAX_RULES = 50000
TOP_POPULAR = 500


def build_transaction_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    liked = ratings[ratings["rating"] >= MIN_RATING].copy()
    top_items = (
        liked["movie_id"].astype(int)
        .value_counts()
        .head(1200)
        .index
        .astype(int)
        .tolist()
    )
    liked = liked[liked["movie_id"].astype(int).isin(top_items)]
    baskets = liked.groupby("user_id")["movie_id"].apply(lambda s: [int(x) for x in s.tolist() if pd.notna(x)]).tolist()
    baskets = [b for b in baskets if len(b) >= 2]
    te = TransactionEncoder()
    arr = te.fit(baskets).transform(baskets, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(arr, columns=[str(c) for c in te.columns_])
    return df


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    txn = build_transaction_matrix(ratings)
    frequent_itemsets = fpgrowth(txn, min_support=MIN_SUPPORT, use_colnames=True, max_len=2)
    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(lambda s: frozenset(int(x) for x in s))
    rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules_df = rules_df.sort_values(["lift", "confidence", "support"], ascending=False).head(MAX_RULES).copy()

    rules = []
    index: dict[int, list[dict]] = defaultdict(list)
    for _, row in rules_df.iterrows():
        antecedent = sorted(int(x) for x in row["antecedents"])
        consequent = sorted(int(x) for x in row["consequents"])
        support = float(row["support"])
        confidence = float(row["confidence"])
        lift = float(row["lift"])
        rule = {
            "antecedent": antecedent,
            "consequent": consequent,
            "support": support,
            "confidence": confidence,
            "lift": lift,
            "support_count": int(round(support * len(txn))),
        }
        rules.append(rule)
        if len(consequent) == 1:
            target = consequent[0]
            for movie_id in antecedent:
                index[movie_id].append({
                    "target": target,
                    "antecedent": antecedent,
                    "confidence": confidence,
                    "lift": lift,
                    "support_count": rule["support_count"],
                })

    for movie_id, rows in index.items():
        rows.sort(key=lambda x: (x["lift"], x["confidence"], x["support_count"]), reverse=True)
        index[movie_id] = rows[:300]

    popularity = (
        ratings[ratings["rating"] >= MIN_RATING]
        .groupby("movie_id")
        .agg(liked_count=("rating", "size"), avg_rating=("rating", "mean"))
        .reset_index()
        .sort_values(["liked_count", "avg_rating"], ascending=False)
    )

    artifact = {
        "algorithm": "mlxtend FP-Growth + association rules",
        "min_rating": MIN_RATING,
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "n_transactions": len(txn),
        "frequent_itemsets": frequent_itemsets.to_dict("records"),
        "rules": rules,
        "recommendation_index": index,
        "popular_movies": popularity.head(TOP_POPULAR)[["movie_id", "liked_count", "avg_rating"]].to_dict("records"),
    }
    joblib.dump(artifact, MODEL_DIR / "fpgrowth_rules.joblib", compress=3)

    report = {
        "algorithm": artifact["algorithm"],
        "transactions": len(txn),
        "frequent_itemsets": len(frequent_itemsets),
        "rules": len(rules),
        "min_rating": MIN_RATING,
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
    }
    (REPORT_DIR / "fpgrowth_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    train()
