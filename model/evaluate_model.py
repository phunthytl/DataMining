import json
import os
import sys
import contextlib
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from recommender import recommend_from_seed, load_liked_ratings, load_movies, popular_movies

REPORT_DIR = ROOT / "reports"

def evaluate_strategy(test_users, liked, total_items, k_list, use_penalty):
    metrics = {k: {'hits': 0, 'precision': [], 'recall': []} for k in k_list}
    all_recommended_items = {k: set() for k in k_list}
    total_evaluated = 0

    desc = f"Đánh giá (Penalty={use_penalty})"
    for user_id in tqdm(test_users, desc=desc):
        user_history = liked[liked['user_id'] == user_id].sort_values(by="rating", ascending=False)
        user_movies = user_history['movie_id'].tolist()

        split_idx = max(int(len(user_movies) * 0.8), len(user_movies) - 2)
        seed_movies = user_movies[:split_idx]
        target_movies = set(user_movies[split_idx:])

        if not target_movies:
            continue

        max_k = max(k_list)

        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            try:
                # Truyền use_penalty vào hàm recommend
                recs_df = recommend_from_seed(seed_movies, top_k=max_k, use_penalty=use_penalty)
                if recs_df.empty:
                    recs_df = popular_movies(top_k=max_k, exclude=set(seed_movies))
                recs = recs_df['movie_id'].tolist()
            except Exception:
                continue

        total_evaluated += 1

        for k in k_list:
            top_k_recs = recs[:k]
            top_k_set = set(top_k_recs)

            all_recommended_items[k].update(top_k_set)
            hits = len(target_movies.intersection(top_k_set))

            if hits > 0:
                metrics[k]['hits'] += 1

            metrics[k]['precision'].append(hits / k if k > 0 else 0)
            metrics[k]['recall'].append(hits / len(target_movies))
            
    # Calculate final metrics
    results = {}
    for k in k_list:
        results[k] = {
            "Hit_Rate": metrics[k]['hits'] / total_evaluated if total_evaluated > 0 else 0,
            "Precision": np.mean(metrics[k]['precision']) if total_evaluated > 0 else 0,
            "Recall": np.mean(metrics[k]['recall']) if total_evaluated > 0 else 0,
            "Coverage_Percent": len(all_recommended_items[k]) / total_items * 100 if total_items > 0 else 0,
            "Unique_Items_Recommended": len(all_recommended_items[k])
        }
    return results, total_evaluated


def compare_models(k_list: List[int] = [5, 10, 15, 20], num_users: int = 1000):
    print("Đang tải dữ liệu...")
    liked = load_liked_ratings()
    movies_df = load_movies()
    total_items = len(movies_df)

    user_counts = liked['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 10].index.tolist()

    np.random.seed(42)
    if num_users and len(valid_users) > num_users:
        test_users = np.random.choice(valid_users, num_users, replace=False)
    else:
        test_users = valid_users

    print(f"Bắt đầu so sánh trên {len(test_users)} người dùng test...")

    # Chạy chiến lược KHÔNG có penalty
    print("\n--- 1. CHẠY THUẬT TOÁN KHÔNG CÓ PENALTY FACTOR ---")
    results_no_penalty, total_evaluated = evaluate_strategy(test_users, liked, total_items, k_list, use_penalty=False)
    
    # Chạy chiến lược CÓ penalty
    print("\n--- 2. CHẠY THUẬT TOÁN CÓ PENALTY FACTOR ---")
    results_with_penalty, _ = evaluate_strategy(test_users, liked, total_items, k_list, use_penalty=True)

    print("\n" + "="*80)
    print(f"BẢNG SO SÁNH KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ({total_evaluated} users)")
    print("="*80)
    
    for k in k_list:
        print(f"\n📍 Top-{k} Recommendations:")
        print(f"  Chỉ số             | KHÔNG Penalty          | CÓ Penalty             | Khác biệt")
        print(f"  -------------------|------------------------|------------------------|-----------------")
        
        hr_no = results_no_penalty[k]['Hit_Rate']
        hr_yes = results_with_penalty[k]['Hit_Rate']
        print(f"  Hit Rate (HR@{k:<2})   | {hr_no:.4f}                 | {hr_yes:.4f}                 | {(hr_yes - hr_no):+.4f}")
        
        pr_no = results_no_penalty[k]['Precision']
        pr_yes = results_with_penalty[k]['Precision']
        print(f"  Precision@{k:<2}      | {pr_no:.4f}                 | {pr_yes:.4f}                 | {(pr_yes - pr_no):+.4f}")
        
        re_no = results_no_penalty[k]['Recall']
        re_yes = results_with_penalty[k]['Recall']
        print(f"  Recall@{k:<2}         | {re_no:.4f}                 | {re_yes:.4f}                 | {(re_yes - re_no):+.4f}")
        
        cov_no = results_no_penalty[k]['Coverage_Percent']
        cov_yes = results_with_penalty[k]['Coverage_Percent']
        print(f"  Catalog Coverage   | {cov_no:05.2f}% ({results_no_penalty[k]['Unique_Items_Recommended']:3d} phim)    | {cov_yes:05.2f}% ({results_with_penalty[k]['Unique_Items_Recommended']:3d} phim)    | {(cov_yes - cov_no):+.2f}%")

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "comparison_metrics_result.json"
    report_path.write_text(json.dumps({
        "num_users": total_evaluated,
        "without_penalty": results_no_penalty,
        "with_penalty": results_with_penalty
    }, indent=4))
    print(f"\n[+] Đã lưu báo cáo so sánh chi tiết tại: {report_path}")

if __name__ == "__main__":
    compare_models(k_list=[5, 10, 15, 20], num_users=1000)
