from pathlib import Path
import pandas as pd

from model.train_cluster_fpgrowth import train_cluster_model

DATA_DIR = Path("data")

def preprocess():
    print("---Preprocessing---")
    
    ratings_path = DATA_DIR / "ratings.csv"
    liked_path = DATA_DIR / "liked.csv"
    
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)
        liked = ratings[ratings["rating"] >= 4.0]
        liked.to_csv(liked_path, index=False)
        print("Updated liked.csv")

def retrain():
    print("---Retraining model---")
    train_cluster_model()

if __name__ == "__main__":
    preprocess()
    retrain()
    print("--- DONE ---")




    