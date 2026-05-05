import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import sqlite3
import time
from pathlib import Path

from model.train_cluster_fpgrowth import train_cluster_model

DATA_DIR = Path("data")
MODEL_DIR = Path("model")
DB_PATH = DATA_DIR / "database.db"
TRAIN_INFO_PATH = MODEL_DIR / "train_info.json"

# Ngưỡng số lượng lượt rating mới để quyết định có retrain hay không
RETRAIN_THRESHOLD = 1000

# Ngưỡng thời gian để bắt buộc retrain (tính bằng ngày)
RETRAIN_DAYS_THRESHOLD = 7
RETRAIN_TIME_SECONDS = RETRAIN_DAYS_THRESHOLD * 24 * 60 * 60

def get_current_ratings_count():
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ratings")
    count = cur.fetchone()[0]
    conn.close()
    return count

def check_and_retrain(force=False):
    print("--- Kiểm tra điều kiện Retrain ---")
    current_ratings = get_current_ratings_count()
    
    last_train_ratings = 0
    last_train_timestamp = 0
    
    if TRAIN_INFO_PATH.exists():
        try:
            info = json.loads(TRAIN_INFO_PATH.read_text(encoding='utf-8'))
            last_train_ratings = info.get("last_train_ratings", 0)
            last_train_timestamp = info.get("last_train_timestamp", 0)
        except Exception:
            pass
            
    new_ratings = current_ratings - last_train_ratings
    time_elapsed = time.time() - last_train_timestamp if last_train_timestamp > 0 else 0
    days_elapsed = time_elapsed / (24 * 60 * 60)
    
    print(f"Tổng số ratings hiện tại: {current_ratings}")
    print(f"Số lượng ratings mới thêm vào: {new_ratings} (Ngưỡng: {RETRAIN_THRESHOLD})")
    
    if last_train_timestamp > 0:
        print(f"Thời gian từ lần train cuối: {days_elapsed:.2f} ngày (Ngưỡng: {RETRAIN_DAYS_THRESHOLD} ngày)")
    else:
        print("Chưa có thông tin lịch sử train.")
    
    is_enough_data = new_ratings >= RETRAIN_THRESHOLD
    is_time_to_retrain = last_train_timestamp > 0 and time_elapsed >= RETRAIN_TIME_SECONDS
    is_first_time = last_train_ratings == 0
    
    if force or is_enough_data or is_time_to_retrain or is_first_time:
        if force:
            print("=> Chế độ BẮT BUỘC (Force Retrain). Đang tiến hành huấn luyện lại...")
        elif is_enough_data:
            print(f"=> Dữ liệu mới ({new_ratings}) đạt ngưỡng. Đang tiến hành huấn luyện lại...")
        elif is_time_to_retrain:
            print(f"=> Đã đến lịch retrain định kỳ (> {RETRAIN_DAYS_THRESHOLD} ngày). Đang tiến hành huấn luyện lại...")
        else:
            print("=> Lần train đầu tiên. Đang tiến hành huấn luyện lại...")
        
        train_cluster_model()
    else:
        print("=> Chưa đạt đủ điều kiện dữ liệu hoặc thời gian định kỳ. Bỏ qua retrain.")

if __name__ == "__main__":
    force_retrain = "--force" in sys.argv
    check_and_retrain(force=force_retrain)
    print("--- HOÀN THÀNH ---")