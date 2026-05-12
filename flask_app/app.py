import sys
import os
import sqlite3
from pathlib import Path
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DB_PATH = DATA_DIR / "database.db"

sys.path.insert(0, str(ROOT))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "cinemadb_dev_secret_key")

# ── Helpers ───────────────────────────────────────────────────────────────────

def password_matches(stored_password: str, candidate_password: str) -> bool:
    if stored_password.startswith(("scrypt:", "pbkdf2:", "argon2:")):
        return check_password_hash(stored_password, candidate_password)
    return stored_password == candidate_password


def parse_birth_year(value: str) -> int:
    try:
        birth_year = int(value)
    except (TypeError, ValueError):
        return 2000
    return birth_year if 1900 <= birth_year <= 2026 else 2000


def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_all_genres() -> list:
    conn = sqlite3.connect(DB_PATH) # Không dùng row_factory để trả df
    df = pd.read_sql("SELECT genres FROM movies WHERE genres IS NOT NULL", conn)
    conn.close()
    genres = df["genres"].str.split("|").explode()
    return sorted(genres[genres != ""].unique().tolist())


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("home") if "user_id" in session else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        account  = request.form.get("account", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db_conn()
        user = conn.execute("SELECT * FROM users WHERE account = ?", (account,)).fetchone()
        conn.close()

        if user and password_matches(user["password"], password):
            session["user_id"] = user["user_id"]
            session["account"] = user["account"]
            return redirect(url_for("home"))
        error = "Tài khoản hoặc mật khẩu không đúng."
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        account    = request.form.get("account", "").strip()
        password   = request.form.get("password", "").strip()
        gender     = request.form.get("gender", "Other")
        birth_year = parse_birth_year(request.form.get("birth_year", "2000"))
        if not account or not password:
            error = "Vui lòng điền đầy đủ thông tin."
        else:
            conn = get_db_conn()
            existing = conn.execute("SELECT user_id FROM users WHERE account = ?", (account,)).fetchone()
            if existing:
                error = "Tài khoản đã tồn tại."
                conn.close()
            else:
                max_id = conn.execute("SELECT MAX(user_id) FROM users").fetchone()[0]
                new_id = int(max_id) + 1 if max_id is not None else 1
                conn.execute(
                    "INSERT INTO users (user_id, account, password, birth_year, gender, favorite_movies) VALUES (?, ?, ?, ?, ?, ?)",
                    (new_id, account, generate_password_hash(password), birth_year, gender, "")
                )
                conn.commit()
                conn.close()
                session["user_id"] = new_id
                session["account"] = account
                session["new_user"] = True
                return redirect(url_for("onboarding"))
    return render_template("register.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Onboarding ────────────────────────────────────────────────────────────────

@app.route("/onboarding")
@login_required
def onboarding():
    conn = sqlite3.connect(DB_PATH)
    try:
        movies_df = pd.read_sql("SELECT * FROM movies ORDER BY num_ratings DESC LIMIT 200", conn)
    except Exception:
        movies_df = pd.read_sql("SELECT * FROM movies LIMIT 200", conn)
    conn.close()
    
    return render_template("onboarding.html", 
                           genres=get_all_genres(), 
                           top_movies=movies_df.to_dict("records"))


@app.route("/onboarding/save", methods=["POST"])
@login_required
def save_onboarding():
    data = request.get_json()
    selected_movies = data.get("movies", [])
    user_id = session["user_id"]
    if selected_movies:
        seed_ids = [int(m) for m in selected_movies][:3]  # Lấy tối đa 3 phim
        conn = get_db_conn()
        conn.execute("UPDATE users SET favorite_movies = ? WHERE user_id = ?", ("|".join(map(str, seed_ids)), user_id))
        conn.commit()
        conn.close()
    session.pop("new_user", None)
    return jsonify({"ok": True})


# ── Home ──────────────────────────────────────────────────────────────────────

@app.route("/home")
@login_required
def home():
    user_id = session["user_id"]
    genres  = get_all_genres()
    
    rec_limit = request.args.get("rec_limit", 16)
    try:
        rec_limit = int(rec_limit)
        if rec_limit not in [8, 16, 24]:
            rec_limit = 16
    except ValueError:
        rec_limit = 16

    # ── Section 1: Gợi ý cá nhân hóa ──
    recs            = []
    is_personalized = False
    cluster_id      = None

    try:
        from recommender import recommend_from_seed, user_history_seed, popular_movies, predict_cluster_from_seed

        cluster_seed = user_history_seed(user_id, limit=None)
        recommend_seed = user_history_seed(user_id, limit=10)

        if recommend_seed:
            cluster_id = predict_cluster_from_seed(cluster_seed)
            recs_df    = recommend_from_seed(recommend_seed, top_k=rec_limit, use_penalty=True)
            if not recs_df.empty:
                recs            = recs_df.to_dict("records")
                is_personalized = True

        if not recs:
            recs = popular_movies(top_k=rec_limit).to_dict("records")

    except Exception as e:
        print("Recommender Error:", e)
        conn = sqlite3.connect(DB_PATH)
        recs = pd.read_sql("SELECT * FROM movies ORDER BY avg_rating DESC LIMIT ?", conn, params=(rec_limit,)).to_dict("records")
        conn.close()

    # ── Section 2: Tất cả phim — có filter + pagination ──
    selected_genre = request.args.get("genre", "")
    search_q       = request.args.get("q", "").strip().lower()
    page           = max(1, request.args.get("page", 1, type=int))
    per_page       = 36  # số phim mỗi trang

    conn = sqlite3.connect(DB_PATH)
    
    query_parts = []
    params = []
    if selected_genre:
        query_parts.append("genres LIKE ?")
        params.append(f"%{selected_genre}%")
    if search_q:
        query_parts.append("LOWER(title) LIKE ?")
        params.append(f"%{search_q}%")
        
    where_clause = ""
    if query_parts:
        where_clause = "WHERE " + " AND ".join(query_parts)
        
    total = conn.execute(f"SELECT COUNT(*) FROM movies {where_clause}", params).fetchone()[0]
    total_pages  = max(1, (total + per_page - 1) // per_page)
    page         = min(page, total_pages)
    offset       = (page - 1) * per_page
    
    sql = f"SELECT * FROM movies {where_clause} ORDER BY avg_rating DESC LIMIT ? OFFSET ?"
    page_movies = pd.read_sql(sql, conn, params=params + [per_page, offset])
    conn.close()

    return render_template(
        "home.html",
        recs            = recs,
        all_movies      = page_movies.to_dict("records"),
        genres          = genres,
        selected_genre  = selected_genre,
        search_q        = search_q,
        is_personalized = is_personalized,
        cluster_id      = cluster_id,
        account         = session.get("account", ""),
        page            = page,
        total_pages     = total_pages,
        total           = total,
        per_page        = per_page,
        rec_limit       = rec_limit,
    )


@app.route("/movie/<int:movie_id>")
@login_required
def movie_detail(movie_id):
    conn = get_db_conn()
    movie = conn.execute("SELECT * FROM movies WHERE movie_id = ?", (movie_id,)).fetchone()
    conn.close()
    if not movie:
        return "Không tìm thấy phim", 404
    return render_template("movie_detail.html", movie=dict(movie))


@app.route("/favorites")
@login_required
def favorites():
    user_id = session["user_id"]
    conn = get_db_conn()
    
    user_row = conn.execute("SELECT favorite_movies FROM users WHERE user_id = ?", (user_id,)).fetchone()
    
    initial_ids = []
    if user_row and user_row["favorite_movies"]:
        favs_str = str(user_row["favorite_movies"])
        if favs_str and favs_str != "nan":
            fav_ids = [int(x) for x in favs_str.split("|") if x.strip().isdigit()]
            initial_ids = fav_ids[:3]  # 3 phim đầu tiên là "ban đầu"
            
    later_ids = []
    user_ratings = {}
    
    ratings_rows = conn.execute("SELECT movie_id, rating FROM ratings WHERE user_id = ?", (user_id,)).fetchall()
    for r in ratings_rows:
        m_id = int(r["movie_id"])
        if m_id not in initial_ids:
            if m_id not in later_ids:
                later_ids.append(m_id)
            user_ratings[m_id] = float(r["rating"])
            
    pd_conn = sqlite3.connect(DB_PATH)
    
    if initial_ids:
        placeholders = ",".join("?" * len(initial_ids))
        initial_movies = pd.read_sql(f"SELECT * FROM movies WHERE movie_id IN ({placeholders})", pd_conn, params=initial_ids).to_dict("records")
    else:
        initial_movies = []
        
    if later_ids:
        placeholders = ",".join("?" * len(later_ids))
        later_movies = pd.read_sql(f"SELECT * FROM movies WHERE movie_id IN ({placeholders})", pd_conn, params=later_ids).to_dict("records")
    else:
        later_movies = []
        
    pd_conn.close()
    conn.close()
    
    return render_template("favorites.html", 
                           initial_favorites=initial_movies, 
                           later_favorites=later_movies,
                           user_ratings=user_ratings)


@app.route("/movie/rate", methods=["POST"])
@login_required
def rate_movie():
    user_id = session["user_id"]
    movie_id = request.form.get("movie_id")
    rating = request.form.get("rating", 5)
    if movie_id and movie_id.isdigit():
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            rating = 5.0
        rating = min(max(rating, 0.5), 5.0)
        movie_id = int(movie_id)

        conn = get_db_conn()
        conn.execute("DELETE FROM ratings WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
        conn.execute("INSERT INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)", (user_id, movie_id, rating))
        conn.commit()
        conn.close()
                
    return redirect(url_for("favorites"))


@app.route("/favorite/remove", methods=["POST"])
@login_required
def remove_favorite():
    user_id = session["user_id"]
    movie_id = request.form.get("movie_id")
    if movie_id and movie_id.isdigit():
        movie_id = int(movie_id)
        
        conn = get_db_conn()
        conn.execute("DELETE FROM ratings WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
        conn.commit()
        conn.close()
                
    return redirect(url_for("favorites"))



# ── API: live search navbar ───────────────────────────────────────────────────

@app.route("/api/search")
@login_required
def api_search():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    conn = sqlite3.connect(DB_PATH)
    results = pd.read_sql("SELECT movie_id, title, genres, poster_url, avg_rating FROM movies WHERE LOWER(title) LIKE ? LIMIT 10", conn, params=(f"%{q}%",))
    conn.close()
    return jsonify(results.to_dict("records"))


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)