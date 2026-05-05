import sys
import os
from pathlib import Path
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

sys.path.insert(0, str(ROOT))

app = Flask(__name__)
app.secret_key = "cinemadb_secret_key_2024"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_users() -> pd.DataFrame:
    path = DATA_DIR / "users.csv"
    if not path.exists():
        return pd.DataFrame(columns=["user_id", "account", "password", "birth_year", "gender", "favorite_movies"])
    return pd.read_csv(path)


def save_users(df: pd.DataFrame):
    df.to_csv(DATA_DIR / "users.csv", index=False)


def load_movies_df() -> pd.DataFrame:
    path = DATA_DIR / "movies_clean.csv"
    if not path.exists():
        path = DATA_DIR / "movies.csv"
    df = pd.read_csv(path)
    df["poster_url"]  = df.get("poster_url",  pd.Series(dtype=str)).fillna("")
    df["overview"]    = df.get("overview",     pd.Series(dtype=str)).fillna("")
    df["avg_rating"]  = df.get("avg_rating",   pd.Series(dtype=float)).fillna(0.0)
    df["num_ratings"] = df.get("num_ratings",  pd.Series(dtype=int)).fillna(0)
    return df


def get_all_genres(movies: pd.DataFrame) -> list:
    genres = movies["genres"].fillna("").str.split("|").explode()
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
        users = load_users()
        user  = users[(users["account"] == account) & (users["password"] == password)]
        if not user.empty:
            row = user.iloc[0]
            session["user_id"] = int(row["user_id"])
            session["account"] = row["account"]
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
        birth_year = request.form.get("birth_year", "2000")
        if not account or not password:
            error = "Vui lòng điền đầy đủ thông tin."
        else:
            users = load_users()
            if account in users["account"].values:
                error = "Tài khoản đã tồn tại."
            else:
                new_id   = int(users["user_id"].max()) + 1 if not users.empty else 1
                new_user = pd.DataFrame([{
                    "user_id": new_id, "account": account, "password": password,
                    "birth_year": int(birth_year), "gender": gender, "favorite_movies": ""
                }])
                users = pd.concat([users, new_user], ignore_index=True)
                save_users(users)
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
    movies = load_movies_df()
    if "num_ratings" in movies.columns:
        top_movies = movies.sort_values("num_ratings", ascending=False).head(200)
    elif "liked_count" in movies.columns:
        top_movies = movies.sort_values("liked_count", ascending=False).head(200)
    else:
        top_movies = movies.head(200)
        
    return render_template("onboarding.html", 
                           genres=get_all_genres(movies), 
                           top_movies=top_movies.to_dict("records"))


@app.route("/onboarding/save", methods=["POST"])
@login_required
def save_onboarding():
    data = request.get_json()
    selected_movies = data.get("movies", [])
    user_id = session["user_id"]
    if selected_movies:
        seed_ids = [int(m) for m in selected_movies][:3]  # Lấy tối đa 3 phim
        users = load_users()
        idx = users[users["user_id"] == user_id].index
        if not idx.empty:
            users.at[idx[0], "favorite_movies"] = "|".join(map(str, seed_ids))
            save_users(users)
    session.pop("new_user", None)
    return jsonify({"ok": True})


# ── Home ──────────────────────────────────────────────────────────────────────

@app.route("/home")
@login_required
def home():
    user_id = session["user_id"]
    movies  = load_movies_df()
    genres  = get_all_genres(movies)
    
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

        seed = user_history_seed(user_id, limit=10)

        if seed:
            cluster_id = predict_cluster_from_seed(seed)
            recs_df    = recommend_from_seed(seed, top_k=rec_limit, use_penalty=True)
            if not recs_df.empty:
                recs            = recs_df.to_dict("records")
                is_personalized = True

        if not recs:
            recs = popular_movies(top_k=rec_limit).to_dict("records")

    except Exception:
        recs = movies.sort_values("avg_rating", ascending=False).head(rec_limit).to_dict("records")

    # ── Section 2: Tất cả phim — có filter + pagination ──
    selected_genre = request.args.get("genre", "")
    search_q       = request.args.get("q", "").strip().lower()
    page           = max(1, request.args.get("page", 1, type=int))
    per_page       = 36  # số phim mỗi trang

    all_movies = movies.copy()
    if selected_genre:
        all_movies = all_movies[all_movies["genres"].str.contains(selected_genre, na=False)]
    if search_q:
        all_movies = all_movies[all_movies["title"].str.lower().str.contains(search_q, na=False)]

    all_movies   = all_movies.sort_values("avg_rating", ascending=False)
    total        = len(all_movies)
    total_pages  = max(1, (total + per_page - 1) // per_page)
    page         = min(page, total_pages)
    offset       = (page - 1) * per_page
    page_movies  = all_movies.iloc[offset : offset + per_page]

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
    movies = load_movies_df()
    movie = movies[movies["movie_id"] == movie_id]
    if movie.empty:
        return "Không tìm thấy phim", 404
    return render_template("movie_detail.html", movie=movie.iloc[0].to_dict())


@app.route("/favorites")
@login_required
def favorites():
    user_id = session["user_id"]
    users = load_users()
    row = users[users["user_id"] == user_id]
    
    initial_ids = []
    if not row.empty:
        favs_str = str(row.iloc[0].get("favorite_movies", ""))
        if favs_str and favs_str != "nan":
            fav_ids = [int(x) for x in favs_str.split("|") if x.strip().isdigit()]
            initial_ids = fav_ids[:3]  # 3 phim đầu tiên là "ban đầu"
            
    later_ids = []
    user_ratings = {}
    ratings_path = DATA_DIR / "ratings.csv"
    if ratings_path.exists():
        ratings_df = pd.read_csv(ratings_path)
        user_liked = ratings_df[ratings_df["user_id"] == user_id]
        if not user_liked.empty:
            for _, r in user_liked.iterrows():
                m_id = int(r["movie_id"])
                if m_id not in initial_ids:
                    if m_id not in later_ids:
                        later_ids.append(m_id)
                    user_ratings[m_id] = float(r["rating"])
    
    movies = load_movies_df()
    initial_movies = movies[movies["movie_id"].isin(initial_ids)].to_dict("records")
    later_movies = movies[movies["movie_id"].isin(later_ids)].to_dict("records")
    
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
        movie_id = int(movie_id)
        rating = float(rating)
        
        ratings_path = DATA_DIR / "ratings.csv"
        if ratings_path.exists():
            ratings_df = pd.read_csv(ratings_path)
            already_liked_idx = ratings_df[(ratings_df["user_id"] == user_id) & (ratings_df["movie_id"] == movie_id)].index
            if not already_liked_idx.empty:
                ratings_df.loc[already_liked_idx, "rating"] = rating
            else:
                new_row = pd.DataFrame([{"user_id": user_id, "movie_id": movie_id, "rating": rating}])
                ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
            ratings_df.to_csv(ratings_path, index=False)
        else:
            new_df = pd.DataFrame([{"user_id": user_id, "movie_id": movie_id, "rating": rating}])
            new_df.to_csv(ratings_path, index=False)
                
    return redirect(url_for("favorites"))


@app.route("/favorite/remove", methods=["POST"])
@login_required
def remove_favorite():
    user_id = session["user_id"]
    movie_id = request.form.get("movie_id")
    if movie_id and movie_id.isdigit():
        movie_id = int(movie_id)
        
        ratings_path = DATA_DIR / "ratings.csv"
        if ratings_path.exists():
            ratings_df = pd.read_csv(ratings_path)
            # Giữ lại các phim không khớp với (user_id và movie_id)
            mask = ~((ratings_df["user_id"] == user_id) & (ratings_df["movie_id"] == movie_id))
            if not mask.all():
                ratings_df = ratings_df[mask]
                ratings_df.to_csv(ratings_path, index=False)
                
    return redirect(url_for("favorites"))



# ── API: live search navbar ───────────────────────────────────────────────────

@app.route("/api/search")
@login_required
def api_search():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    movies  = load_movies_df()
    results = movies[movies["title"].str.lower().str.contains(q, na=False)].head(10)
    return jsonify(results[["movie_id", "title", "genres", "poster_url", "avg_rating"]].to_dict("records"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)