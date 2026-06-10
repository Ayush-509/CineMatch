import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(244, 114, 182, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(59, 130, 246, 0.18), transparent 25%),
        radial-gradient(circle at bottom left, rgba(16, 185, 129, 0.15), transparent 24%),
        linear-gradient(135deg, #070b16 0%, #0f172a 45%, #111827 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.hero-wrap {
    padding: 1.5rem 0 1rem 0;
}

.badge {
    display: inline-block;
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    color: #f9a8d4;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 3.4rem;
    line-height: 1.05;
    font-weight: 850;
    color: #ffffff;
    margin-bottom: 0.6rem;
    letter-spacing: -0.03em;
}

.hero-title span {
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.08rem;
    color: #cbd5e1;
    max-width: 760px;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}

.glass-card {
    background: rgba(15, 23, 42, 0.62);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 1.25rem;
    box-shadow: 0 12px 40px rgba(0,0,0,0.22);
    backdrop-filter: blur(16px);
}

.metric-card {
    background: linear-gradient(180deg, rgba(30,41,59,0.9), rgba(15,23,42,0.75));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 22px;
    padding: 1.3rem 1rem;
    text-align: center;
    box-shadow: 0 12px 28px rgba(0,0,0,0.18);
}

.metric-number {
    font-size: 2rem;
    font-weight: 850;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.metric-label {
    font-size: 0.98rem;
    color: #94a3b8;
    font-weight: 600;
}

.section-heading {
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    margin: 1.6rem 0 0.9rem 0;
    letter-spacing: -0.02em;
}

.muted-text {
    color: #cbd5e1;
    line-height: 1.8;
    font-size: 1rem;
}

.result-card {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(180deg, rgba(15,23,42,0.95), rgba(17,24,39,0.88));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 1.15rem;
    min-height: 250px;
    box-shadow: 0 16px 34px rgba(0,0,0,0.22);
    margin-bottom: 1rem;
}

.result-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(96,165,250,0.08), rgba(244,114,182,0.06), rgba(16,185,129,0.06));
    pointer-events: none;
}

.rank-chip {
    position: relative;
    z-index: 1;
    display: inline-block;
    padding: 0.38rem 0.78rem;
    border-radius: 999px;
    background: rgba(59,130,246,0.18);
    color: #bfdbfe;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 0.9rem;
}

.movie-title {
    position: relative;
    z-index: 1;
    color: #ffffff;
    font-size: 1.18rem;
    font-weight: 800;
    line-height: 1.4;
    margin-bottom: 0.75rem;
}

.movie-desc {
    position: relative;
    z-index: 1;
    color: #dbe4f0;
    font-size: 0.96rem;
    line-height: 1.7;
}

.stButton > button {
    background: linear-gradient(90deg, #ec4899, #8b5cf6, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.8rem 1.4rem !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 24px rgba(139,92,246,0.28) !important;
}

.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {
    color: white !important;
    border: none !important;
    box-shadow: 0 0 0 0.2rem rgba(168,85,247,0.25) !important;
}

/* Keep Streamlit widgets readable instead of forcing custom popup colors */
label {
    color: #e2e8f0 !important;
    font-weight: 650 !important;
}

.stSlider label,
.stSelectbox label {
    color: #e2e8f0 !important;
}

/* Sidebar hidden look on page */
hr {
    border-color: rgba(255,255,255,0.08);
}

@media (max-width: 768px) {
    .hero-title {
        font-size: 2.4rem;
    }
    .section-heading {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    movie_dict = pickle.load(open("movie_list.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    movies = pd.DataFrame(movie_dict)
    return movies, similarity

movies, similarity = load_data()
def make_snippet(row):
    overview = str(row.get("overview", "")).strip()

    if overview and overview.lower() != "nan":
        words = overview.split()
        short_text = " ".join(words[:25])
        if len(words) > 25:
            short_text += "..."
        return short_text

    return "Description not available."
def recommend(movie_title, top_k=5):
    movie_index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[movie_index]

    movie_list_sorted = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1: top_k + 1]

    results = []
    for rank, (idx, score) in enumerate(movie_list_sorted, start=1):
        row = movies.iloc[idx]

        results.append({
            "rank": rank,
            "movie_id": row["movie_id"],
            "title": row["title"],
            "overview": row.get("overview", ""),
            "similarity": float(score)
        })

    return results
st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)

st.markdown('<div class="hero-title">Discover your next <span>favorite movie</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">CineMatch is a content-based recommendation system that uses NLP, feature engineering, and cosine similarity to suggest movies based on storyline, genre, cast, and crew similarities.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-number">{movies.shape[0]}</div><div class="metric-label">Movies in dataset</div></div>',
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        '<div class="metric-card"><div class="metric-number">NLP</div><div class="metric-label">Feature engineering pipeline</div></div>',
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        '<div class="metric-card"><div class="metric-number">ML</div><div class="metric-label">Cosine similarity engine</div></div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="section-heading">Find similar movies</div>', unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    selected_movie_name = st.selectbox(
        "Select a movie",
        sorted(movies["title"].values)
    )
    top_k = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )
    go = st.button("Recommend Movies")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.4rem; margin-top:0;">Model pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="muted-text">
        • Combined overview, genres, keywords, cast, and crew into a unified tag field.<br>
        • Applied stemming and text preprocessing using NLTK.<br>
        • Converted textual content into vectors using CountVectorizer.<br>
        • Ranked nearest movies using cosine similarity.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
if go:
    results = recommend(selected_movie_name, top_k=top_k)

    st.markdown('<div class="section-heading">Recommended for you</div>', unsafe_allow_html=True)

    rows = [results[i:i+5] for i in range(0, len(results), 5)]

    for row_movies in rows:
        cols = st.columns(len(row_movies))
        for col, movie in zip(cols, row_movies):
            with col:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="rank-chip">Rank #{movie['rank']}</div>
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-desc">{make_snippet(movie)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )