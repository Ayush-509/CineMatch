import streamlit as st
import pickle
import pandas as pd
import requests
import ast
from rapidfuzz import process

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

TMDB_API_KEY = "fd8a58e039731f7020c05ce9c3e22797"
TMDB_BASE    = "https://api.themoviedb.org/3"
IMG_BASE     = "https://image.tmdb.org/t/p/w500"

# ── Load ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("movie_list.pkl", "rb") as f:
        data = pickle.load(f)
    with open("similarity.pkl", "rb") as f:
        sim = pickle.load(f)
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df, sim

movies, similarity = load_artifacts()
col_title = "title" if "title" in movies.columns else movies.columns[1]

# ── TMDB ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def tmdb_search(title):
    try:
        r = requests.get(f"{TMDB_BASE}/search/movie",
                         params={"api_key": TMDB_API_KEY, "query": title}, timeout=8)
        res = r.json().get("results", [])
        return res[0] if res else None
    except: return None

@st.cache_data(ttl=3600)
def tmdb_details(tid):
    try:
        r = requests.get(f"{TMDB_BASE}/movie/{tid}",
                         params={"api_key": TMDB_API_KEY, "append_to_response": "videos"}, timeout=8)
        return r.json()
    except: return {}

def get_poster(path):
    return IMG_BASE + path if path else None

def get_trailer(details):
    for v in details.get("videos", {}).get("results", []):
        if v.get("type") == "Trailer" and v.get("site") == "YouTube":
            return v["key"]
    return None

def fmt_money(val):
    try:
        v = int(val)
        if v == 0: return "N/A"
        if v >= 1_000_000_000: return f"${v/1e9:.1f}B"
        if v >= 1_000_000: return f"${v/1e6:.1f}M"
        return f"${v:,}"
    except: return "N/A"

def fmt_year(val):
    try: return str(pd.to_datetime(val).year)
    except: return "N/A"

# ── Recommend ──────────────────────────────────────────────────────────────────
def recommend(title, n=5):
    titles = movies[col_title].tolist()
    match  = process.extractOne(title, titles, score_cutoff=60)
    if not match: return []
    idx = movies[movies[col_title] == match[0]].index[0]
    dists = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    return [movies.iloc[i][col_title] for i, _ in dists]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, body, .stApp { font-family:'Inter',sans-serif; }
.stApp { background: linear-gradient(135deg,#0a0a0f,#0d1117,#0a0f1e); color:#e2e8f0; }

.hero { text-align:center; padding:2.5rem 1rem 1.5rem;
  background:linear-gradient(180deg,rgba(99,102,241,.12),transparent);
  border-bottom:1px solid rgba(99,102,241,.2); margin-bottom:1.5rem; }
.hero h1 { font-size:2.8rem; font-weight:800; margin:0; }
.hero h1 span { background:linear-gradient(135deg,#6366f1,#a78bfa,#ec4899);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#94a3b8; margin-top:.3rem; }

.metric-row { display:flex; gap:.8rem; justify-content:center; margin-bottom:1.5rem; flex-wrap:wrap; }
.mc { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  border-radius:12px; padding:1rem 1.8rem; text-align:center; min-width:120px; }
.mc .v { font-size:1.6rem; font-weight:700; color:#a78bfa; }
.mc .l { font-size:.7rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; }

.glass { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:1.5rem; margin-bottom:1.2rem; }

.sh { font-size:1.05rem; font-weight:600; color:#e2e8f0; margin-bottom:.8rem;
  display:flex; align-items:center; gap:.5rem; }
.sh::after { content:''; flex:1; height:1px; background:rgba(255,255,255,.08); }

.dt { font-size:1.5rem; font-weight:700; color:#f1f5f9; margin:0 0 .2rem; }
.tg { color:#94a3b8; font-style:italic; font-size:.88rem; margin-bottom:.7rem; }
.badge { display:inline-block; background:rgba(99,102,241,.2);
  border:1px solid rgba(99,102,241,.35); color:#a78bfa;
  border-radius:20px; padding:.15rem .65rem; font-size:.7rem; margin:.1rem; }
.ir { display:flex; gap:1.5rem; margin:.7rem 0; flex-wrap:wrap; }
.ii .l { font-size:.65rem; color:#64748b; text-transform:uppercase; }
.ii .v { font-size:.9rem; font-weight:600; color:#e2e8f0; }
.ov { color:#cbd5e1; line-height:1.7; font-size:.88rem; margin-top:.5rem; }

.rc { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  border-radius:12px; overflow:hidden; transition:transform .2s,border-color .2s; }
.rc:hover { transform:translateY(-3px); border-color:rgba(99,102,241,.5); }
.rc img { width:100%; height:220px; object-fit:cover; display:block; }
.rc .np { width:100%; height:220px; background:rgba(99,102,241,.08);
  display:flex; align-items:center; justify-content:center;
  color:#475569; font-size:.75rem; text-align:center; }
.rc .rt { padding:.45rem .5rem; font-size:.75rem; font-weight:600;
  color:#e2e8f0; text-align:center; line-height:1.3; }
.rc .ry { font-size:.65rem; color:#64748b; text-align:center; padding-bottom:.45rem; }

.stButton>button { background:linear-gradient(135deg,#6366f1,#a78bfa) !important;
  color:white !important; border:none !important; border-radius:10px !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 <span>CineMatch</span></h1>
  <p>AI-powered content-based movie recommendation engine</p>
</div>""", unsafe_allow_html=True)

# ── Metrics ────────────────────────────────────────────────────────────────────
total = len(movies)
gc = 20
if "genres" in movies.columns:
    try:
        ag = set()
        for g in movies["genres"].dropna():
            items = ast.literal_eval(g) if isinstance(g, str) else g
            if isinstance(items, list):
                for i in items:
                    n = i if isinstance(i, str) else i.get("name","")
                    if n: ag.add(n)
        gc = len(ag)
    except: gc = 20

st.markdown(f"""
<div class="metric-row">
  <div class="mc"><div class="v">{total:,}</div><div class="l">Movies</div></div>
  <div class="mc"><div class="v">{gc}</div><div class="l">Genres</div></div>
  <div class="mc"><div class="v">NLP</div><div class="l">Engine</div></div>
  <div class="mc"><div class="v">CosSim</div><div class="l">Algorithm</div></div>
</div>""", unsafe_allow_html=True)

# ── Controls ───────────────────────────────────────────────────────────────────
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sh">🔍 Find Your Next Movie</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    sel = st.selectbox("Movie", sorted(movies[col_title].dropna().unique()), label_visibility="collapsed")
with c2:
    num = st.slider("N", 1, 15, 5, label_visibility="collapsed")
with c3:
    go  = st.button("✨ Recommend", use_container_width=True, type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if go:
    st.session_state["movie"] = sel
    st.session_state["n"]     = num

if "movie" in st.session_state:
    mt = st.session_state["movie"]
    n  = st.session_state.get("n", 5)

    with st.spinner("Fetching details..."):
        res  = tmdb_search(mt)
        det  = tmdb_details(res["id"]) if res else {}

    poster      = get_poster(res.get("poster_path") if res else None)
    tagline     = det.get("tagline", "")
    overview    = det.get("overview", "")
    trailer_key = get_trailer(det)
    budget      = fmt_money(det.get("budget", 0))
    revenue     = fmt_money(det.get("revenue", 0))
    rt          = det.get("runtime")
    runtime     = f"{rt} min" if rt else "N/A"
    rv          = det.get("vote_average")
    rating      = f"{rv:.1f} ⭐" if isinstance(rv, (int, float)) else "N/A"

    yr = "N/A"
    if res and res.get("release_date"): yr = res["release_date"][:4]
    elif "release_date" in movies.columns:
        row = movies[movies[col_title] == mt]
        if not row.empty: yr = fmt_year(row.iloc[0].get("release_date",""))

    gl = [g["name"] for g in det.get("genres", [])]
    if not gl and "genres" in movies.columns:
        row = movies[movies[col_title] == mt]
        if not row.empty:
            raw = row.iloc[0].get("genres","[]")
            try:
                items = ast.literal_eval(raw) if isinstance(raw, str) else raw
                gl = [i if isinstance(i,str) else i.get("name","") for i in items]
            except: pass

    if not overview and "overview" in movies.columns:
        row = movies[movies[col_title] == mt]
        if not row.empty:
            overview = str(row.iloc[0].get("overview","")).strip()
    if not overview:
        overview = "Description not available."

    gh = "".join(f'<span class="badge">{g}</span>' for g in gl) or '<span class="badge">N/A</span>'

    # ── Detail Panel ───────────────────────────────────────────────────────────
    st.markdown('<div class="sh">🎥 Movie Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    cp, ci = st.columns([1, 3])

    with cp:
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown('<div style="height:320px;background:rgba(99,102,241,.1);border-radius:12px;display:flex;align-items:center;justify-content:center;color:#475569;">🎬 No Poster</div>', unsafe_allow_html=True)

    with ci:
        st.markdown(f"""
        <div class="dt">{mt} ({yr})</div>
        <div class="tg">{tagline or "No tagline available"}</div>
        <div style="margin-bottom:.5rem;">{gh}</div>
        <div class="ir">
          <div class="ii"><div class="l">Rating</div><div class="v">{rating}</div></div>
                    <div class="ii"><div class="l">Runtime</div><div class="v">{runtime}</div></div>
          <div class="ii"><div class="l">Budget</div><div class="v">{budget}</div></div>
          <div class="ii"><div class="l">Box Office</div><div class="v">{revenue}</div></div>
        </div>
        <div class="ov">{overview}</div>
        """, unsafe_allow_html=True)

        if trailer_key:
            st.markdown(
                f'<a href="https://www.youtube.com/watch?v={trailer_key}" target="_blank">'
                f'<button style="margin-top:1rem;background:linear-gradient(135deg,#ef4444,#dc2626);'
                f'color:white;border:none;border-radius:10px;padding:.5rem 1.4rem;'
                f'font-weight:600;cursor:pointer;font-size:.85rem;">▶ Watch Trailer</button></a>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<p style="color:#64748b;margin-top:.8rem;font-size:.8rem;">🎬 Trailer not available</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recommendations ────────────────────────────────────────────────────────
    st.markdown('<div class="sh">🍿 Recommended For You</div>', unsafe_allow_html=True)

    rec_titles = recommend(mt, n)

    if not rec_titles:
        st.warning("No recommendations found. Try a different movie.")
    else:
        with st.spinner("Loading recommendations..."):
            rec_data = []
            for title in rec_titles:
                r2     = tmdb_search(title)
                poster2 = get_poster(r2.get("poster_path") if r2 else None)
                year2   = r2["release_date"][:4] if r2 and r2.get("release_date") else "N/A"
                rec_data.append({"title": title, "poster": poster2, "year": year2})

        cols_per_row = min(5, len(rec_data))
        rows = [rec_data[i:i+cols_per_row] for i in range(0, len(rec_data), cols_per_row)]

        for row in rows:
            cols = st.columns(len(row))
            for col, mov in zip(cols, row):
                with col:
                    if mov["poster"]:
                        st.markdown(f"""
                        <div class="rc">
                          <img src="{mov['poster']}" alt="{mov['title']}"/>
                          <div class="rt">{mov['title']}</div>
                          <div class="ry">{mov['year']}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="rc">
                          <div class="np">🎬<br>{mov['title']}</div>
                          <div class="rt">{mov['title']}</div>
                          <div class="ry">{mov['year']}</div>
                        </div>""", unsafe_allow_html=True)

    # ── How It Works ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">⚙️ How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    steps = [
        ("📥", "Data Ingestion",  "TMDB 5000 movies & credits merged and cleaned"),
        ("🔤", "NLP Processing",  "Tags from genres, cast, crew, keywords & overview"),
        ("📊", "Vectorization",   "CountVectorizer converts tags to feature vectors"),
        ("🎯", "Similarity",      "Cosine similarity ranks the closest movies"),
    ]
    for col, (icon, title, desc) in zip([p1, p2, p3, p4], steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center;padding:.8rem;">
              <div style="font-size:1.8rem;">{icon}</div>
              <div style="font-weight:600;color:#a78bfa;margin:.3rem 0 .2rem;font-size:.88rem;">{title}</div>
              <div style="font-size:.73rem;color:#64748b;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem;color:#334155;font-size:.75rem;
border-top:1px solid rgba(255,255,255,.06);margin-top:1.5rem;">
  CineMatch &nbsp;|&nbsp; Built with Streamlit &amp; TMDB API &nbsp;|&nbsp; Content-Based Filtering
</div>""", unsafe_allow_html=True)