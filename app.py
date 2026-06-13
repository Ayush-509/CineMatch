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
IMG_SMALL    = "https://image.tmdb.org/t/p/w185"

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
all_titles = sorted(movies[col_title].dropna().unique().tolist())

# ── TMDB helpers ──────────────────────────────────────────────────────────────
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
                         params={"api_key": TMDB_API_KEY,
                                 "append_to_response": "videos,credits"}, timeout=8)
        return r.json()
    except: return {}

def get_poster(path, base=IMG_BASE): return base + path if path else None

def get_trailer(details):
    for v in details.get("videos", {}).get("results", []):
        if v.get("type") == "Trailer" and v.get("site") == "YouTube":
            return v["key"]
    return None

def fmt_money(val):
    try:
        if pd.isna(val) or val == "" or str(val).lower() == "nan": return "N/A"
        v = int(float(val))
        if v == 0: return "N/A"
        if v >= 1_000_000_000: return f"${v/1e9:.1f}B"
        if v >= 1_000_000: return f"${v/1e6:.1f}M"
        return f"${v:,}"
    except: return "N/A"

def recommend(title, n=5):
    match = process.extractOne(title, all_titles, score_cutoff=60)
    if not match: return []
    idx = movies[movies[col_title] == match[0]].index[0]
    dists = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    return [movies.iloc[i][col_title] for i, _ in dists]

def build_movie_data(title):
    # 1. Gather API Search and Detail Info (Primarily for dynamic asset/credits media)
    res = tmdb_search(title)
    det = tmdb_details(res["id"]) if res else {}
    
    poster      = get_poster(res.get("poster_path") if res else None)
    backdrop    = get_poster(res.get("backdrop_path") if res else None, "https://image.tmdb.org/t/p/w1280")
    trailer_key = get_trailer(det)

    # 2. Extract matching movie row from local CSV
    row = movies[movies[col_title] == title]
    has_local = not row.empty
    
    # 3. Pull Data directly from your dataset row layout
    gl = []
    if has_local and "genres" in movies.columns:
        local_genres = row.iloc[0].get("genres", "")
        if pd.notna(local_genres) and local_genres:
            try:
                parsed_genres = ast.literal_eval(str(local_genres))
                if isinstance(parsed_genres, list):
                    gl = [str(g) for g in parsed_genres]
            except:
                gl = [g.strip() for g in str(local_genres).replace("[","").replace("]","").replace("'","").replace('"',"").split(",") if g.strip()]
    
    if not gl:
        gl = [g["name"] for g in det.get("genres", [])]

    # Map directly to local dataset columns
    budget_raw  = row.iloc[0].get("budget", 0) if has_local else det.get("budget", 0)
    revenue_raw = row.iloc[0].get("revenue", 0) if has_local else det.get("revenue", 0)
    tagline     = row.iloc[0].get("tagline", "") if has_local else det.get("tagline", "")
    overview    = row.iloc[0].get("overview", "") if has_local else det.get("overview", "")
    runtime_raw = row.iloc[0].get("runtime", None) if has_local else det.get("runtime")
    
    # Local assignments for rating metrics
    rv = row.iloc[0].get("vote_average", "N/A") if has_local else det.get("vote_average", "N/A")
    vc = row.iloc[0].get("vote_count", 0) if has_local else det.get("vote_count", 0)
    
    if has_local and "release_date" in movies.columns:
        local_date = str(row.iloc[0].get("release_date", ""))
        yr = local_date[:4] if len(local_date) >= 4 and local_date.lower() != "nan" else "N/A"
    else:
        yr = res["release_date"][:4] if res and res.get("release_date") else "N/A"

    if pd.isna(tagline) or str(tagline).lower() == "nan": tagline = ""
    if pd.isna(overview) or str(overview).lower() == "nan": overview = "No description available."

    # Format numeric entries for rating UI layouts safely
    try:
        rating_fmt = f"{float(rv):.1f}" if pd.notna(rv) and str(rv).lower() != "nan" else "N/A"
    except:
        rating_fmt = "N/A"

    try:
        votes_fmt = f"{int(float(vc)):,}" if pd.notna(vc) and vc != "" else "N/A"
    except:
        votes_fmt = "N/A"

    # 4. Process live Cast and Crew credits through API
    cast = []
    for p in det.get("credits", {}).get("cast", [])[:10]:
        cast.append({
            "name":      p.get("name", ""),
            "character": p.get("character", ""),
            "photo":     get_poster(p.get("profile_path"), IMG_SMALL),
        })

    crew_roles = {"Director", "Producer", "Screenplay", "Story", "Original Music Composer"}
    crew = []
    seen = set()
    for p in det.get("credits", {}).get("crew", []):
        job = p.get("job", "")
        name = p.get("name", "")
        if job in crew_roles and name not in seen:
            seen.add(name)
            crew.append({"name": name, "job": job})

    return dict(title=title, poster=poster, backdrop=backdrop, tagline=tagline,
                overview=overview, trailer_key=trailer_key,
                budget=fmt_money(budget_raw),
                revenue=fmt_money(revenue_raw),
                runtime=f"{int(float(runtime_raw))} min" if pd.notna(runtime_raw) and runtime_raw != "" else "N/A",
                rating=rating_fmt, votes=votes_fmt,
                year=yr, genres=gl, cast=cast, crew=crew)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("selected", None), ("n_recs", 5), ("saved_movies", [])]:
    if k not in st.session_state: st.session_state[k] = v

def pick(title):
    st.session_state.selected = title

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, body, .stApp { font-family:'Inter',sans-serif; }
.stApp { background:#07090f; color:#e2e8f0; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding-top:0 !important; max-width:1320px !important; }

.hero { text-align:center; padding:2.2rem 1rem 1.4rem;
  background:linear-gradient(180deg,rgba(99,102,241,.1),transparent);
  border-bottom:1px solid rgba(99,102,241,.12); margin-bottom:1.5rem; }
.hero h1 { font-size:2.6rem; font-weight:800; margin:0; }
.hero h1 span { background:linear-gradient(135deg,#6366f1,#a78bfa,#ec4899);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#475569; margin:.3rem 0 0; font-size:.88rem; }

.metric-row { display:flex; gap:.8rem; justify-content:center; margin-bottom:1.5rem; flex-wrap:wrap; }
.mc { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  border-radius:12px; padding:1rem 1.8rem; text-align:center; min-width:120px; }
.mc .v { font-size:1.6rem; font-weight:700; color:#a78bfa; }
.mc .l { font-size:.7rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; }

div[data-testid="stTextInput"] input {
  background: #111827 !important;   
  border: 1px solid #2d3748 !important;
  border-radius: 10px !important;
  color: #f1f5f9 !important;        
  font-size: 0.95rem !important;
  padding: 0.65rem 1rem !important;
  transition: none !important;
  box-shadow: none !important;
}

div[data-testid="stTextInput"] input::placeholder {
  color: #94a3b8 !important;
}

div[data-testid="stTextInput"] input:hover {
  border-color: #3b82f6 !important;
}

div[data-testid="stTextInput"] input:focus {
  outline: none !important;
  border-color: #60a5fa !important;
  box-shadow: none !important;
  background: #0f172a !important;
}

.stButton>button { background:linear-gradient(135deg,#6366f1,#a78bfa)!important;
  color:white!important; border:none!important; border-radius:10px!important;
  font-weight:600!important; transition:opacity .2s!important; }
.stButton>button:hover { opacity:.82!important; }

[data-testid="stSlider"]>div>div>div { background:rgba(99,102,241,.25)!important; }
[data-testid="stSlider"]>div>div>div>div { background:linear-gradient(135deg,#6366f1,#a78bfa)!important; }

.glass { background:rgba(15, 23, 42, 0.85); border:1px solid rgba(255,255,255,.07);
  border-radius:16px; padding:1.4rem; margin-bottom:1.2rem; backdrop-filter: blur(12px); }

.sh { font-size:.95rem; font-weight:700; color:#e2e8f0; margin:1.2rem 0 .8rem;
  display:flex; align-items:center; gap:.5rem; }
.sh::after { content:''; flex:1; height:1px; background:rgba(255,255,255,.07); }

.badge { display:inline-block; background:rgba(99,102,241,.18);
  border:1px solid rgba(99,102,241,.3); color:#a78bfa;
  border-radius:20px; padding:.15rem .65rem; font-size:.7rem; margin:.1rem; }

.dt  { font-size:1.7rem; font-weight:800; color:#f1f5f9; margin-bottom:.15rem; }
.tg  { color:#64748b; font-style:italic; font-size:.85rem; margin-bottom:.6rem; }
.ir  { display:flex; gap:1.8rem; margin:.6rem 0; flex-wrap:wrap; }
.ii .l { font-size:.62rem; color:#64748b; text-transform:uppercase; letter-spacing:.06em; }
.ii .v { font-size:.9rem; font-weight:600; color:#e2e8f0; }
.ov  { color:#94a3b8; line-height:1.75; font-size:.86rem; margin-top:.6rem; }

.stat-row { display:flex; gap:.7rem; flex-wrap:wrap; margin:.8rem 0; }
.stat { background:rgba(99,102,241,.1); border:1px solid rgba(99,102,241,.2);
  border-radius:10px; padding:.55rem 1rem; text-align:center; flex:1; min-width:80px; }
.stat .sv { font-size:1.1rem; font-weight:700; color:#a78bfa; }
.stat .sl { font-size:.62rem; color:#475569; text-transform:uppercase; letter-spacing:.06em; margin-top:.1rem; }

.cast-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07);
  border-radius:12px; overflow:hidden; text-align:center; transition:transform .2s, border-color .2s; }
.cast-card:hover { transform:translateY(-3px); border-color:rgba(167,139,250,.4); }
.cast-card img { width:100%; height:150px; object-fit:cover; object-position:top; display:block; }
.cast-card .noph { width:100%; height:150px; background:rgba(99,102,241,.08);
  display:flex; align-items:center; justify-content:center; font-size:2rem; }
.cast-card .cn { font-size:.75rem; font-weight:700; color:#e2e8f0; padding:.4rem .4rem .1rem; line-height:1.2; }
.cast-card .cc { font-size:.65rem; color:#64748b; padding-bottom:.4rem; }

.crew-pill { display:inline-flex; align-items:center; gap:.4rem;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  border-radius:30px; padding:.3rem .85rem; margin:.25rem; }
.crew-pill .cj { font-size:.62rem; color:#6366f1; text-transform:uppercase;
  font-weight:700; letter-spacing:.06em; }
.crew-pill .cm { font-size:.78rem; color:#e2e8f0; font-weight:600; }

.rc { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07);
  border-radius:12px 12px 0px 0px; overflow:hidden; margin-bottom: 0px;}
.rc img { width:100%; height:200px; object-fit:cover; display:block; }
.rc .np { width:100%; height:200px; background:rgba(99,102,241,.07);
  display:flex; align-items:center; justify-content:center; color:#475569; font-size:1.5rem; }
.rc .rt { padding:.6rem .5rem .1rem; font-size:.8rem; font-weight:700;
  color:#e2e8f0; text-align:center; line-height:1.3; }
.rc .ry { font-size:.68rem; color:#64748b; text-align:center; padding-bottom:.6rem; }
.rc:hover { transform:translateY(-3px); border-color:rgba(167,139,250,.4); }
.rec-wrapper { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255,255,255,.07); border-radius: 12px; margin-bottom: 1rem; overflow: hidden;}

label[data-testid="stWidgetLabel"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 <span>CineMatch</span></h1>
  <p>Discover Similar Films · Explore Cast & Crew · Watch Trailers</p>
</div>""", unsafe_allow_html=True)

# ── Metrics ────────────────────────────────────────────────────────────────────
total = len(movies)
gc = 20
if "genres" in movies.columns:
    try:
        ag = set()
        for g in movies["genres"].dropna():
            items = ast.literal_eval(str(g)) if isinstance(g, str) else g
            if isinstance(items, list):
                for n in items:
                    if n: ag.add(str(n))
        gc = len(ag)
    except: gc = 20

st.markdown(f"""
<div class="metric-row">
  <div class="mc"><div class="v">{total:,}</div><div class="l">Movies</div></div>
  <div class="mc"><div class="v">{gc}</div><div class="l">Genres</div></div>
  <div class="mc"><div class="v">NLP</div><div class="l">Engine</div></div>
  <div class="mc"><div class="v">CosSim</div><div class="l">Algorithm</div></div>
</div>""", unsafe_allow_html=True)

# ── Search ────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([4, 1, 1])
with c1:
    chosen = st.text_input("Search", value=st.session_state.selected if st.session_state.selected else "", 
                            placeholder="🔍 Type a movie title and hit enter...")
with c2:
    st.session_state.n_recs = st.slider("Results", 1, 15,
        st.session_state.n_recs, label_visibility="collapsed")
with c3:
    go = st.button("✨ Recommend", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if (go or chosen) and chosen != st.session_state.selected:
    match = process.extractOne(chosen, all_titles, score_cutoff=50)
    if match:
        st.session_state.selected = match[0]
    else:
        st.session_state.selected = chosen

# ── Detail page ───────────────────────────────────────────────────────────────
if st.session_state.selected:
    mt = st.session_state.selected

    bc, _ = st.columns([1, 6])
    with bc:
        if st.button("← Back"):
            st.session_state.selected = None
            st.rerun()

    with st.spinner("Loading..."):
        md = build_movie_data(mt)

    if md["poster"]:
        glass_style = f"""
        <style>
        .movie-poster-backdrop-glass {{
            background: linear-gradient(rgba(15, 23, 42, 0.94), rgba(15, 23, 42, 0.97)), url('{md["poster"]}');
            background-size: cover;
            background-position: center;
        }}
        </style>
        """
        st.markdown(glass_style, unsafe_allow_html=True)
        backdrop_class = "glass movie-poster-backdrop-glass"
    else:
        backdrop_class = "glass"

    st.markdown(f'<div class="{backdrop_class}">', unsafe_allow_html=True)
    cp, ci = st.columns([1, 3])
    with cp:
        if md["poster"]:
            st.image(md["poster"], use_container_width=True)
        else:
            st.markdown('<div style="height:300px;background:rgba(99,102,241,.08);border-radius:10px;display:flex;align-items:center;justify-content:center;color:#475569;font-size:2rem;">🎬</div>', unsafe_allow_html=True)

    with ci:
        gh = "".join(f'<span class="badge">{g}</span>' for g in md["genres"]) or '<span class="badge">N/A</span>'
        st.markdown(f"""
        <div class="dt">{mt} <span style="font-size:1rem;color:#64748b;font-weight:400;">({md['year']})</span></div>
        <div class="tg">{md['tagline'] or 'No tagline available'}</div>
        <div style="margin-bottom:.7rem;">{gh}</div>
        <div class="stat-row">
          <div class="stat"><div class="sv">⭐ {md['rating']}</div><div class="sl">Rating</div></div>
          <div class="stat"><div class="sv">{md['votes']}</div><div class="sl">Votes</div></div>
          <div class="stat"><div class="sv">{md['runtime']}</div><div class="sl">Runtime</div></div>
          <div class="stat"><div class="sv">{md['budget']}</div><div class="sl">Budget</div></div>
          <div class="stat"><div class="sv">{md['revenue']}</div><div class="sl">Box Office</div></div>
        </div>
        <div class="ov" style="margin-bottom: 1.2rem;">{md['overview']}</div>""", unsafe_allow_html=True)

        if md["trailer_key"]:
            st.markdown('<div style="font-size:0.78rem; font-weight:700; color:#a78bfa; margin-bottom:0.4rem; text-transform:uppercase; letter-spacing:0.05em;">▶ Official Trailer Preview</div>', unsafe_allow_html=True)
            video_col, _ = st.columns([1.0, 2.0])
            with video_col:
                st.video(f"https://www.youtube.com/watch?v={md['trailer_key']}")
                
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Cast ──────────────────────────────────────────────────────────────────
    if md["cast"]:
        st.markdown('<div class="sh">🎭 Top Cast</div>', unsafe_allow_html=True)
        ccols = st.columns(min(10, len(md["cast"])))
        for col, p in zip(ccols, md["cast"]):
            with col:
                if p["photo"]:
                    st.markdown(f'<div class="cast-card"><img src="{p["photo"]}"/>'
                                f'<div class="cn">{p["name"]}</div>'
                                f'<div class="cc">{p["character"]}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="cast-card"><div class="noph">👤</div>'
                                f'<div class="cn">{p["name"]}</div>'
                                f'<div class="cc">{p["character"]}</div></div>', unsafe_allow_html=True)

    # ── Crew ──────────────────────────────────────────────────────────────────
    if md["crew"]:
        st.markdown('<div class="sh">🎬 Crew</div>', unsafe_allow_html=True)
        pills = "".join(
            f'<span class="crew-pill"><span class="cj">{p["job"]}</span>'
            f'<span class="cm">{p["name"]}</span></span>'
            for p in md["crew"]
        )
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;">{pills}</div>', unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    rec_titles = recommend(mt, st.session_state.n_recs)
    if rec_titles:
        st.markdown('<div class="sh">🍿 You Might Also Like</div>', unsafe_allow_html=True)
        cols_per_row = min(5, len(rec_titles))
        for row_start in range(0, len(rec_titles), cols_per_row):
            batch = rec_titles[row_start:row_start + cols_per_row]
            rcols = st.columns(len(batch))
            for col, rtitle in zip(rcols, batch):
                with col:
                    r2      = tmdb_search(rtitle)
                    poster2 = get_poster(r2.get("poster_path") if r2 else None)
                    year2   = r2["release_date"][:4] if r2 and r2.get("release_date") else "N/A"
                    
                    st.markdown('<div class="rec-wrapper">', unsafe_allow_html=True)
                    if poster2:
                        st.markdown(f'<div class="rc"><img src="{poster2}"/>'
                                    f'<div class="rt">{rtitle}</div>'
                                    f'<div class="ry">{year2}</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="rc"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoWcWg0E8pSjBNi0TtiZsqu8uD2PAr_K11DA&s"/>'
                                    f'<div class="rt">{rtitle}</div>'
                                    f'<div class="ry">{year2}</div></div>', unsafe_allow_html=True)
                    
                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        if st.button("👁️ View", key=f"view_{rtitle}", use_container_width=True):
                            pick(rtitle)
                            st.rerun()
                    with btn_c2:
                        if st.button("🔖 Save", key=f"save_{rtitle}", use_container_width=True):
                            if rtitle not in st.session_state.saved_movies:
                                st.session_state.saved_movies.append(rtitle)
                                st.toast(f"Saved {rtitle} to bookmarks!", icon="✨")
                            else:
                                st.toast(f"{rtitle} is already saved!", icon="ℹ️")
                    st.markdown('</div>', unsafe_allow_html=True)

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

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 1rem;color:#64748b;">
      <div style="font-size:4rem;margin-bottom:1rem;">🎬</div>
      <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;">Search for a movie above to get started</div>
    </div>""", unsafe_allow_html=True)
