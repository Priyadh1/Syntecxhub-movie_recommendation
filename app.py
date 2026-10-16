import ast
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineVibe", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.stApp {background:#0b1020;color:#f7f7fb}
.block-container {max-width:1200px;padding-top:2rem}
.hero {padding:1.5rem 1.8rem;border-radius:20px;background:linear-gradient(120deg,#22254a,#512b5e,#123e53);margin-bottom:1rem}
.hero h1 {color:#ffffff !important;margin:0}
.hero p {color:#e2e5f4 !important;margin:0}
div.stButton>button {border-radius:12px;font-weight:650;min-height:2.5rem}
.stTextArea textarea, .stTextInput input {
  color:#ffffff !important; background:#141a2e !important;
  caret-color:#ffffff !important; border:1px solid #65708f !important;
  -webkit-text-fill-color:#ffffff !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
  color:#b8c2dc !important; -webkit-text-fill-color:#b8c2dc !important; opacity:1;
}
label, .stMarkdown, p, span {color:inherit}
</style>
""", unsafe_allow_html=True)

def parse_names(value, limit=None):
    try:
        data = value if isinstance(value, list) else ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    names = [str(x.get("name")) for x in data if isinstance(x, dict) and x.get("name")]
    return names[:limit] if limit else names

def directors(value):
    try:
        data = ast.literal_eval(value) if isinstance(value, str) else []
        return [x["name"] for x in data if isinstance(x, dict) and x.get("job") == "Director"]
    except (ValueError, SyntaxError, TypeError):
        return []

def clean_text(value):
    if isinstance(value, list):
        value = " ".join(map(str, value))
    return re.sub(r"[^a-zA-Z0-9 ]", " ", value.lower()) if isinstance(value, str) else ""

@st.cache_data(show_spinner=False)
def load_data(movies_path, credits_path, movie_stamp, credit_stamp):
    m = pd.read_csv(movies_path)
    c = pd.read_csv(credits_path)
    df = m.merge(c, left_on="id", right_on="movie_id", how="inner", suffixes=("", "_credits"))
    df["genres"] = df["genres"].apply(parse_names)
    df["keywords"] = df["keywords"].apply(parse_names)
    df["cast"] = df["cast"].apply(lambda x: parse_names(x, 3))
    df["crew"] = df["crew"].apply(directors)
    df["overview"] = df["overview"].fillna("")
    df["tags"] = (
        df["genres"].apply(clean_text) + " " + df["keywords"].apply(clean_text) + " " +
        df["cast"].apply(clean_text) + " " + df["crew"].apply(clean_text) + " " +
        df["overview"].apply(clean_text)
    )
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    matrix = vectorizer.fit_transform(df["tags"])
    return df, vectorizer, matrix

def recommend(df, matrix, seeds, count=5, excluded=None):
    if not seeds:
        return []
    scores = cosine_similarity(matrix[seeds], matrix).mean(axis=0)
    blocked = set(seeds) | set(excluded or [])
    for idx in blocked:
        scores[idx] = -1
    return [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1] if scores[i] >= 0][:count]

def show_movie(df, idx, score=None):
    row = df.iloc[idx]
    with st.container(border=True):
        st.subheader(str(row["title"]))
        year = str(row.get("release_date", ""))[:4]
        rating = pd.to_numeric(row.get("vote_average"), errors="coerce")
        meta = [x for x in [year if year != "nan" else "", f"⭐ {rating:.1f}/10" if pd.notna(rating) else "", f"Match {score:.0%}" if score is not None else ""] if x]
        if meta:
            st.caption(" · ".join(meta))
        if row["genres"]:
            st.write("  ".join(f"`{g}`" for g in row["genres"][:4]))
        st.write(str(row["overview"]).strip() or "No overview available.")

root = Path(__file__).resolve().parent
movie_file = next((p for p in [root/"tmdb_5000_movies.csv", root/"data"/"tmdb_5000_movies.csv"] if p.exists()), None)
credit_file = next((p for p in [root/"tmdb_5000_credits.csv", root/"data"/"tmdb_5000_credits.csv"] if p.exists()), None)
if credit_file is None:
    zip_file = next((p for p in [root/"tmdb_5000_credits.zip", root/"data"/"tmdb_5000_credits.zip"] if p.exists()), None)
    if zip_file:
        try:
            with zipfile.ZipFile(zip_file) as z:
                member = next((n for n in z.namelist() if Path(n).name == "tmdb_5000_credits.csv"), None)
                if member:
                    credit_file = root/"tmdb_5000_credits.csv"
                    if not credit_file.exists():
                        credit_file.write_bytes(z.read(member))
        except (OSError, zipfile.BadZipFile):
            credit_file = None

st.markdown('<div class="hero"><h1>🎬 FlickFusion</h1><p>One place for your movie mood, your crew, and your next great watch.</p></div>', unsafe_allow_html=True)
if movie_file is None or credit_file is None:
    st.error("Dataset files not found. Put tmdb_5000_movies.csv and tmdb_5000_credits.csv (or tmdb_5000_credits.zip) beside app.py.")
    st.stop()

try:
    with st.spinner("Preparing the movie collection..."):
        df, vectorizer, matrix = load_data(str(movie_file), str(credit_file), movie_file.stat().st_mtime, credit_file.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not prepare datasets: {exc}")
    st.stop()

st.caption(f"{len(df):,} movies ready to explore.")
similar_tab, vibe_tab, mood_tab, swipe_tab, group_tab, stream_tab = st.tabs([
    "🎞️ Similar Movies", "✨ Vibe Check", "🎭 Emoji Mood",
    "💚 Swipe to Discover", "👥 Watch Together", "📺 Where to Watch"
])

with similar_tab:
    titles = df["title"].fillna("Untitled").astype(str).tolist()
    default = titles.index("Avatar") if "Avatar" in titles else 0
    selected = st.selectbox("Choose a movie you like", titles, index=default)
    count = st.slider("Recommendations", 3, 10, 5, key="similar_count")
    if st.button("Find similar movies", type="primary"):
        seed = int(df.index[df["title"].fillna("").astype(str) == selected][0])
        st.session_state["similar_results"] = recommend(df, matrix, [seed], count)
    for i, score in st.session_state.get("similar_results", []):
        show_movie(df, i, score)

with vibe_tab:
    query = st.text_area(
        "Describe the feeling, setting, or story you want",
        placeholder="A rainy Sunday in Paris with a dark psychological twist...",
        height=120,
        key="vibe_query",
        help="Click inside the box and type your description. Your text and cursor should be visible."
    )
    count = st.slider("Results", 3, 10, 5, key="vibe_count")
    if st.button("✨ Match my vibe", type="primary"):
        if not query.strip():
            st.warning("Write a vibe first.")
        else:
            q = vectorizer.transform([clean_text(query)])
            scores = cosine_similarity(q, matrix).ravel()
            st.session_state["vibe_results"] = [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:count]]
    for i, score in st.session_state.get("vibe_results", []):
        show_movie(df, i, score)

with mood_tab:
    mood_map = {
        "🧠 Mind-bending": "psychological mystery mind bending twist",
        "🩸 Dark & intense": "dark thriller crime suspense intense",
        "🤠 Adventure": "adventure action journey western",
        "🚀 Sci-fi": "science fiction space futuristic alien technology",
        "😭 Emotional": "emotional family love friendship drama",
        "😂 Feel-good": "comedy funny joyful feel good",
        "👻 Spooky": "horror supernatural ghosts frightening",
        "❤️ Romance": "romance love relationship romantic",
        "🕵️ Mystery": "mystery detective investigation secrets",
        "🌿 Calm & cosy": "quiet gentle friendship nature heartwarming",
    }
    chosen = st.multiselect("Choose one or more moods", list(mood_map))
    count = st.slider("Results", 3, 10, 5, key="mood_count")
    if st.button("🎭 Create my mood mix", type="primary"):
        if not chosen:
            st.warning("Choose at least one mood.")
        else:
            q = vectorizer.transform([" ".join(mood_map[x] for x in chosen)])
            scores = cosine_similarity(q, matrix).ravel()
            st.session_state["mood_results"] = [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:count]]
    for i, score in st.session_state.get("mood_results", []):
        show_movie(df, i, score)

with swipe_tab:
    st.write("Like or pass on eight films, then get recommendations based on your likes.")
    if "deck" not in st.session_state:
        popularity = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
        pool = popularity.nlargest(min(60, len(df))).index.to_numpy()
        rng = np.random.default_rng()
        st.session_state["deck"] = rng.choice(pool, size=min(8, len(pool)), replace=False).tolist()
        st.session_state["deck_pos"] = 0
        st.session_state["likes"] = []
        st.session_state["passes"] = []
    pos = st.session_state["deck_pos"]
    deck = st.session_state["deck"]
    if pos < len(deck):
        idx = deck[pos]
        row = df.iloc[idx]
        st.progress(pos / len(deck), text=f"Movie {pos+1} of {len(deck)}")
        st.markdown(f"### {row['title']}")
        rating = pd.to_numeric(row.get("vote_average"), errors="coerce")
        if pd.notna(rating):
            st.caption(f"⭐ {rating:.1f}/10")
        st.write(str(row["overview"])[:500])
        left, right = st.columns(2)
        with left:
            if st.button("👎 Pass", use_container_width=True, key=f"pass_{pos}"):
                st.session_state["passes"].append(idx)
                st.session_state["deck_pos"] += 1
                st.rerun()
        with right:
            if st.button("💚 Like", type="primary", use_container_width=True, key=f"like_{pos}"):
                st.session_state["likes"].append(idx)
                st.session_state["deck_pos"] += 1
                st.rerun()
    else:
        st.success("Discovery round complete!")
        liked = st.session_state["likes"]
        st.write(f"You liked {len(liked)} of {len(deck)} movies.")
        if liked:
            st.subheader("Picked for your taste")
            for i, score in recommend(df, matrix, liked, 6, st.session_state["passes"]):
                show_movie(df, i, score)
        else:
            st.info("No likes yet. Restart the round and like a few movies.")
        if st.button("Restart discovery"):
            for key in ["deck", "deck_pos", "likes", "passes"]:
                st.session_state.pop(key, None)
            st.rerun()

with group_tab:
    st.subheader("👥 What should we watch together?")
    st.write("Add each person's favourite movie and any movies they do not want. Recommendations aim to balance everyone's taste while avoiding stated dislikes.")
    st.caption("Each participant chooses one seed movie for this prototype. You can extend this to several favourites per person later.")
    people = st.number_input("Number of people", min_value=2, max_value=3, value=2, step=1)
    participant_seeds = []
    participant_dislikes = []
    title_options = df["title"].fillna("Untitled").astype(str).tolist()
    for person_num in range(1, int(people) + 1):
        with st.expander(f"Person {person_num}", expanded=True):
            seed_title = st.selectbox(
                f"Person {person_num}'s favourite movie",
                title_options,
                index=title_options.index("Avatar") if person_num == 1 and "Avatar" in title_options else 0,
                key=f"group_seed_{person_num}",
            )
            dislike_titles = st.multiselect(
                f"Movies Person {person_num} dislikes (optional)",
                title_options,
                key=f"group_dislikes_{person_num}",
            )
            seed_idx = int(df.index[df["title"].fillna("").astype(str) == seed_title][0])
            participant_seeds.append(seed_idx)
            participant_dislikes.extend(
                int(i) for i in df.index[df["title"].fillna("").astype(str).isin(dislike_titles)]
            )
    group_count = st.slider("Group recommendations", 3, 10, 5, key="group_count")
    if st.button("Find a movie for everyone 🤝", type="primary", key="group_btn"):
        # Joint score: average similarity to each person's chosen favourite.
        # Exclude every seed and every explicitly disliked movie.
        group_results = recommend(
            df, matrix, participant_seeds, group_count,
            excluded=set(participant_dislikes)
        )
        st.session_state["group_results"] = group_results
        st.session_state["group_seed_titles"] = [
            str(df.iloc[i]["title"]) for i in participant_seeds
        ]
    if "group_results" in st.session_state:
        seeds_text = ", ".join(st.session_state.get("group_seed_titles", []))
        st.caption(f"Balancing preferences from: {seeds_text}")
        for i, score in st.session_state.get("group_results", []):
            show_movie(df, i, score)

with stream_tab:
    st.subheader("📺 Streaming availability")
    st.write("Streaming catalogues change by country and date. This panel is ready for a provider API, but it will not guess where a title is available.")
    st.info("To show verified Netflix / Prime Video / Disney+ availability, configure a streaming-provider API key and region. Do not add a secret API key directly to a public GitHub file.")
    st.markdown(
        "For now, use [JustWatch](https://www.justwatch.com/) to verify availability. "
        "A future API integration can show provider badges here once an API key and permitted data source are configured."
    )
    stream_title = st.selectbox(
        "Find a title to check",
        df["title"].fillna("Untitled").astype(str).tolist(),
        key="stream_title"
    )
    st.write(f"Selected title: **{stream_title}**")
    st.link_button(
        "Check streaming availability on JustWatch",
        "https://www.justwatch.com/"
    )

st.divider()
st.caption("FlickFusion · Content-based recommendations powered by TMDB 5000 metadata. Match scores are similarity estimates, not ratings.")
