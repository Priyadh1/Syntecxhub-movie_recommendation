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
