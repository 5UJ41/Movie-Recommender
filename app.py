"""Streamlit front-end for the content-based movie recommender."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ART = Path(__file__).parent / "artifacts"
IMG_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://placehold.co/500x750/1f2937/9ca3af?text=No+Poster"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_data
def load_artifacts():
    movies = pd.read_csv(ART / "movies.csv")
    data = np.load(ART / "neighbors.npz")
    return movies, data["neighbors"], data["scores"]


def get_api_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.environ.get("TMDB_API_KEY")


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_poster(movie_id: int, api_key: str | None) -> str:
    if not api_key:
        return PLACEHOLDER
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
            params={"api_key": api_key},
            timeout=5,
        )
        r.raise_for_status()
        path = r.json().get("poster_path")
        return IMG_BASE + path if path else PLACEHOLDER
    except requests.RequestException:
        return PLACEHOLDER


def recommend(title: str, n: int):
    movies, neighbors, scores = load_artifacts()
    i = movies.index[movies["title"] == title][0]
    return movies.iloc[neighbors[i][:n]].assign(score=scores[i][:n])


# ---------- UI ----------
st.title("🎬 Movie Recommender")
st.caption("Pick a movie you like and get similar ones, based on plot, genres, keywords, cast and director.")

try:
    movies, _, _ = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python build_model.py` first (see README).")
    st.stop()

col_a, col_b = st.columns([3, 1])
with col_a:
    choice = st.selectbox(
        "Movie",
        movies["title"].tolist(),
        index=None,
        placeholder="Type to search, e.g. Batman Begins",
    )
with col_b:
    n = st.slider("How many?", 3, 10, 5)

if st.button("Recommend", type="primary", disabled=choice is None):
    recs = recommend(choice, n)
    key = get_api_key()
    with ThreadPoolExecutor(max_workers=8) as ex:
        posters = list(ex.map(lambda m: fetch_poster(m, key), recs["movie_id"]))

    st.subheader(f"Because you liked *{choice}*")
    per_row = 5
    rows = [range(i, min(i + per_row, len(recs))) for i in range(0, len(recs), per_row)]
    for row in rows:
        cols = st.columns(per_row)
        for col, j in zip(cols, row):
            r = recs.iloc[j]
            with col:
                st.image(posters[j], use_container_width=True)
                st.markdown(f"**{r['title']}**")
                st.caption(f"{r['genres']}  ·  match {r['score']:.0%}")
                with st.expander("Overview"):
                    st.write(r["overview"])

if not get_api_key():
    st.info("No TMDB API key set, so posters show as placeholders. See README to add one.")
