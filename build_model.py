
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 20  # neighbours stored per movie (the app can show up to this many)
ps = PorterStemmer()


def names(obj):
    """All 'name' values from a JSON-like string column."""
    return [d["name"] for d in ast.literal_eval(obj)]


def top3_names(obj):
    return [d["name"] for d in ast.literal_eval(obj)[:3]]


def director(obj):
    for d in ast.literal_eval(obj):
        if d["job"] == "Director":
            return [d["name"]]
    return []


def squash(words):
    """'Science Fiction' -> 'ScienceFiction' so multi-word names stay one token."""
    return [w.replace(" ", "") for w in words]


def stem(text):
    return " ".join(ps.stem(w) for w in text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", default="data/tmdb_5000_movies.csv")
    ap.add_argument("--credits", default="data/tmdb_5000_credits.csv")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    movies = pd.read_csv(args.movies)
    credits = pd.read_csv(args.credits)
    df = movies.merge(credits, on="title")
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    df = df.dropna().drop_duplicates(subset="title").reset_index(drop=True)

    genres = df["genres"].apply(names)
    keywords = df["keywords"].apply(names)
    cast = df["cast"].apply(top3_names)
    crew = df["crew"].apply(director)
    overview_words = df["overview"].apply(str.split)

    tags = (
        overview_words
        + genres.apply(squash)
        + keywords.apply(squash)
        + cast.apply(squash)
        + crew.apply(squash)
    )
    tags = tags.apply(lambda x: " ".join(x).lower()).apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(tags)  # sparse, no .toarray() needed
    sim = cosine_similarity(vectors)
    np.fill_diagonal(sim, -1.0)  # a movie should never recommend itself

    # top-K neighbours per movie, best first
    idx = np.argpartition(-sim, TOP_K, axis=1)[:, :TOP_K]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, idx], axis=1)
    neighbors = idx[rows, order].astype(np.int32)
    scores = sim[rows, neighbors].astype(np.float32)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "movie_id": df["movie_id"],
            "title": df["title"],
            "genres": genres.apply(lambda g: ", ".join(g)),
            "overview": df["overview"],
        }
    ).to_csv(out / "movies.csv", index=False)
    np.savez_compressed(out / "neighbors.npz", neighbors=neighbors, scores=scores)

    print(f"{len(df)} movies -> {out}/movies.csv, {out}/neighbors.npz")
    first = df.loc[0, "title"]
    print(f"Sanity check, top 5 for '{first}':", [df.loc[i, "title"] for i in neighbors[0][:5]])


if __name__ == "__main__":
    main()
