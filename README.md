# 🎬 Movie Recommender App

An interactive, content-based movie recommendation web application built with **Streamlit**, **Scikit-learn**, and **Pandas**. The recommender analyzes plot overviews, genres, keywords, cast, and crew metadata from the TMDB 5000 dataset to calculate cosine similarity between movies.

## ✨ Features

- **Content-Based Filtering**: Recommends movies based on shared plot themes, genres, keywords, actors, and directors.
- **Efficient Inference**: Pre-computes top-K neighbors using vector similarity to keep recommendations fast and memory-friendly.
- **Dynamic Posters**: Integrates with the TMDB API to fetch high-resolution movie posters in real time.
- **Interactive UI**: Searchable dropdowns, configurable recommendation limits, and movie overview expanders.
