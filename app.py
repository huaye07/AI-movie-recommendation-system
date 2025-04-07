import pandas as pd
import re
import pickle
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv')

# Preprocessing
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)

merged_data = pd.merge(ratings, movies[['id', 'original_title', 'overview', 'imdb_id']],
                       left_on='movieId', right_on='id', how='inner')
merged_data = merged_data[['userId', 'original_title', 'overview', 'rating', 'imdb_id']]
merged_data['overview'] = merged_data['overview'].fillna('')

# Helper functions
def fetch_poster(imdb_id, api_key="61d9a9ee"):
    if pd.isna(imdb_id) or not imdb_id:
        return None
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True" and data.get("Poster") and data.get("Poster") != "N/A":
                return data.get("Poster")
    except:
        return None
    return None

poster_cache = {}

def fetch_poster_with_cache(imdb_id, api_key="61d9a9ee"):
    if imdb_id in poster_cache:
        return poster_cache[imdb_id]
    poster_url = fetch_poster(imdb_id, api_key)
    poster_cache[imdb_id] = poster_url
    return poster_url

def recommend_movies(user_id, search_query, merged_data):
    user_ratings = merged_data[merged_data['userId'] == user_id]
    if user_ratings.empty:
        return pd.DataFrame()

    filtered_movies = merged_data[
        merged_data['original_title'].str.contains(search_query, case=False, na=False) |
        merged_data['overview'].str.contains(search_query, case=False, na=False)
    ].copy()

    if filtered_movies.empty:
        top_rated_movies = user_ratings.sort_values(by='rating', ascending=False)
        return top_rated_movies[['original_title', 'overview', 'rating', 'imdb_id']].head(10)

    all_movies = pd.concat([user_ratings[['original_title', 'overview']],
                            filtered_movies[['original_title', 'overview']]])
    tfidf = TfidfVectorizer(stop_words='english', max_features=100000, ngram_range=(1, 5))
    tfidf_matrix = tfidf.fit_transform(all_movies['overview'])

    user_movie_indices = range(len(user_ratings))
    cosine_similarities = cosine_similarity(tfidf_matrix[user_movie_indices], tfidf_matrix[len(user_ratings):])

    if cosine_similarities.shape[0] == 0:
        return pd.DataFrame()

    avg_similarities = cosine_similarities.mean(axis=0)
    filtered_movies['similarity_score'] = avg_similarities

    top_recommendations = filtered_movies.sort_values(by='similarity_score', ascending=False)
    return top_recommendations[['original_title', 'overview', 'similarity_score', 'imdb_id']].drop_duplicates('original_title').head(10)

# Streamlit UI
theme_css = """
<style>
body {
    background-image: url('https://images.unsplash.com/photo-1606312618775-003bfe2f4c92');
    background-size: cover;
    background-attachment: fixed;
}
.title {
    font-size: 50px;
    text-align: center;
    font-family: 'Bebas Neue', cursive;
    color: #ff4c4c;
    padding-top: 20px;
}
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #f0f0f0;
    margin-bottom: 30px;
    font-family: 'Segoe UI', sans-serif;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">
"""

st.markdown(theme_css, unsafe_allow_html=True)

st.markdown('<div class="title">Recommendations For You 🍿</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tell us what you feel like watching and we\'ll do the rest 🎬</div>', unsafe_allow_html=True)

user_id = st.number_input("Your User ID", min_value=1, max_value=1000, help="Choose a user ID between 1 and 1000")
search_query = st.text_input("Movie title, keyword, or genre", help="Search something like 'action', 'romance', 'space', etc.")

if st.button('Get Recommendations'):
    recommendations = recommend_movies(user_id, search_query, merged_data)
    recommendations = recommendations.reset_index(drop=True)
    recommendations['poster_url'] = recommendations['imdb_id'].apply(fetch_poster_with_cache)

    st.markdown("---")
    for _, row in recommendations.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            if row['poster_url']:
                st.image(row['poster_url'], use_column_width=True)
            else:
                st.write("No Image Available")
        with col2:
            st.subheader(row['original_title'])
            st.write(row['overview'])
            st.caption(f"Similarity Score: {row.get('similarity_score', 0):.2f}")
