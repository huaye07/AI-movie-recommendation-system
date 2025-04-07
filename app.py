import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st
import requests

# Load datasets
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv')

# Convert 'id' column in movies and 'movieId' column in ratings to the same type
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])  # Drop rows where 'id' is NaN
movies['id'] = movies['id'].astype(int)  # Convert to integer after dropping NaN
ratings['movieId'] = ratings['movieId'].astype(int)

# Merge datasets based on movieId
merged_data = pd.merge(ratings, movies[['id', 'original_title', 'overview', 'imdb_id']], 
                       left_on='movieId', right_on='id', how='inner')

# Filter relevant columns
merged_data = merged_data[['userId', 'original_title', 'overview', 'rating', 'imdb_id']]

# Handle NaN values in the 'overview' column by replacing NaN with an empty string
merged_data['overview'] = merged_data['overview'].fillna('')

# Function to check if a string contains only English characters (no non-English characters)
def is_english_title(title):
    return bool(re.match('^[A-Za-z0-9\\s:;,.!?()\\-]+$', title))
# Function to fetch movie posters from OMDb API
def fetch_poster(imdb_id, api_key="61d9a9ee"):
    if pd.isna(imdb_id) or not imdb_id:
        return None  # Return None if IMDb ID is invalid
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True" and data.get("Poster") and data.get("Poster") != "N/A":
                return data.get("Poster")
    except Exception as e:
        print(f"Error fetching poster for IMDb ID {imdb_id}: {e}")
    return None


poster_cache = {}

def fetch_poster_with_cache(imdb_id, api_key="61d9a9ee"):
    if imdb_id in poster_cache:
        return poster_cache[imdb_id]
    poster_url = fetch_poster(imdb_id, api_key)
    poster_cache[imdb_id] = poster_url
    return poster_url

def recommend_movies(user_id, search_query, merged_data):
    # Filter movies rated by the user
    user_ratings = merged_data[merged_data['userId'] == user_id]
    
    if user_ratings.empty:
        return pd.DataFrame()  # Return an empty dataframe if no ratings are found

    # Filter movies based on the search query
    filtered_movies = merged_data[
        merged_data['original_title'].str.contains(search_query, case=False, na=False) | 
        merged_data['overview'].str.contains(search_query, case=False, na=False)
    ].copy()  # Create a copy to avoid SettingWithCopyWarning

    if filtered_movies.empty:
        # If no movies match the search query, recommend the top-rated movies from user's history
        top_rated_movies = user_ratings.sort_values(by='rating', ascending=False)
        return top_rated_movies[['original_title', 'overview', 'rating', 'imdb_id']].head(10)

    # Combine user-rated movies and filtered movies for similarity calculation
    all_movies = pd.concat([user_ratings[['original_title', 'overview']], filtered_movies[['original_title', 'overview']]])
    tfidf = TfidfVectorizer(stop_words='english', max_features=100000, ngram_range=(1, 5))
    tfidf_matrix = tfidf.fit_transform(all_movies['overview'])

    # Compute cosine similarity for the user-rated movies with filtered movies
    user_movie_indices = range(len(user_ratings))  # Indices of user-rated movies in the combined list
    cosine_similarities = cosine_similarity(tfidf_matrix[user_movie_indices], tfidf_matrix[len(user_ratings):])

    if cosine_similarities.shape[0] == 0:  # If there are no valid similarity scores
        return pd.DataFrame()  # Return an empty dataframe

    # Calculate the average similarity score for each of the filtered movies
    avg_similarities = cosine_similarities.mean(axis=0)
    filtered_movies['similarity_score'] = avg_similarities  # Safely assign new column

    # Sort movies based on similarity score and recommend the top 10
    top_recommendations = filtered_movies.sort_values(by='similarity_score', ascending=False)
    return top_recommendations[['original_title', 'overview', 'similarity_score', 'imdb_id']].drop_duplicates(subset='original_title').head(10)

theme_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');

html, body, .stApp {
    height: 100%;
    margin: 0;
    background-color: #6a1b9a;  /* Deep purple background */
    background-image: url('https://1000logos.net/wp-content/uploads/2022/11/Clemson-Tigers-Logo.png');
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    font-family: 'Outfit', sans-serif;
}

.stApp, .stApp > div, .stApp > div > div {
    background: transparent !important;
}

.title {
    font-size: 60px;
    text-align: center;
    color: #ffffff;  /* White color for better contrast against the purple background */
    padding-top: 40px;
    padding-bottom: 20px;
    text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.6);  /* Dark shadow for better readability */
}

.subtitle {
    font-size: 24px;
    text-align: center;
    color: #ecf0f1;  /* Light grayish color for subtitle */
    margin-bottom: 40px;
    font-weight: 300;
}

.stButton > button {
    background-color: #9b59b6;  /* Lighter purple button to match the theme */
    color: white;
    font-weight: bold;
    padding: 0.6rem 1.2rem;
    border-radius: 12px;
    border: none;
    transition: all 0.3s ease-in-out;
}

.stButton > button:hover {
    background-color: #8e44ad;  /* Darker purple when hovered */
    transform: scale(1.05);
}

/* Input box container styling */
.stTextInput > div > div,
.stNumberInput > div > div {
    background-color: transparent !important; /* Make container fully transparent */
}

/* Input field styling */
.stTextInput input,
.stNumberInput input {
    background-color: rgba(107, 27, 154, 0.7) !important; /* Purple background with 70% opacity */
    color: white !important; /* White text color */
    border-radius: 8px !important; /* Rounded corners */
    border: 2px solid #9b59b6 !important; /* Purple border */
    padding: 8px 12px !important; /* Internal spacing */
}

/* Focus state styling */
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #8e44ad !important; /* Darker purple border on focus */
    box-shadow: 0 0 0 2px rgba(142, 68, 173, 0.5) !important; /* Glow effect */
}

/* Placeholder text styling */
.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: rgba(255, 255, 255, 0.6) !important; /* Semi-transparent white placeholder text */
}
.movie-card {
    background-color: rgba(0, 0, 0, 0.5);  /* Semi-transparent dark background for cards */
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

st.markdown('<div class="title">Movie Recommendations For You 🍿</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tell us what you feel like watching and we\'ll do the rest 🎬</div>', unsafe_allow_html=True)
 
user_id = st.number_input("Your User ID", min_value=1, max_value=1000, help="Choose a user ID between 1 and 1000")
search_query = st.text_input("Movie title, keyword, or genre", help="Search something like 'action', 'romance', 'space', etc.")

# Pre-fetch poster URLs for recommendations
def fetch_posters_for_recommendations(recommended_movies):
    """Fetch posters for all recommended movies."""
    recommended_movies['poster_url'] = recommended_movies['imdb_id'].apply(fetch_poster_with_cache)
    return recommended_movies

# Button to generate recommendations
if st.button('Get Recommendations'):
    with st.spinner('Finding the best movies for you... 🎬'):
        # Get recommended movies
        recommended_movies = recommend_movies(user_id, search_query, merged_data)

        # Pre-fetch all poster URLs
        recommended_movies = fetch_posters_for_recommendations(recommended_movies)

        # Reset index for recommendations
        recommended_movies = recommended_movies.reset_index(drop=True)

        st.write("Top Recommended Movies for You:")

        for _, row in recommended_movies.iterrows():
            with st.container():
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 4])

                with col1:
                    if row['poster_url'] and row['poster_url'].lower() != "n/a":
                        st.image(row['poster_url'], width=200)
                    else:
                        st.text("No Image")

                with col2:
                    st.markdown(f"### {row['original_title']}")
                    st.write(row['overview'])

                st.markdown('</div>', unsafe_allow_html=True)

    st.success("Recommendations loaded successfully! ✅")

    # Save the model components using pickle
    with open('movie_recommendation_model.pkl', 'wb') as model_file:
        pickle.dump({'tfidf': TfidfVectorizer, 'movies': merged_data}, model_file)
