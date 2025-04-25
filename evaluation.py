import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import requests
from tqdm import tqdm
import numpy as np

# Load movie metadata and user ratings
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv')

# Ensure consistent ID types for merging
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)

# Merge datasets on movie ID
merged_data = pd.merge(
    ratings,
    movies[['id', 'original_title', 'overview', 'imdb_id']],
    left_on='movieId',
    right_on='id',
    how='inner'
)
merged_data = merged_data[['userId', 'original_title', 'overview', 'rating', 'imdb_id']]
merged_data['overview'] = merged_data['overview'].fillna('')

# Check if a title is in English (basic check using regex)
def is_english_title(title):
    return bool(re.match('^[A-Za-z0-9\\s:;,.!?()\\-]+$', title))

# Fetch movie poster from OMDb API
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
    except Exception as e:
        print(f"Error fetching poster for IMDb ID {imdb_id}: {e}")
    return None

# Poster cache to reduce API calls
poster_cache = {}

def fetch_poster_with_cache(imdb_id, api_key="61d9a9ee"):
    if imdb_id in poster_cache:
        return poster_cache[imdb_id]
    poster_url = fetch_poster(imdb_id, api_key)
    poster_cache[imdb_id] = poster_url
    return poster_url

# Recommend movies for a specific user based on TF-IDF cosine similarity
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

    # Build TF-IDF matrix for combined movies (user history + search results)
    all_movies = pd.concat([
        user_ratings[['original_title', 'overview']],
        filtered_movies[['original_title', 'overview']]
    ])
    tfidf = TfidfVectorizer(stop_words='english', max_features=100000, ngram_range=(1, 5))
    tfidf_matrix = tfidf.fit_transform(all_movies['overview'])

    user_movie_indices = range(len(user_ratings))
    cosine_similarities = cosine_similarity(tfidf_matrix[user_movie_indices], tfidf_matrix[len(user_ratings):])

    if cosine_similarities.shape[0] == 0:
        return pd.DataFrame()

    avg_similarities = cosine_similarities.mean(axis=0)
    filtered_movies['similarity_score'] = avg_similarities

    top_recommendations = filtered_movies.sort_values(by='similarity_score', ascending=False)
    return top_recommendations[['original_title', 'overview', 'similarity_score', 'imdb_id']].drop_duplicates(subset='original_title').head(10)

# Generate top recommendations for all users
def generate_recommendations_for_all_users(merged_data, max_users=None):
    recommendations = []
    user_ids = merged_data['userId'].unique()
    if max_users:
        user_ids = user_ids[:max_users]

    print(f"Generating recommendations for {len(user_ids)} users...")

    for user_id in tqdm(user_ids, desc="Recommending"):
        user_recommendations = recommend_movies(user_id, '', merged_data)
        
        if user_recommendations.empty:
            continue

        user_recommendations['liked'] = 1  # Assume all recommended movies are liked
        user_recommendations['userId'] = user_id  # Add user ID

        recommendations.append(user_recommendations)

    return pd.concat(recommendations, ignore_index=True)

# Evaluate precision, recall, F1, Hit@k, and NDCG@k
def evaluate_recommendations(recommended_movies, merged_data, like_threshold=3, top_k=10):
    # Create ground truth: which movies each user liked
    true_relevant = merged_data[['userId', 'original_title', 'rating']].copy()
    true_relevant['liked'] = (true_relevant['rating'] >= like_threshold).astype(int)

    # Normalize movie titles for reliable merging
    true_relevant['original_title'] = true_relevant['original_title'].str.strip().str.lower()
    recommended_movies['original_title'] = recommended_movies['original_title'].str.strip().str.lower()

    # Mark all recommended items with recommended=1
    recommended_movies['recommended'] = 1

    # Merge to align recommendations with true relevance
    eval_df = pd.merge(
        true_relevant,
        recommended_movies[['userId', 'original_title', 'recommended']],
        on=['userId', 'original_title'],
        how='left'
    )

    eval_df['recommended'] = eval_df['recommended'].fillna(0).astype(int)

    # Extract true/false labels
    y_true = eval_df['liked']
    y_pred = eval_df['recommended']

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    #recall = recall_score(y_true, y_pred)
    #f1 = f1_score(y_true, y_pred)
    hit_at_k = eval_hit_at_k(eval_df, top_k)
    ndcg_at_k = eval_ndcg_at_k(eval_df, top_k)

    print("\n📊 Evaluation Metrics for Recommendations:")
    print(f"Precision:  {precision:.4f}")
    #print(f"Recall:     {recall:.4f}")
    #print(f"F1 Score:   {f1:.4f}")
    print(f"Hit@{top_k}:  {hit_at_k:.4f}")
    print(f"NDCG@{top_k}: {ndcg_at_k:.4f}")

# Compute Hit@k metric
def eval_hit_at_k(eval_df, top_k=10):
    hits = 0
    for user_id in eval_df['userId'].unique():
        user_data = eval_df[eval_df['userId'] == user_id]
        top_k_recommended = user_data.sort_values(by='recommended', ascending=False).head(top_k)
        hits += top_k_recommended['liked'].sum()
    return hits / len(eval_df['userId'].unique())

# Compute NDCG@k metric
def eval_ndcg_at_k(eval_df, top_k=10):
    dcgs = 0
    idcgs = 0
    for user_id in eval_df['userId'].unique():
        user_data = eval_df[eval_df['userId'] == user_id]
        top_k_recommended = user_data.sort_values(by='recommended', ascending=False).head(top_k)
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(top_k_recommended['liked'])])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(top_k, user_data['liked'].sum()))])
        dcgs += dcg
        idcgs += idcg
    return dcgs / idcgs if idcgs > 0 else 0.0

# Run the pipeline
recommended_movies = generate_recommendations_for_all_users(merged_data, max_users=4500)
evaluate_recommendations(recommended_movies, merged_data)

# Save model data
with open('movie_recommendation_model.pkl', 'wb') as model_file:
    pickle.dump({'tfidf': TfidfVectorizer, 'movies': merged_data}, model_file)
