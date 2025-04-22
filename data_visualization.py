import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set visual style
sns.set(style="whitegrid")

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

# Create a new column: number of words in movie overview
merged_data['overview_length'] = merged_data['overview'].apply(lambda x: len(x.split()))

# 1. Distribution of user ratings
plt.figure(figsize=(8, 5))
sns.histplot(merged_data['rating'], bins=10, kde=False)
plt.title('Distribution of User Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')
plt.show()

# 2. Top 10 most rated movies
top_rated = (
    merged_data.groupby('original_title')['rating']
    .count()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_rated.values, y=top_rated.index)
plt.title('Top 10 Most Rated Movies')
plt.xlabel('Number of Ratings')
plt.ylabel('Movie Title')
plt.show()

# 3. Average rating of top 10 most frequently rated movies
avg_rating = (
    merged_data.groupby('original_title')['rating']
    .agg(['count', 'mean'])
    .sort_values(by='count', ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_rating['mean'], y=avg_rating.index)
plt.title('Average Rating of Top 10 Most Rated Movies')
plt.xlabel('Average Rating')
plt.ylabel('Movie Title')
plt.show()

# 4. Distribution of overview lengths
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['overview_length'], bins=30, kde=True)
plt.title('Distribution of Overview Lengths')
plt.xlabel('Number of Words in Overview')
plt.ylabel('Number of Movies')
plt.show()

# 5. Word cloud of all movie overviews
all_text = ' '.join(merged_data['overview'].dropna().tolist())

wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(all_text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Movie Overviews')
plt.show()

# 6. Correlation heatmap
# Only keep numerical columns for correlation
correlation_data = merged_data[['rating', 'overview_length']]
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap: Rating vs Overview Length')
plt.show()
