# 🎬 AI Movie Recommendation System

This is an AI-powered movie recommendation system that helps users find movies based on their past ratings and interests using natural language search. Built with Streamlit and powered by NLP techniques like TF-IDF and cosine similarity.

👉 **Try it live:**  
[https://ai-movie-recommendation-system-h7ow2ahcckt2tjyyuehxai.streamlit.app](https://ai-movie-recommendation-system-h7ow2ahcckt2tjyyuehxai.streamlit.app)

---

## 🚀 Features

- 🔍 Search by title, keyword, or genre (e.g. "action", "romance", "space")
- 📊 Personalized recommendations based on user ratings
- 🧠 Uses TF-IDF and cosine similarity to compute content-based similarity
- 🖼 Fetches movie posters using the OMDb API
- 🎨 Beautiful custom Streamlit UI with Clemson-themed background
- ✅ Easy to run locally or deploy on Streamlit Cloud

---

## 🧠 How It Works

1. **Data Loading**: Loads movie metadata and user ratings from CSV files
2. **Filtering & Merging**: Combines the datasets and processes them
3. **Recommendation**: Uses TF-IDF on movie overviews and cosine similarity to find movies similar to the user's rated content
4. **Poster Fetching**: Queries OMDb API to retrieve movie posters using IMDb IDs
5. **UI**: Interactive frontend built with Streamlit

---

## 📁 Files

- `app.py` – Main Streamlit application
- `movies_metadata.csv` – Movie information dataset (from Kaggle)
- `ratings.csv` – User ratings dataset (from Kaggle)
- `README.md` – You’re reading it!

---

## 🛠 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-movie-recommendation-system.git
cd ai-movie-recommendation-system
