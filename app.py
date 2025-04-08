import os
import re
import string
import joblib
import pandas as pd
import requests
from flask import Flask, render_template, request, flash, redirect, url_for

app = Flask(__name__)

# Load models and movie data
api_key_env = os.environ.get('TMDB_API_KEY')
movies_data = joblib.load('Model/movies_data.joblib')
similarity = joblib.load('Model/similarity.joblib')
sentiment_model = joblib.load('Model/sentiment_analysis_model.pkl')
vectorizer = joblib.load('Model/tfidf_vectorizer.pkl')
movies_df = pd.DataFrame(movies_data)

# Utility functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(review):
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review])
    sentiment = sentiment_model.predict(vectorized_review)[0]
    return "positive" if sentiment == 1 else "negative"

def poster(movie_id):
    key = api_key_env if api_key_env is not None else '25408b9a528f2fe822b3c26614771054'
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.exceptions.RequestException:
        return ""
    return ""

def get_movie_reviews(movie_id):
    key = api_key_env if api_key_env is not None else '25408b9a528f2fe822b3c26614771054'
    reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={key}"
    try:
        response = requests.get(reviews_url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        reviews = [{"author": review.get("author", "Unknown"), "content": review.get("content", "")}
                   for review in data.get("results", [])]
        return reviews
    except requests.exceptions.RequestException as e:
        flash(f"Error fetching reviews: {e}", 'error')
        return []

def recommend(movie):
    try:
        movie_index = movies_df[movies_df['title'] == movie].index[0]
    except IndexError:
        return [], []
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    recommended_movies = []
    recommended_posters = []
    for i in movies_list:
        rec_movie = movies_df.iloc[i[0]]
        recommended_movies.append(rec_movie.title)
        recommended_posters.append(poster(rec_movie.id))
    return recommended_movies, recommended_posters

# Register analyze_sentiment in Jinja environment so we can call it from the template
app.jinja_env.globals['analyze_sentiment'] = analyze_sentiment

@app.route("/", methods=["GET"])
def index():
    selected_movie = request.args.get("selected_movie", default="")
    movie_list = sorted(movies_df["title"].unique().tolist())
    
    if selected_movie:
        try:
            movie_row = movies_df[movies_df['title'] == selected_movie].iloc[0]
        except IndexError:
            flash("Movie not found.", "error")
            return redirect(url_for("index"))
        
        movie_info = {
            "id": movie_row.id,
            "title": selected_movie,
            "overview": movie_row.overview,
            "genre": movie_row.genres,
            "cast": movie_row.cast,
            "crew": movie_row.crew,
            "release_date": movie_row.release_date,
            "rating": movie_row.vote_average,
            "poster": poster(movie_row.id)
        }
        reviews = get_movie_reviews(movie_row.id)
        rec_names, rec_posters = recommend(selected_movie)
        recommendations = [{"name": name, "poster": poster_url} for name, poster_url in zip(rec_names, rec_posters)]
    else:
        movie_info = None
        reviews = []
        recommendations = []
    
    return render_template(
        "index.html",
        movie_list=movie_list,
        selected_movie=selected_movie,
        movie_info=movie_info,
        reviews=reviews,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run()