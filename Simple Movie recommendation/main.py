import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    "Movie_ID": [1, 2, 3, 4, 5],
    "Title": ["Inception", "Interstellar", "The Dark Knight", "Avatar", "Titanic"],
    "Genre": ["Sci-Fi, Action", "Sci-Fi, Drama", "Action, Crime", "Fantasy, Adventure", "Romance, Drama"],
    "Director": ["Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "James Cameron", "James Cameron"],
    "Cast": ["Leonardo DiCaprio", "Matthew McConaughey", "Christian Bale", "Sam Worthington", "Leonardo DiCaprio"]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Movie Dataset:\n", df)

# Combine features into a single string for each movie
df['Combined_Features'] = df['Genre'] + " " + df['Director'] + " " + df['Cast']

# Vectorize combined features
vectorizer = CountVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['Combined_Features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Recommend movies
def recommend_movies(movie_title, top_n=3):
    if movie_title not in df['Title'].values:
        return "Movie not found in the dataset."
    
    movie_idx = df[df['Title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommendations = [df.iloc[i[0]]['Title'] for i in sorted_scores]
    return recommendations

# Test the recommendation system
input_movie = "Inception"
recommendations = recommend_movies(input_movie)

print(f"\nMovies recommended for '{input_movie}': {recommendations}")

# Save dataset with cosine similarity matrix
cosine_df = pd.DataFrame(cosine_sim, columns=df['Title'], index=df['Title'])
cosine_df.to_csv("movie_recommendation_similarity.csv", index=True)
print("\nSimilarity matrix saved to 'movie_recommendation_similarity.csv'.")
