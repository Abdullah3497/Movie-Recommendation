import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset (must have 'title' and 'tags' columns)
movies = pd.read_csv('final_movies.csv')

# Vectorize 'tags' column using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
vectors = tfidf.fit_transform(movies['tags'])

# Normalize vectors
norm_vectors = normalize(vectors)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
movies['cluster'] = kmeans.fit_predict(norm_vectors)

# Recommendation function
def recommend_movies(movie_title):
    idx = movies[movies['title'].str.lower() == movie_title.lower()].index
    if len(idx) == 0:
        return []
    
    idx = idx[0]
    movie_vector = norm_vectors[idx]
    similarity = cosine_similarity(movie_vector, norm_vectors)
    sim_scores = list(enumerate(similarity[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [i for i, _ in sim_scores[1:6]]
    return movies.iloc[recommended_indices]['title'].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation ")

movie_list = movies['title'].sort_values().unique()
selected_movie = st.selectbox("Select a movie you like:", movie_list)

if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend_movies(selected_movie)
        if not recommendations:
            st.warning("Movie not found in dataset.")
        else:
            st.success("Top 5 similar movies:")
            for title in recommendations:
                st.write(f"✅ {title}")
