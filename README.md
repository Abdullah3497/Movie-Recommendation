This project is a movie recommendation system built using KMeans clustering, TF-IDF vectorization, and a Streamlit UI. It recommends similar movies based on the description using unsupervised learning techniques.

📌 Features
📖 Recommend movies based on descriptions using cosine similarity

🔍 TF-IDF for text vectorization

🔗 KMeans clustering to group similar movies

🌐 Streamlit web app for a clean user interface

🧠 No labels required (Unsupervised)

📁 Dataset
final_movies.csv

Must contain at least:

title: movie title

description: short description or overview

⚙️ How It Works
TF-IDF Vectorization: Converts movie descriptions into numerical form.

Normalization: Ensures fair distance calculation.

KMeans Clustering: Groups similar descriptions into clusters.

Cosine Similarity: Finds the most similar movies to the selected one.
