import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load the dataset
df_with_cnt = pd.read_csv('df_with_cnt.csv')

# Create the pivot table for the recommendation system
book_pivot = df_with_cnt.pivot_table(columns='User-ID', index='Book-Title', values='Book-Rating', aggfunc='mean')
# book_pivot = df_with_cnt.pivot_table(columns='User-ID', index='Book-Title', values='Book-Rating')
book_pivot.fillna(0, inplace=True)

# Convert the pivot table to a sparse matrix
book_sparse = csr_matrix(book_pivot)

# Train the Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
# model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# Function to fetch poster URLs for recommendations
def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion:
        # Retrieve the URL for the recommended book
        book_name = book_pivot.index[book_id]
        url = df_with_cnt[df_with_cnt['Book-Title'] == book_name]['Image-URL-L'].values[0]
        poster_url.append(url)
    return poster_url

# Function to recommend books
def recommend_book(book_name):
    # Find the index of the given book
    book_id = np.where(book_pivot.index == book_name)[0][0]
    # Get the nearest neighbors
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    # Get the list of recommended book titles (excluding the first one to avoid recommending the same book)
    books_list = [book_pivot.index[suggestion[0][i]] for i in range(1, len(suggestion[0]))]
    
    # Fetch poster URLs for the recommended books (also exclude the first one)
    poster_url = fetch_poster(suggestion[0][1:])
    
    return books_list, poster_url

# Streamlit app setup
st.title("Book Recommendation System")
st.write("Find similar books based on what you like!")

# List of book titles for the dropdown
book_titles = df_with_cnt['Book-Title'].drop_duplicates().sort_values().tolist()

# User input for book selection
selected_books = st.selectbox("Type or select a book from the dropdown", book_titles)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_books[0])
        st.image(poster_url[0])
    with col2:
        st.text(recommended_books[1])
        st.image(poster_url[1])

    with col3:
        st.text(recommended_books[2])
        st.image(poster_url[2])
    with col4:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col5:
        st.text(recommended_books[4])
        st.image(poster_url[4])


