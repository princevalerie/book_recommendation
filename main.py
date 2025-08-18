import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import random

# Load the dataset
@st.cache_data
def load_data():
    df_with_cnt = pd.read_csv('df_with_cnt.csv')
    return df_with_cnt

df_with_cnt = load_data()

# Create the pivot table for the recommendation system
@st.cache_data
def create_pivot_table(df):
    book_pivot = df.pivot_table(
        columns='User-ID', 
        index='Book-Title', 
        values='Book-Rating', 
        aggfunc='mean'
    )
    book_pivot.fillna(0, inplace=True)
    
    # Filter buku dengan minimal interaksi
    book_counts = (book_pivot > 0).sum(axis=1)
    popular_books = book_counts[book_counts >= 3].index
    book_pivot_filtered = book_pivot.loc[popular_books]
    
    return book_pivot_filtered

book_pivot = create_pivot_table(df_with_cnt)
book_sparse = csr_matrix(book_pivot.values)

@st.cache_resource
def train_model():
    model = NearestNeighbors(
        n_neighbors=30,
        algorithm='brute',
        metric='cosine'
    )
    model.fit(book_sparse)
    return model

model = train_model()

def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion:
        try:
            book_name = book_pivot.index[book_id]
            url_series = df_with_cnt[df_with_cnt['Book-Title'] == book_name]['Image-URL-L']
            if not url_series.empty:
                url = url_series.iloc[0]
                poster_url.append(url)
            else:
                poster_url.append("https://via.placeholder.com/150x200?text=No+Image")
        except:
            poster_url.append("https://via.placeholder.com/150x200?text=Error")
    return poster_url

def recommend_book(book_name):
    try:
        book_indices = np.where(book_pivot.index == book_name)[0]
        if len(book_indices) == 0:
            return [], []
        
        book_id = book_indices[0]
        
        # Dapatkan lebih banyak neighbors
        distance, suggestion = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1), 
            n_neighbors=min(25, len(book_pivot))
        )
        
        # Exclude buku itu sendiri
        recommended_indices = suggestion[0][1:]
        distances = distance[0][1:]
        
        # Tambahkan randomization dengan weighted sampling
        weights = 1 / (distances + 1e-8)
        weights = weights / weights.sum()
        noise = np.random.random(len(weights)) * 0.3
        weights = weights + noise
        weights = weights / weights.sum()
        
        # Sampling dengan replacement=False
        try:
            candidate_count = min(10, len(recommended_indices))
            selected_candidates = np.random.choice(
                recommended_indices, 
                size=candidate_count, 
                replace=False, 
                p=weights[:len(recommended_indices)]
            )
        except:
            selected_candidates = recommended_indices[:10]
        
        # Implementasi diversity
        final_recommendations = []
        used_authors = set()
        
        candidate_list = list(selected_candidates)
        random.shuffle(candidate_list)
        
        input_book_info = df_with_cnt[df_with_cnt['Book-Title'] == book_name]
        input_author = None
        if not input_book_info.empty and 'Book-Author' in input_book_info.columns:
            input_author = input_book_info['Book-Author'].iloc[0]
        
        for idx in candidate_list:
            if len(final_recommendations) >= 5:
                break
                
            book_title = book_pivot.index[idx]
            
            if book_title == book_name:
                continue
            
            book_info = df_with_cnt[df_with_cnt['Book-Title'] == book_title]
            add_book = True
            
            if not book_info.empty and 'Book-Author' in book_info.columns:
                author = book_info['Book-Author'].iloc[0]
                
                author_count = sum(1 for rec_idx in final_recommendations 
                                 if not df_with_cnt[df_with_cnt['Book-Title'] == book_pivot.index[rec_idx]].empty
                                 and 'Book-Author' in df_with_cnt.columns
                                 and df_with_cnt[df_with_cnt['Book-Title'] == book_pivot.index[rec_idx]]['Book-Author'].iloc[0] == author)
                
                if author_count >= 2:
                    add_book = False
                
                if author == input_author and author_count >= 1:
                    add_book = False
            
            if add_book:
                final_recommendations.append(idx)
        
        # Jika masih kurang dari 5
        if len(final_recommendations) < 5:
            for idx in candidate_list:
                if len(final_recommendations) >= 5:
                    break
                if idx not in final_recommendations:
                    book_title = book_pivot.index[idx]
                    if book_title != book_name:
                        final_recommendations.append(idx)
        
        random.shuffle(final_recommendations)
        
        # Get book titles and posters
        books_list = [book_pivot.index[idx] for idx in final_recommendations[:5]]
        poster_url = fetch_poster(final_recommendations[:5])
        
        return books_list, poster_url
        
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return [], []

# Simple UI
st.title("📚 Book Recommendation System")
st.markdown("Find similar books based on what you like!")

# Book selection
available_books = sorted(book_pivot.index.tolist())
selected_book = st.selectbox(
    "Type or select a book from the dropdown",
    available_books,
    index=0
)

# Show recommendation button - Simple version
if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_book)
    
    if recommended_books and poster_url:
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
    else:
        st.error("No recommendations found. Try another book!")

