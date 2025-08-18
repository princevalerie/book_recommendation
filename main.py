import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():
    df_with_cnt = pd.read_csv('df_with_cnt.csv')
    return df_with_cnt

df_with_cnt = load_data()

# Create the pivot table for the recommendation system
@st.cache_data
def create_pivot_table(df):
    # FIX 1: Gunakan aggregation function yang tepat untuk menghindari duplikasi
    book_pivot = df.pivot_table(
        columns='User-ID', 
        index='Book-Title', 
        values='Book-Rating', 
        aggfunc='mean'  # Menggunakan rata-rata untuk menghindari duplikasi
    )
    book_pivot.fillna(0, inplace=True)
    
    # FIX 2: Filter buku dengan minimal interaksi untuk mengurangi noise
    # Hanya ambil buku yang memiliki minimal 5 rating
    book_counts = (book_pivot > 0).sum(axis=1)
    popular_books = book_counts[book_counts >= 5].index
    book_pivot_filtered = book_pivot.loc[popular_books]
    
    return book_pivot_filtered

book_pivot = create_pivot_table(df_with_cnt)

# Convert the pivot table to a sparse matrix
book_sparse = csr_matrix(book_pivot.values)

# FIX 3: Gunakan algoritma yang lebih cocok dan parameter yang tepat
@st.cache_resource
def train_model():
    model = NearestNeighbors(
        n_neighbors=20,  # Mencari lebih banyak neighbors
        algorithm='brute',  # Lebih akurat untuk dataset kecil-menengah
        metric='cosine'  # Cosine similarity lebih cocok untuk collaborative filtering
    )
    model.fit(book_sparse)
    return model

model = train_model()

# Function to fetch poster URLs for recommendations
def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion:
        try:
            # Retrieve the URL for the recommended book
            book_name = book_pivot.index[book_id]
            url_series = df_with_cnt[df_with_cnt['Book-Title'] == book_name]['Image-URL-L']
            if not url_series.empty:
                url = url_series.iloc[0]
                poster_url.append(url)
            else:
                # Default image jika tidak ada
                poster_url.append("https://via.placeholder.com/150x200?text=No+Image")
        except Exception as e:
            poster_url.append("https://via.placeholder.com/150x200?text=Error")
    return poster_url

# FIX 4: Improve recommendation function dengan diversity dan filtering
def recommend_book(book_name, diversity_factor=0.3):
    try:
        # Find the index of the given book
        book_indices = np.where(book_pivot.index == book_name)[0]
        if len(book_indices) == 0:
            return [], []
        
        book_id = book_indices[0]
        
        # Get more neighbors to increase diversity
        distance, suggestion = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1), 
            n_neighbors=min(15, len(book_pivot))  # Ambil lebih banyak kandidat
        )
        
        # FIX 5: Tambahkan diversity dalam rekomendasi
        recommended_indices = suggestion[0][1:]  # Exclude the book itself
        distances = distance[0][1:]
        
        # Sort berdasarkan distance (similarity)
        sorted_pairs = sorted(zip(recommended_indices, distances), key=lambda x: x[1])
        
        # Implementasi diversity: pilih buku dengan kombinasi similarity dan keunikan
        final_recommendations = []
        used_authors = set()
        
        # Coba dapatkan informasi author jika ada
        input_book_info = df_with_cnt[df_with_cnt['Book-Title'] == book_name]
        input_author = input_book_info['Book-Author'].iloc[0] if not input_book_info.empty and 'Book-Author' in input_book_info.columns else None
        
        for idx, dist in sorted_pairs:
            if len(final_recommendations) >= 5:
                break
                
            book_title = book_pivot.index[idx]
            book_info = df_with_cnt[df_with_cnt['Book-Title'] == book_title]
            
            # Skip jika sama dengan buku input
            if book_title == book_name:
                continue
                
            # Tambahkan diversity berdasarkan author (jika data tersedia)
            if not book_info.empty and 'Book-Author' in book_info.columns:
                author = book_info['Book-Author'].iloc[0]
                
                # Jika sudah ada 2 buku dari author yang sama, skip
                if len(final_recommendations) >= 2 and author in used_authors:
                    continue
                    
                # Jika author sama dengan input book dan sudah ada 1 rekomendasi, pertimbangkan diversity
                if author == input_author and len([r for r in final_recommendations]) >= 1:
                    if np.random.random() > diversity_factor:
                        continue
                
                used_authors.add(author)
            
            final_recommendations.append(idx)
        
        # Jika masih kurang dari 5, tambahkan sisanya
        for idx, dist in sorted_pairs:
            if len(final_recommendations) >= 5:
                break
            if idx not in final_recommendations:
                final_recommendations.append(idx)
        
        # Get the list of recommended book titles
        books_list = [book_pivot.index[idx] for idx in final_recommendations[:5]]
        
        # Fetch poster URLs for the recommended books
        poster_url = fetch_poster(final_recommendations[:5])
        
        return books_list, poster_url
        
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return [], []

# Streamlit app setup
st.title("📚 Book Recommendation System")
st.write("Find similar books based on what you like!")

# FIX 6: Filter book titles yang ada di pivot table
available_books = book_pivot.index.tolist()
book_titles = sorted(available_books)

# User input for book selection
selected_books = st.selectbox(
    "Type or select a book from the dropdown", 
    book_titles,
    help="Select a book to get personalized recommendations"
)

# FIX 7: Tambahkan informasi tentang buku yang dipilih
if selected_books:
    book_info = df_with_cnt[df_with_cnt['Book-Title'] == selected_books].iloc[0]
    st.write(f"**Selected Book:** {selected_books}")
    if 'Book-Author' in book_info:
        st.write(f"**Author:** {book_info['Book-Author']}")

# FIX 8: Tambahkan loading state dan error handling
if st.button('Show Recommendation'):
    with st.spinner('Finding similar books...'):
        recommended_books, poster_url = recommend_book(selected_books)
        
        if recommended_books:
            st.success(f"Found {len(recommended_books)} recommendations!")
            
            # Display recommendations
            cols = st.columns(5)
            for i, (book, url) in enumerate(zip(recommended_books, poster_url)):
                with cols[i]:
                    st.text(book[:50] + "..." if len(book) > 50 else book)  # Truncate long titles
                    st.image(url, use_column_width=True)
                    
                    # Tambahkan info tambahan jika ada
                    book_details = df_with_cnt[df_with_cnt['Book-Title'] == book]
                    if not book_details.empty and 'Book-Author' in book_details.columns:
                        st.caption(f"by {book_details['Book-Author'].iloc[0]}")
        else:
            st.error("Sorry, couldn't find recommendations for this book. Please try another one.")

# FIX 9: Tambahkan statistik dataset
with st.expander("Dataset Information"):
    st.write(f"Total books in recommendation system: {len(book_pivot)}")
    st.write(f"Total users: {len(book_pivot.columns)}")
    st.write(f"Total ratings: {(book_pivot > 0).sum().sum()}")
    
    # Show distribution of ratings per book
    ratings_per_book = (book_pivot > 0).sum(axis=1)
    st.write(f"Average ratings per book: {ratings_per_book.mean():.1f}")
    st.write(f"Books with most ratings: {ratings_per_book.max()}")
