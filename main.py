import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load dataset
@st.cache_data
def load_data():
    df_with_cnt = pd.read_csv('df_with_cnt.csv')
    return df_with_cnt

df_with_cnt = load_data()

# Membuat pivot table untuk sistem rekomendasi
@st.cache_data
def create_pivot_table(df):
    book_pivot = df.pivot_table(
        columns='User-ID',
        index='Book-Title',
        values='Book-Rating',
        aggfunc='mean'
    )
    book_pivot.fillna(0, inplace=True)

    # Filter buku dengan minimal interaksi (>= 3 rating)
    book_counts = (book_pivot > 0).sum(axis=1)
    popular_books = book_counts[book_counts >= 3].index
    book_pivot_filtered = book_pivot.loc[popular_books]

    return book_pivot_filtered

book_pivot = create_pivot_table(df_with_cnt)
book_sparse = csr_matrix(book_pivot.values)

# Melatih model Nearest Neighbors
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

# Fungsi ambil poster buku
def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion:
        try:
            book_name = book_pivot.index[book_id]
            url_series = df_with_cnt[df_with_cnt['Book-Title'] == book_name]['Image-URL-L']
            if not url_series.empty:
                poster_url.append(url_series.iloc[0])
            else:
                poster_url.append("https://via.placeholder.com/150x200?text=No+Image")
        except:
            poster_url.append("https://via.placeholder.com/150x200?text=Error")
    return poster_url

# Fungsi rekomendasi buku TANPA noise
def recommend_book(book_name):
    try:
        book_indices = np.where(book_pivot.index == book_name)[0]
        if len(book_indices) == 0:
            return [], []

        book_id = book_indices[0]

        # Cari neighbors terdekat berdasarkan cosine similarity
        distances, suggestions = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1),
            n_neighbors=min(25, len(book_pivot))
        )

        # Urutkan berdasarkan jarak terkecil → paling mirip
        neighbors_sorted = sorted(
            zip(distances[0][1:], suggestions[0][1:]),
            key=lambda x: x[0]
        )
        recommended_indices = [idx for _, idx in neighbors_sorted]

        # Implementasi diversity → hindari penulis berulang lebih dari 2x
        final_recommendations = []
        used_authors = {}

        input_book_info = df_with_cnt[df_with_cnt['Book-Title'] == book_name]
        input_author = input_book_info['Book-Author'].iloc[0] if not input_book_info.empty else None

        for idx in recommended_indices:
            if len(final_recommendations) >= 5:
                break

            book_title = book_pivot.index[idx]
            book_info = df_with_cnt[df_with_cnt['Book-Title'] == book_title]
            author = book_info['Book-Author'].iloc[0] if not book_info.empty else None

            # Batasi maksimal 2 buku per penulis
            if author:
                used_authors[author] = used_authors.get(author, 0) + 1
                if used_authors[author] > 2:
                    continue

                # Jangan rekomendasikan buku dari penulis input lebih dari 1x
                if author == input_author and used_authors[author] > 1:
                    continue

            final_recommendations.append(idx)

        # Ambil hanya 5 rekomendasi
        final_recommendations = final_recommendations[:5]
        books_list = [book_pivot.index[idx] for idx in final_recommendations]
        poster_url = fetch_poster(final_recommendations)

        return books_list, poster_url

    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return [], []

# ==========================
# Streamlit UI
# ==========================
st.title("📚 Book Recommendation System")

# Dropdown untuk memilih buku
available_books = sorted(book_pivot.index.tolist())
selected_book = st.selectbox(
    "Type or select a book from the dropdown",
    available_books,
    index=0
)

# Tampilkan rekomendasi
if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_book)

    if recommended_books and poster_url:
        cols = st.columns(5)
        for col, url, book in zip(cols, poster_url, recommended_books):
            with col:
                st.image(url)
                st.caption(book)
    else:
        st.error("No recommendations found. Try another book!")

