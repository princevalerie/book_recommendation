# -*- coding: utf-8 -*-
"""Colaborative_Filtering.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zIMZikCK828A9jqiHWv7NCq_mJE9blbG
"""

# prompt: ekstrak/content/drive/MyDrive/intern_file/BX-Book-Ratings.csv

import pandas as pd

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Specify the file path
file_path = '/content/drive/MyDrive/intern_file/BX-Book-Ratings.csv'

try:
  # Read the CSV file into a pandas DataFrame
  df = pd.read_csv(file_path, encoding='latin-1', sep=';')  # Specify encoding and separator if needed

  # Now you can work with the DataFrame 'df'
  df.head() # Display the first few rows

except FileNotFoundError:
  print(f"Error: File not found at {file_path}")
except Exception as e:
  print(f"An error occurred: {e}")

df_user_rating = df.copy()
df_user_rating

df_user_rating.info()

df_user_rating.duplicated().sum()

file_path = '/content/drive/MyDrive/intern_file/BX-Books.csv'

df_book = pd.read_csv(file_path, encoding='latin-1', sep=';', on_bad_lines='warn')

df_book

df_book.info()

df_book.duplicated().sum()

# prompt: menampilkan tabel berisi null

# Display DataFrame with null values
df_book[df_book.isnull().any(axis=1)]

# prompt: drop baris yang berisi null

# Drop rows with any null values
df_book = df_book.dropna()

# Display the DataFrame after dropping null values
df_book

# Drop columns 'Image-URL-S' and 'Image-URL-M'
df_book = df_book.drop(columns=['Image-URL-S', 'Image-URL-M'], errors='ignore')

df_book

df_book.info()

# prompt: merge book ke user rating

# Merge the two DataFrames based on the 'ISBN' column
merged_df = pd.merge(df_user_rating, df_book, on='ISBN', how='inner')

# Display the merged DataFrame
merged_df

merged_df['Book-Rating'].value_counts()

df = merged_df.copy()
df

df = df[df['Book-Rating']>0]
df

num_rating=df.groupby('Book-Title')['Book-Rating'].count().reset_index()
num_rating

num_rating.rename(columns={'Book-Rating':'Cnt_Rating'},inplace=True)

df_with_cnt=df.merge(num_rating,on='Book-Title')
df_with_cnt

df_with_cnt=df_with_cnt[df_with_cnt['Cnt_Rating']>60]

df_with_cnt.drop_duplicates(['User-ID','Book-Rating'],inplace=True)

df_with_cnt

df_with_cnt.to_csv('df_with_cnt.csv',index=False)

book_pivot=df_with_cnt.pivot_table(columns='User-ID',index='Book-Title',values='Book-Rating', aggfunc='mean')
book_pivot.fillna(0,inplace=True)
book_pivot

book_pivot.to_csv('book_pivot.csv',index=False)

from scipy.sparse import csr_matrix
book_sparse=csr_matrix(book_pivot)
book_sparse

# prompt: distance,suggestion=model.kneighbors(book_pivot.iloc[237,:].values.reshape(1,-1),n_neighbors=6)

from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

distance, suggestion = model.kneighbors(book_pivot.iloc[237, :].values.reshape(1, -1), n_neighbors=6)

distance

suggestion

for i in range(len(suggestion[0])):
    books = book_pivot.index[suggestion[0][i]]
    print(books)

books_name=book_pivot.index

import numpy as np

def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    print(f"Rekomendasi untuk buku '{book_name}':")
    for i in range(1, len(suggestion[0])):  # Mulai dari 1 untuk menghindari buku yang sama
        recommended_book = book_pivot.index[suggestion[0][i]]
        print(recommended_book)

book_name="A Bend in the Road"
recommend_book(book_name)

book_name="The Drawing of the Three (The Dark Tower, Book 2)"
recommend_book(book_name)