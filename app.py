import streamlit as st
import numpy as np
import gensim
import os
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# -----------------------------
# Fix NLTK for Streamlit Cloud
# -----------------------------
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

stop_words = set(stopwords.words("english"))

# -----------------------------
# Load files
# -----------------------------
model = gensim.models.Word2Vec.load("word2vec.model")
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# -----------------------------
# Function: Get Query Embedding
# -----------------------------
def get_query_embedding(query):
    tokens = [
        w.lower()
        for w in word_tokenize(query)
        if w.lower() not in stop_words and w.isalpha()
    ]

    vectors = []

    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Document Search Engine")

query = st.text_input("Enter your query:")

if query:
    query_embedding = get_query_embedding(query)
    similarities = cosine_similarity(
        [query_embedding], embeddings
    )[0]

    ranked_indices = np.argsort(similarities)[::-1]

    st.subheader("Top Results:")

    for idx in ranked_indices[:5]:
        st.write(f"Score: {similarities[idx]:.4f}")
        st.write(documents[idx])
        st.write("---")
