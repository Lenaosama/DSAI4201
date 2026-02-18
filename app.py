import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Load trained Word2Vec model
model = Word2Vec.load("word2vec.model")

stop_words = set(stopwords.words("english"))

def retrieve_top_k(query_embedding, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# Streamlit UI
st.title("Information Retrieval using Document Embeddings")

query = st.text_input("Enter your query:")

# REAL query embedding function
def get_query_embedding(query):
    tokens = [
        w.lower()
        for w in word_tokenize(query)
        if w.isalnum() and w.lower() not in stop_words
    ]

    vectors = [model.wv[w] for w in tokens if w in model.wv]

    if len(vectors) == 0:
        return None

    return np.mean(vectors, axis=0)

if st.button("Search"):

    query_embedding = get_query_embedding(query)

    if query_embedding is None:
        st.warning("Query words not found in vocabulary.")
    else:
        results = retrieve_top_k(query_embedding, embeddings)

        st.write("### Top 10 Relevant Documents:")

        for doc, score in results:
            st.write(f"- **{doc.strip()}** (Score: {score:.4f})")
