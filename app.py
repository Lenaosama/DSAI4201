import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

def retrieve_top_k(query_embedding, embeddings, k=10):
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]   # <- IMPORTANT: take first row

    top_k_indices = similarities.argsort()[-k:][::-1]

    return [(documents[i], similarities[i]) for i in top_k_indices]


st.title("Information Retrieval using Document Embeddings")

query = st.text_input("Enter your query:")

def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1])

if st.button("Search"):
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings, k=10)

    st.write("### Top 10 Relevant Documents:")
    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")

