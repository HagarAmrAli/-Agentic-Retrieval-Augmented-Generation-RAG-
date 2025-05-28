# app.py - Fully self-contained Streamlit app for Agentic RAG Travel Chatbot using OpenAI v1 embeddings

import os
import streamlit as st
import pandas as pd
import re
import json
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI

# === Set API Key ===
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # Or use st.secrets for secure storage
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# === Embedding Function ===
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# === Qdrant Setup and Indexing ===
@st.cache_resource
def load_qdrant_and_index():
    df = pd.read_csv("tripadvisor_hotel_reviews.csv")
    df.dropna(subset=["Review"], inplace=True)

    def clean_text(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        return text.strip()

    df['clean_review'] = df['Review'].apply(clean_text)

    chunks = []
    chunk_id = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentences = re.split(r'(?<=[.!?]) +', row['clean_review'])
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) < 500:
                current_chunk += " " + sentence
            else:
                chunks.append({"id": chunk_id, "text": current_chunk.strip()})
                chunk_id += 1
                current_chunk = sentence
        if current_chunk:
            chunks.append({"id": chunk_id, "text": current_chunk.strip()})
            chunk_id += 1

    # Connect to Qdrant (use ":memory:" for in-memory or localhost if running Qdrant locally)
    qdrant = QdrantClient("localhost", port=6333)

    # Create or recreate the collection
    qdrant.recreate_collection(
        collection_name="travel_chunks",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    # Embed and upload
    vectors = [get_embedding(chunk["text"]) for chunk in chunks]
    payloads = [{"text": chunk["text"]} for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    points = [
        PointStruct(id=id, vector=vec, payload=payload)
        for id, vec, payload in zip(ids, vectors, payloads)
    ]

    qdrant.upsert(collection_name="travel_chunks", points=points)
    return qdrant

# === Chunk Retrieval ===
def retrieve_chunks(query, qdrant, top_k=5):
    vector = get_embedding(query)
    hits = qdrant.search(
        collection_name="travel_chunks",
        query_vector=vector,
        limit=top_k
    )
    return [hit.payload["text"] for hit in hits]

# === GPT-4 Response Generation ===
def generate_response(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"""You are a helpful travel assistant. Based on the information below, answer the user's query in a friendly and informative tone.

### Query:
{query}

### Context:
{context}

### Answer:"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# === Streamlit UI ===
st.set_page_config(page_title="🌍 Travel Guide Chatbot", layout="centered")
st.title("🌍 Travel Guide Chatbot")

query = st.text_input("Ask a travel-related question:")

if query:
    with st.spinner("Processing your request..."):
        qdrant = load_qdrant_and_index()
        retrieved = retrieve_chunks(query, qdrant)
        response = generate_response(query, retrieved)
    st.markdown("---")
    st.subheader("✈️ Response:")
    st.success(response)

    with st.expander("🔍 See Retrieved Chunks"):
        for i, chunk in enumerate(retrieved, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")
