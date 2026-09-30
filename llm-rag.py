import os
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

QUESTION_PLACEHOLDER = "e.g. How many days of notice period do I need to serve after probation?"


# --- Cached resources (loaded once per session, not on every rerun) --------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# 1. EXTRACT ------------------------------------------------------------
def extract_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# 2. CHUNK ----------------------------------------------------------------
def create_chunks(text, chunk_size=700, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# 3. EMBED + INDEX -----------------------------------------------------------
def build_index(chunks: list[str], embedding_model):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


# 4. RETRIEVE ---------------------------------------------------------------
def retrieve(question: str, chunks: list[str], index, embedding_model, top_k: int = 2):
    question_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(question_vector, top_k)
    return [(chunks[i], scores[0][rank]) for rank, i in enumerate(indices[0])]


# 5. GENERATE -----------------------------------------------------------
def generate_answer(question: str, context: str, api_key: str) -> str:
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    if not api_key:
        return (
            "No Gemini API key provided — showing retrieved context only.\n\n"
            f"Relevant context found:\n{context}"
        )

    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text


# --- Streamlit UI ------------------------------------------------------
st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄")
st.title("📄 Ask questions about your PDF")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Gemini API key", type="password", help="Get one at aistudio.google.com/apikey")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF from the sidebar to get started.")
    st.stop()

embedding_model = load_embedding_model()

# Rebuild the index only when a new file is uploaded (cached per file by name+size)
@st.cache_data(show_spinner="Reading and indexing PDF...")
def process_pdf(file_bytes, file_name):
    import io
    raw_text = extract_text(io.BytesIO(file_bytes))
    chunks = create_chunks(raw_text)
    return chunks

file_bytes = uploaded_file.getvalue()
chunks = process_pdf(file_bytes, uploaded_file.name)
index = build_index(chunks, embedding_model)

st.success(f"Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

question = st.text_input("Your question", placeholder=QUESTION_PLACEHOLDER)

if st.button("Get answer", type="primary") and question:
    with st.spinner("Retrieving relevant context and generating answer..."):
        top_chunks = retrieve(question, chunks, index, embedding_model, top_k=2)
        context = "\n---\n".join(c for c, _ in top_chunks)
        answer = generate_answer(question, context, api_key)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Show retrieved context (for debugging)"):
        for chunk, score in top_chunks:
            st.markdown(f"**score={score:.3f}**")
            st.text(chunk)