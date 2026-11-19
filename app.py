import io
import re
import hashlib

import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuChat AI")
st.write("Your personal AI assistant for asking questions about PDF documents.")

# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


# ==========================================
# 3. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdfs(uploaded_files):

    documents = []

    for uploaded_file in uploaded_files:

        pdf = PdfReader(
            io.BytesIO(uploaded_file.getvalue())
        )

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text() or ""

            text = re.sub(r"\s+", " ", text).strip()

            if text:

                documents.append({
                    "text": text,
                    "source": uploaded_file.name,
                    "page": page_number
                })

    return documents


# ==========================================
# 4. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_text_into_chunks(
    documents,
    chunk_size=900,
    overlap=150
):

    chunks = []

    for document in documents:

        text = document["text"]

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"]
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks


# ==========================================
# 5. CREATE EMBEDDINGS
# ==========================================

def create_embeddings(chunks, embedding_model):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype("float32")


# ==========================================
# 6. STORE EMBEDDINGS IN FAISS
# ==========================================

def create_vector_database(chunks, embedding_model):

    embeddings = create_embeddings(
        chunks,
        embedding_model
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ==========================================
# 7. RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def retrieve_documents(
    question,
    index,
    chunks,
    embedding_model,
    top_k=4
):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        result = chunks[int(idx)].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ==========================================
# 8. GENERATE ANSWER USING GROQ
# ==========================================

def generate_answer(
    question,
    retrieved_documents,
    api_key,
    model_name
):

    client = Groq(api_key=api_key)

    context = ""

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context += f"""
        [Source {i}]
        Document: {document['source']}
        Page: {document['page']}

        Content:
        {document['text']}

        ---------------------------
        """

    system_prompt = """
    You are DocuChat AI, a document question-answering assistant.

    Your task is to answer questions using ONLY the
    information provided in the retrieved document context.

    Rules:

    1. Do not invent information.
    2. Do not use outside knowledge.
    3. If the answer is not available in the documents,
       clearly say that you could not find the answer.
    4. Explain answers in simple language.
    5. Cite sources using [Source 1], [Source 2], etc.
    """

    user_prompt = f"""
    DOCUMENT CONTEXT:

    {context}

    USER QUESTION:

    {question}

    Provide a clear and accurate answer with source citations.
    """

    response = client.chat.completions.create(

        model=model_name,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1,

        max_tokens=1500
    )

    return response.choices[0].message.content


# ==========================================
# 9. SESSION STATE
# ==========================================

if "vector_index" not in st.session_state:

    st.session_state.vector_index = None

if "chunks" not in st.session_state:

    st.session_state.chunks = []

if "document_signature" not in st.session_state:

    st.session_state.document_signature = None

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================
# 10. SIDEBAR SETTINGS
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    model_name = st.selectbox(
        "Choose LLM",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=8,
        value=4
    )

    st.divider()

    st.markdown("### How it works")

    st.caption("PDF → Text → Chunks → Embeddings → FAISS → Groq AI")


# ==========================================
# 11. PDF UPLOAD
# ==========================================

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    signature_content = "".join(
        hashlib.md5(
            file.getvalue()
        ).hexdigest()
        for file in uploaded_files
    )

    signature = hashlib.sha256(
        signature_content.encode()
    ).hexdigest()

    if signature != st.session_state.document_signature:

        with st.spinner("Processing your documents..."):

            documents = extract_text_from_pdfs(
                uploaded_files
            )

            chunks = split_text_into_chunks(
                documents
            )

            if not chunks:

                st.error(
                    "No readable text found in the PDFs. "
                    "Scanned PDFs require OCR."
                )

                st.session_state.vector_index = None
                st.session_state.chunks = []

            else:

                embedding_model = load_embedding_model()

                vector_index = create_vector_database(
                    chunks,
                    embedding_model
                )

                st.session_state.vector_index = vector_index

                st.session_state.chunks = chunks

                st.session_state.document_signature = signature

                st.session_state.messages = []

        if st.session_state.vector_index is not None:

            st.success(
                f"Successfully processed {len(uploaded_files)} PDF(s). "
                f"Created {len(st.session_state.chunks)} chunks."
            )


# ==========================================
# 12. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📄 View Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"**{source['source']} — Page {source['page']}**"
                    )

                    st.write(source["text"])

                    st.divider()


# ==========================================
# 13. CHATBOT INTERACTION
# ==========================================

question = st.chat_input(
    "Ask something about your PDF documents..."
)

if question:

    if st.session_state.vector_index is None:

        st.warning("Please upload a PDF first.")

    elif not api_key:

        st.warning("Please enter your Groq API key in the sidebar.")

    else:

        # Display user question

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        # Generate assistant answer

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching documents and generating answer..."
                ):

                    embedding_model = load_embedding_model()

                    # Retrieve relevant chunks

                    retrieved_documents = retrieve_documents(
                        question,
                        st.session_state.vector_index,
                        st.session_state.chunks,
                        embedding_model,
                        top_k
                    )

                    # Generate LLM answer

                    answer = generate_answer(
                        question,
                        retrieved_documents,
                        api_key,
                        model_name
                    )

                st.markdown(answer)

                # Display source documents

                with st.expander("📄 View Retrieved Sources"):

                    for source in retrieved_documents:

                        st.markdown(
                            f"**{source['source']} — Page {source['page']}**"
                        )

                        st.write(source["text"])

                        st.divider()

                # Save conversation

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": retrieved_documents

                })

            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )


# ==========================================
# 14. CLEAR CHAT
# ==========================================

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()