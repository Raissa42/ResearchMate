"""
ResearchMate - Agentic RAG System for Multi-Document Research Analysis
=========================================================================
PHASE 14 - Everything Connected

This file is now the FULL application - the UI you built in Phase 5 is
wired to the real pipeline: PDF loading (6), chunking (7), embeddings
(8), ChromaDB (9), retrieval (10), Gemini (11), the RAG chain (12), and
the Agent's dynamic routing (13).

Run this file with:
    streamlit run app.py
"""

import streamlit as st

# Importing config.settings FIRST guarantees .env is loaded and
# GOOGLE_API_KEY is validated before anything else in the app runs.
from config import settings

from utils.pdf_loader import load_multiple_pdfs
from utils.text_splitter import split_multiple_documents
from vectorstore.chroma_store import (
    add_chunks_to_vectorstore,
    clear_vectorstore,
    get_vectorstore_chunk_count,
)
from agents.router import route_question


# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="ResearchMate",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# -------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "uploader_key" not in st.session_state:
    # Streamlit's file_uploader widget doesn't clear itself just because
    # we reset a variable - changing its `key` forces Streamlit to treat
    # it as a BRAND NEW widget, which is what actually clears the
    # uploaded files shown in the UI. We increment this counter below
    # whenever the user wants to start fresh.
    st.session_state.uploader_key = 0

if "top_k" not in st.session_state:
    st.session_state.top_k = 4  # default number of chunks retrieved per question


# -------------------------------------------------------------------
# 3. REAL PROCESSING LOGIC
# Kept as a plain function (not inline in the sidebar block) so it can
# be unit-tested on its own, separately from Streamlit's UI code.
# -------------------------------------------------------------------
def process_uploaded_files(uploaded_files, progress_callback=None) -> dict:
    """
    Runs the FULL ingestion pipeline on newly uploaded PDFs:
    load text -> split into chunks -> clear old data -> embed & store.

    Parameters
    ----------
    progress_callback : callable, optional
        If given, called with (fraction: float, label: str) after each
        major step, so the UI can show a progress bar (Phase 15).

    Returns
    -------
    dict
        {"num_files": int, "num_chunks": int, "filenames": [...]}
    """
    def report(fraction, label):
        if progress_callback:
            progress_callback(fraction, label)

    report(0.1, "Reading PDF text...")
    documents = load_multiple_pdfs(uploaded_files)

    if not documents:
        raise ValueError(
            "No readable text could be extracted from the uploaded file(s). "
            "They may be scanned images or corrupted PDFs."
        )

    report(0.4, "Splitting into chunks...")
    chunks = split_multiple_documents(documents)

    report(0.6, "Clearing previous session data...")
    # Per Raissa's choice: start fresh each time, rather than
    # accumulating every document ever uploaded across sessions.
    clear_vectorstore()

    report(0.75, "Generating embeddings and storing in ChromaDB...")
    add_chunks_to_vectorstore(chunks)

    report(1.0, "Done!")

    return {
        "num_files": len(documents),
        "num_chunks": len(chunks),
        "filenames": list(documents.keys()),
    }


def reset_session() -> None:
    """
    Wipes everything for a clean slate: clears the vector database,
    clears the chat history, clears the "loaded papers" list, and
    forces the file_uploader widget to reset by changing its key.
    """
    clear_vectorstore()
    st.session_state.chat_history = []
    st.session_state.uploaded_docs = []
    st.session_state.documents_processed = False
    st.session_state.uploader_key += 1  # forces a fresh file_uploader widget


# -------------------------------------------------------------------
# 4. SIDEBAR - Document Upload Area
# -------------------------------------------------------------------
with st.sidebar:
    st.title("📚 ResearchMate")
    st.caption("Agentic RAG for Multi-Document Research Analysis")

    st.divider()

    st.subheader("Upload Research Papers")
    uploaded_files = st.file_uploader(
        label="Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload the research papers you want to analyze.",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    process_button = st.button("Process Documents", use_container_width=True)

    if process_button:
        if uploaded_files:
            progress_bar = st.progress(0, text="Starting...")

            def update_progress(fraction, label):
                progress_bar.progress(fraction, text=label)

            try:
                result = process_uploaded_files(uploaded_files, progress_callback=update_progress)
                st.session_state.uploaded_docs = result["filenames"]
                st.session_state.documents_processed = True
                progress_bar.empty()
                st.success(
                    f"Processed {result['num_files']} document(s) "
                    f"into {result['num_chunks']} chunks."
                )
            except Exception as error:
                progress_bar.empty()
                st.error(f"Failed to process documents: {error}")
        else:
            st.warning("Please upload at least one PDF first.")

    st.divider()

    if st.session_state.uploaded_docs:
        st.subheader("Loaded Papers")
        for doc_name in st.session_state.uploaded_docs:
            st.write(f"- {doc_name}")
        st.caption(f"{get_vectorstore_chunk_count()} chunks indexed")

    st.divider()

    # Settings - lets the user control retrieval depth without touching
    # code. Higher top_k = more context per answer, but slower/costlier.
    with st.expander("⚙️ Settings"):
        st.session_state.top_k = st.slider(
            "Chunks retrieved per question",
            min_value=2, max_value=10,
            value=st.session_state.get("top_k", 4),
            help="How many document chunks are retrieved to answer a normal question. "
                 "Higher = more context, but slower and more expensive.",
        )

    st.divider()

    # Reset button - clears vectorstore, chat, uploaded file list, and
    # forces the file_uploader widget to visually clear too.
    if st.button("🔄 Start New Session", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.caption("Built with LangChain + Google Gemini")


# -------------------------------------------------------------------
# 5. MAIN AREA - Chat Interface
# -------------------------------------------------------------------
st.header("Ask ResearchMate")

if not st.session_state.documents_processed:
    st.info("Upload and process at least one PDF from the sidebar to get started.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"Sources: {', '.join(message['sources'])}")

user_question = st.chat_input("Ask a question about your uploaded papers...")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if not st.session_state.documents_processed:
            answer = "Please upload and process at least one PDF before asking questions."
            sources = []
            st.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                try:
                    result = route_question(user_question, top_k=st.session_state.get("top_k", 4))
                    answer = result["answer"]
                    sources = result["sources"]
                    st.markdown(answer)
                    if sources:
                        st.caption(f"Sources: {', '.join(sources)}")
                    st.caption(f"Pipeline used: {result['pipeline_used']}")
                except Exception as error:
                    answer = f"Something went wrong while answering: {error}"
                    sources = []
                    st.markdown(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
