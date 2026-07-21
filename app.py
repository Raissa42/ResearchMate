# Replace the entire content of app.py with this improved version
# Key changes:
# - Added custom dark theme CSS to match the screenshot
# - Rebuilt main area with greeting + action cards
# - Enhanced sidebar to look closer to the image
# - Kept all backend logic intact
# - Added sample suggestion buttons like in the image

import streamlit as st
import os

# Import config and utilities (unchanged)
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
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="ResearchMate",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# CUSTOM CSS - Matches the dark modern UI from the screenshot
# -------------------------------------------------------------------
st.markdown("""
<style>
    /* Main dark theme */
    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    .css-1d391kg, section[data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2a3444;
    }
    
    /* Card styling for action buttons */
    .card {
        background-color: #1e2537;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a3444;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        height: 48px;
        font-weight: 500;
    }
    
    /* Chat/Answer box */
    .answer-box {
        background-color: #1e2537;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #2a3444;
    }
    
    /* Uploaded file items */
    .uploaded-file {
        background-color: #252b3d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "top_k" not in st.session_state:
    st.session_state.top_k = 4

# -------------------------------------------------------------------
# PROCESSING FUNCTIONS (unchanged)
# -------------------------------------------------------------------
def process_uploaded_files(uploaded_files, progress_callback=None) -> dict:
    def report(fraction, label):
        if progress_callback:
            progress_callback(fraction, label)

    report(0.1, "Reading PDF text...")
    documents = load_multiple_pdfs(uploaded_files)

    if not documents:
        raise ValueError("No readable text could be extracted...")

    report(0.4, "Splitting into chunks...")
    chunks = split_multiple_documents(documents)

    report(0.6, "Clearing previous session data...")
    clear_vectorstore()

    report(0.75, "Generating embeddings...")
    add_chunks_to_vectorstore(chunks)

    report(1.0, "Done!")
    return {
        "num_files": len(documents),
        "num_chunks": len(chunks),
        "filenames": list(documents.keys()),
    }

def reset_session():
    clear_vectorstore()
    st.session_state.chat_history = []
    st.session_state.uploaded_docs = []
    st.session_state.documents_processed = False
    st.session_state.uploader_key += 1

# -------------------------------------------------------------------
# SIDEBAR - Improved to match screenshot
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 28px;">🔬</span>
        <div>
            <h2 style="margin: 0; color: #a78bfa;">ResearchMate</h2>
            <p style="margin: 0; color: #94a3b8; font-size: 14px;">Agentic RAG for Multi-Document</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("📤 UPLOAD PAPERS")
    st.markdown("""
    <div style="background: #1e2537; border: 2px dashed #4b5563; border-radius: 12px; padding: 32px 16px; text-align: center; margin-bottom: 16px;">
        <p style="font-size: 32px; margin: 0;">☁️</p>
        <p style="margin: 8px 0 4px;">Drag & drop PDFs here</p>
        <p style="color: #64748b; font-size: 14px;">or</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        label="",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("📄 Browse Files", type="primary", use_container_width=True):
        # Streamlit handles this via the uploader
        pass

    process_button = st.button("🚀 Process Documents", type="primary", use_container_width=True)

    if process_button and uploaded_files:
        progress_bar = st.progress(0)
        try:
            def update_progress(f, l):
                progress_bar.progress(f, text=l)
            result = process_uploaded_files(uploaded_files, update_progress)
            st.session_state.uploaded_docs = result["filenames"]
            st.session_state.documents_processed = True
            st.success(f"✅ Processed {result['num_files']} documents")
        except Exception as e:
            st.error(str(e))

    st.divider()

    if st.session_state.uploaded_docs:
        st.subheader("📚 LOADED PAPERS")
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"""
            <div class="uploaded-file">
                📄 {doc}<br>
                <small style="color:#22c55e;">✓ Processed</small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    if st.button("🔄 Start New Session", use_container_width=True):
        reset_session()
        st.rerun()

# -------------------------------------------------------------------
# MAIN AREA - Matches the screenshot layout
# -------------------------------------------------------------------
col_logo, col_title, col_user = st.columns([1, 4, 1])
with col_logo:
    st.markdown("# 🔬")
with col_title:
    st.title("Hello, Researcher! 👋")
with col_user:
    st.markdown("""
    <div style="text-align: right; padding-top: 12px;">
        <span style="background: #1e2937; padding: 4px 12px; border-radius: 9999px; font-size: 13px;">Share</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("Upload your research papers and ask anything. I'll analyze, compare, extract, and summarize — with sources.")

# Action Cards
st.markdown("### Quick Actions")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="card">
        <h3 style="margin:0 0 8px;">📝 Summarize</h3>
        <p style="color:#94a3b8; font-size:14px;">Get a concise summary of your documents.</p>
        <div style="margin-top: 20px; color: #a5b4fc;">→</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
        <h3 style="margin:0 0 8px;">📋 Extract</h3>
        <p style="color:#94a3b8; font-size:14px;">Extract key information, entities, or data.</p>
        <div style="margin-top: 20px; color: #4ade80;">→</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <h3 style="margin:0 0 8px;">💬 Ask a Question</h3>
        <p style="color:#94a3b8; font-size:14px;">Get answers to specific questions from your papers.</p>
        <div style="margin-top: 20px; color: #60a5fa;">→</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card">
        <h3 style="margin:0 0 8px;">⚖️ Compare</h3>
        <p style="color:#94a3b8; font-size:14px;">Compare topics, findings, or methodologies.</p>
        <div style="margin-top: 20px; color: #fbbf24;">→</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Answer Section
st.subheader("ResearchMate Answer")
question = st.text_input("What are the latest processors for smart phones, tablets and desktops?", 
                        placeholder="Ask a question about your uploaded papers...")

if st.button("Ask", type="primary"):
    if st.session_state.documents_processed and question:
        with st.spinner("Analyzing documents..."):
            try:
                result = route_question(question, top_k=st.session_state.top_k)
                answer = result["answer"]
                sources = result.get("sources", [])
                
                st.markdown(f"""
                <div class="answer-box">
                    {answer}
                    <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid #334155;">
                        <strong>Sources:</strong> {', '.join(sources) if sources else 'None'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please process documents first.")

# Suggestion chips (like in the image)
st.markdown("**Try asking:**")
suggestions = ["Summarize the first document", "Extract key features", "Compare both papers", "What are the main findings?"]
cols = st.columns(len(suggestions))
for i, sug in enumerate(suggestions):
    with cols[i]:
        if st.button(sug, key=f"sug_{i}"):
            # Could trigger a question here in future
            st.info(f"Selected: {sug}")

st.caption("ResearchMate can make mistakes. Always verify important information.")
