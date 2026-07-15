# 📚 ResearchMate

**An Agentic Retrieval-Augmented Generation (RAG) System for Multi-Document Research Analysis, powered by Google Gemini.**

ResearchMate lets you upload multiple research papers (PDFs) and ask questions, request summaries, compare documents, and extract key information — all answered strictly from your uploaded content, with sources cited, and no hallucination.

---

## ✨ Features

- 📄 Upload and process multiple PDF research papers at once
- 🔎 Semantic search over document content (not just keyword matching)
- 🤖 **Dynamic Agent routing** — automatically detects whether you want a normal answer, a summary, a comparison, or an extraction, and runs the appropriate pipeline
- 📝 Summarize a single paper or all uploaded papers at once
- ⚖️ Compare two papers side-by-side (similarities, differences, methodology)
- 🎯 Extraction queries (e.g. "list all methodologies")
- 🧭 Every answer cites which document it came from
- 🚫 Refuses to answer if the information isn't in the uploaded documents — never hallucinates
- 🔄 One-click session reset to start fresh with new documents
- ⚙️ Adjustable retrieval depth via a settings panel

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
Text Extraction (pypdf)
    │
    ▼
Chunking (LangChain RecursiveCharacterTextSplitter)
    │
    ▼
Embeddings (Google Gemini - gemini-embedding-001)
    │
    ▼
Vector Storage (ChromaDB, persistent)
    │
    ▼
┌───────────────────────────────────────┐
│           AI Agent (Router)           │
│  Classifies intent -> picks pipeline  │
└───────────────────────────────────────┘
    │
    ├── Standard Q&A     → Semantic retrieval + Gemini
    ├── Summarization     → Full document content + Gemini
    ├── Comparison        → Two full documents + Gemini
    └── Extraction        → Wider semantic retrieval + Gemini
    │
    ▼
Grounded Answer + Source Citations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| LLM | Google Gemini (`gemini-2.5-flash`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Database | ChromaDB (persistent) |
| PDF Parsing | pypdf |

---

## 📂 Project Structure

```
ResearchMate/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── settings.py         # Loads API keys, central config
├── utils/
│   ├── pdf_loader.py        # PDF text extraction
│   └── text_splitter.py     # Chunking logic
├── embeddings/
│   └── embedder.py          # Gemini embedding wrapper
├── vectorstore/
│   └── chroma_store.py      # ChromaDB persistence layer
├── chains/
│   ├── retriever.py          # Semantic search
│   ├── gemini_client.py      # Gemini chat model + RAG prompt
│   ├── rag_chain.py           # Standard Q&A pipeline
│   ├── summarizer.py          # Summarization pipeline
│   └── comparator.py          # Comparison pipeline
└── agents/
    └── router.py              # Dynamic intent-based routing
```

---

## 🚀 Getting Started (Local)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ResearchMate.git
cd ResearchMate

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Open .env and paste your real Gemini API key
# Get one at: https://aistudio.google.com/app/apikey

# 5. Run the app
streamlit run app.py
```

---

## ☁️ Deployment

Deployed on **Streamlit Community Cloud**: `[your deployed link here]`

To deploy your own copy:
1. Push this repository to your own GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to `app.py`
3. In the app's **Secrets** settings, add:
   ```toml
   GOOGLE_API_KEY = "your_real_key_here"
   ```

---

## 🎓 Academic Context

Built as a final AI Engineering project, evolving from an earlier project (MindEase - a student mental health chatbot), applying the same Streamlit + Gemini foundations to a more advanced agentic RAG architecture.

**Faculty Supervisor:** Harjeet Kaur

---

## 🔮 Future Improvements

- Page-number-level citation tracking
- LLM-based intent classification as a fallback to keyword routing
- Conversation memory for natural follow-up questions
- Export answers/comparisons as CSV, TXT, or Markdown
- Multi-turn "Gap Finder" for identifying open research questions across papers

---

## 📄 License

This project is for academic purposes.
