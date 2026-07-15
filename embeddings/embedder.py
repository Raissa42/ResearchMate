"""
embeddings/embedder.py
=========================================================================
PHASE 8 - Embeddings

Job of this file: convert text (chunks OR user questions) into
EMBEDDINGS - lists of numbers that represent the MEANING of that text.

-------------------------------------------------------------------------
WHAT IS AN EMBEDDING? (plain English)
-------------------------------------------------------------------------
An embedding turns a piece of text into a long list of numbers (a vector),
for example: [0.021, -0.114, 0.87, ... ] (1536+ numbers long).

The key idea: texts with SIMILAR MEANING end up with vectors that are
close together in this numeric space, even if they don't share any of
the same words. For example:
    "The car broke down"        -> vector A
    "The vehicle stopped working" -> vector B
    "I like pizza"                -> vector C

Vector A and B will be close together (similar meaning).
Vector C will be far from both (unrelated meaning).

This is what lets us do SEMANTIC SEARCH later (Phase 10) - finding
relevant chunks based on MEANING, not just keyword matching.

-------------------------------------------------------------------------
WHY TWO DIFFERENT "TASK TYPES"?
-------------------------------------------------------------------------
Gemini's embedding model can slightly adjust how it embeds text depending
on whether that text is:
    - a DOCUMENT chunk that will be searched later   (RETRIEVAL_DOCUMENT)
    - a QUESTION that is searching for something      (RETRIEVAL_QUERY)

Using the correct task_type for each case makes retrieval measurably more
accurate, since the model optimizes the vector slightly differently for
"things to be found" versus "things doing the finding".
"""

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Importing config.settings triggers load_dotenv() and validates that
# GOOGLE_API_KEY exists BEFORE we try to use it below. This must come
# before we read GEMINI_EMBEDDING_MODEL from the environment.
from config import settings  # noqa: F401  (imported for its side effect)

# Stable, well-supported embedding model (as of mid-2026).
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def get_document_embedder() -> GoogleGenerativeAIEmbeddings:
    """
    Returns an embedder configured for embedding DOCUMENT CHUNKS
    (i.e. the text chunks we extracted from PDFs in Phase 7).

    Use this when embedding chunks BEFORE storing them in ChromaDB.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
    )


def get_query_embedder() -> GoogleGenerativeAIEmbeddings:
    """
    Returns an embedder configured for embedding a USER'S QUESTION.

    Use this when a user types a question, right before we search
    ChromaDB for the most relevant chunks.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY",
    )


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Takes the list of chunk dictionaries from Phase 7 (text_splitter.py)
    and attaches an "embedding" vector to each one.

    Parameters
    ----------
    chunks : list[dict]
        Output of text_splitter.split_multiple_documents(), shaped like:
            [{"text": "...", "source": "Unit4.pdf", "chunk_id": 0}, ...]

    Returns
    -------
    list[dict]
        The same chunks, each now with an added "embedding" key:
            [{"text": "...", "source": "...", "chunk_id": 0,
              "embedding": [0.02, -0.11, ...]}, ...]

    Note
    ----
    We embed all chunk texts in ONE batch call rather than looping and
    calling the API once per chunk. This is both faster and cheaper,
    since it avoids one network round-trip per chunk.
    """
    if not chunks:
        return []

    embedder = get_document_embedder()

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.embed_documents(texts)  # one batched API call

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks


def embed_query(question: str) -> list[float]:
    """
    Embeds a single user question, ready to be compared against
    stored chunk embeddings in ChromaDB (Phase 10).

    Parameters
    ----------
    question : str
        The user's typed question, e.g. "What is reinforcement learning?"

    Returns
    -------
    list[float]
        The embedding vector for this question.
    """
    embedder = get_query_embedder()
    return embedder.embed_query(question)
