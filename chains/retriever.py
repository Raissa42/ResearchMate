"""
chains/retriever.py
=========================================================================
PHASE 10 - Retriever

Job of this file: given a user's question, find the most relevant
stored chunks from ChromaDB using SEMANTIC SEARCH (similarity of
meaning, not keyword matching).

-------------------------------------------------------------------------
WHAT IS SIMILARITY SEARCH? (plain English)
-------------------------------------------------------------------------
Every chunk we stored in Phase 9 has an embedding vector - its position
in a 3072-dimensional "meaning space". When a user asks a question, we:

  1. Embed the question itself (same meaning space, same 3072 numbers)
  2. Ask ChromaDB: "which stored chunk vectors are geometrically
     CLOSEST to this question's vector?"
  3. Return the top K closest chunks (K = how many we want, e.g. 4)

"Closest" is typically measured using cosine similarity - basically,
how similar the DIRECTION of two vectors is, regardless of their length.
Two vectors pointing in nearly the same direction = very similar meaning.

This is fundamentally different from keyword search (like Ctrl+F) -
a question like "What is RL?" can match a chunk that says "reinforcement
learning" even though NOT ONE WORD is shared between them, because
their MEANINGS are close in this vector space.
"""

from langchain_chroma import Chroma

from vectorstore.chroma_store import get_vectorstore

# How many chunks to retrieve per question. More = more context for
# Gemini to work with, but also more tokens (cost) and noise.
DEFAULT_TOP_K = 4


def retrieve_relevant_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Finds the most relevant stored chunks for a given question.

    Parameters
    ----------
    question : str
        The user's typed question.
    top_k : int
        How many chunks to retrieve, ranked by relevance (default: 4).

    Returns
    -------
    list[dict]
        A list of retrieved chunks, each shaped like:
            {
                "text": "...chunk content...",
                "source": "Unit4.pdf",
                "chunk_id": 0,
                "similarity_score": 0.82
            }
        Sorted from MOST relevant to LEAST relevant.

    Note
    ----
    Chroma's similarity_search_with_score() handles embedding the
    question for us internally (using the vectorstore's configured
    embedding_function), so we don't need to call embed_query()
    ourselves here.
    """
    vectorstore: Chroma = get_vectorstore()

    # Returns a list of (Document, distance_score) tuples.
    # NOTE: Chroma's default distance is often "L2" (lower = more similar),
    # not cosine similarity directly - we normalize this below so the
    # rest of our app can just think in terms of "higher score = better".
    results = vectorstore.similarity_search_with_score(question, k=top_k)

    retrieved_chunks = []
    for document, distance in results:
        retrieved_chunks.append({
            "text": document.page_content,
            "source": document.metadata.get("source", "unknown"),
            "chunk_id": document.metadata.get("chunk_id", -1),
            # Convert distance (lower=better) into an intuitive
            # 0-1 "similarity_score" (higher=better) for display in the UI.
            "similarity_score": 1 / (1 + distance),
        })

    return retrieved_chunks


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a clean text block ready to be
    inserted into a Gemini prompt (used in Phase 11-12).

    Each chunk is clearly labeled with its source, so Gemini can
    (and must, per our prompt instructions) cite which document
    each piece of information came from.

    Parameters
    ----------
    chunks : list[dict]
        Output of retrieve_relevant_chunks().

    Returns
    -------
    str
        A formatted string like:
            [Source: Unit4.pdf]
            <chunk text>

            [Source: Unit6.pdf]
            <chunk text>
    """
    if not chunks:
        return "No relevant content was found in the uploaded documents."

    formatted_blocks = []
    for chunk in chunks:
        block = f"[Source: {chunk['source']}]\n{chunk['text']}"
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)
