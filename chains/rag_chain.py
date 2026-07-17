"""
chains/rag_chain.py
=========================================================================
PHASE 12 - RAG Chain

Job of this file: combine everything from Phase 10 (retrieval) and
Phase 11 (generation) into ONE clean function - the "standard RAG
pipeline" that most of your app's questions will flow through.

-------------------------------------------------------------------------
WHY WRAP THIS UP INSTEAD OF CALLING retriever.py + gemini_client.py
DIRECTLY EVERYWHERE?
-------------------------------------------------------------------------
Without this file, every place in this app that wants to "ask a
question about the documents" would need to:
    1. Import retriever.py
    2. Call retrieve_relevant_chunks()
    3. Import gemini_client.py
    4. Call format_chunks_for_prompt()
    5. Call ask_gemini()

That's a lot of repeated plumbing. By wrapping it into ONE function
here, the rest of the app (app.py, and later the Agent in Phase 13)
just calls run_rag_pipeline(question) and gets a complete answer back -
with the retrieved sources attached, so the UI can display them.

This is also exactly what your Dynamic Pipeline design calls the
"Standard Retrieval Pipeline" - the path used for normal factual
questions (as opposed to the Summarization, Comparison, or Extraction
pipelines we'll build starting Phase 13).
"""

from chains.retriever import retrieve_relevant_chunks, format_chunks_for_prompt
from chains.gemini_client import ask_gemini

# The exact phrase Gemini is instructed to use when nothing relevant
# is found - defined here too so other code (like the Agent) can check
# for it without hardcoding the string in multiple places.
NOT_FOUND_MESSAGE = "I could not find this information inside the uploaded documents."


def run_rag_pipeline(question: str, top_k: int = 4) -> dict:
    """
    Runs the full standard RAG pipeline: retrieve relevant chunks,
    then generate a grounded answer from Gemini.

    Parameters
    ----------
    question : str
        The user's typed question.
    top_k : int
        How many chunks to retrieve and give to Gemini as context.

    Returns
    -------
    dict
        {
            "answer": "...",              # Gemini's grounded answer
            "sources": ["Unit4.pdf", ...], # unique list of source filenames used
            "retrieved_chunks": [...]      # full chunk details, for debugging/UI
        }
    """
    retrieved_chunks = retrieve_relevant_chunks(question, top_k=top_k)

    context = format_chunks_for_prompt(retrieved_chunks)

    answer = ask_gemini(context, question)

    # Build a de-duplicated, ordered list of source filenames actually used.
    # (dict.fromkeys() is a simple trick to remove duplicates while
    # preserving the original order - a set alone would lose ordering.)
    unique_sources = list(dict.fromkeys(chunk["source"] for chunk in retrieved_chunks))

    return {
        "answer": answer,
        "sources": unique_sources,
        "retrieved_chunks": retrieved_chunks,
    }
