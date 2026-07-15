"""
vectorstore/chroma_store.py
=========================================================================
PHASE 9 - Vector Database (ChromaDB)

Job of this file: store chunk embeddings persistently on disk, and give
us back a "vectorstore" object we can search later (Phase 10).

-------------------------------------------------------------------------
WHAT IS A VECTOR DATABASE, IN PLAIN ENGLISH?
-------------------------------------------------------------------------
A normal database stores rows and lets you search by exact matches
("find rows where name = 'Raissa'"). A VECTOR database instead stores
embeddings (lists of numbers) and lets you search by SIMILARITY -
"find the vectors that are closest in meaning to this other vector."

Under the hood, ChromaDB organizes these vectors so it can quickly find
the "nearest neighbors" of any given vector, without comparing against
every single stored vector one by one (which would be slow once you have
thousands of chunks).

-------------------------------------------------------------------------
WHY PERSISTENT STORAGE?
-------------------------------------------------------------------------
Without persistence, every time you restart the Streamlit app, ChromaDB
would forget everything and you'd have to re-upload and re-embed your
PDFs (which costs API calls and time). By pointing Chroma at a folder
on disk (vectorstore/chroma_db/), the database survives app restarts.

-------------------------------------------------------------------------
NOTE ON EMBEDDINGS: Chroma handles embedding internally
-------------------------------------------------------------------------
In Phase 8, embedder.py manually created embeddings so you could SEE
and understand what an embedding looks like. From here on, we let the
Chroma vectorstore call the embedder FOR us automatically whenever we
add or search text - this is the standard LangChain pattern, and avoids
us manually managing embedding vectors ourselves.
"""

from langchain_chroma import Chroma

from config import settings
from embeddings.embedder import get_document_embedder


def get_vectorstore() -> Chroma:
    """
    Returns a Chroma vectorstore connected to our persistent folder
    on disk (config.settings.CHROMA_PERSIST_DIR).

    If the folder already has data from a previous run, that data is
    loaded automatically - you don't need to re-add it.

    Returns
    -------
    Chroma
        A LangChain Chroma vectorstore instance, ready to add texts
        to or search.
    """
    return Chroma(
        collection_name="researchmate_docs",
        embedding_function=get_document_embedder(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def add_chunks_to_vectorstore(chunks: list[dict]) -> Chroma:
    """
    Adds a list of text chunks (from Phase 7's text_splitter.py) into
    the persistent Chroma vectorstore. Chroma will call Gemini's
    embedding API internally for each chunk's text.

    Parameters
    ----------
    chunks : list[dict]
        Shaped like: [{"text": "...", "source": "Unit4.pdf", "chunk_id": 0}, ...]
        (the raw output of text_splitter.py - "embedding" key not required)

    Returns
    -------
    Chroma
        The vectorstore, now containing the newly added chunks.
    """
    if not chunks:
        raise ValueError("No chunks provided - nothing to add to the vectorstore.")

    vectorstore = get_vectorstore()

    texts = [chunk["text"] for chunk in chunks]

    # Metadata lets us trace every stored chunk back to its source file
    # and chunk number - this is what powers "this answer came from X.pdf".
    metadatas = [
        {"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
        for chunk in chunks
    ]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)

    return vectorstore


def get_vectorstore_chunk_count() -> int:
    """
    Utility function: returns how many chunks are currently stored.
    Useful for debugging and for showing the user "X chunks indexed"
    in the Streamlit sidebar later.
    """
    vectorstore = get_vectorstore()
    return vectorstore._collection.count()


def list_all_sources() -> list[str]:
    """
    Returns a list of unique source filenames currently stored in the
    vectorstore, e.g. ["Unit4.pdf", "Unit6.pdf"].

    Used by the Summarization and Comparison pipelines (Phase 13) to
    know WHICH documents exist, without the user having to re-tell us.
    """
    vectorstore = get_vectorstore()
    all_data = vectorstore._collection.get(include=["metadatas"])

    sources = set()
    for metadata in all_data["metadatas"]:
        if metadata and "source" in metadata:
            sources.add(metadata["source"])

    return sorted(sources)


def get_all_chunks_for_source(source_name: str) -> list[str]:
    """
    Returns ALL stored chunk texts belonging to ONE specific document,
    in original chunk order.

    Unlike similarity search (which returns only the "most relevant"
    chunks for a question), this returns EVERYTHING from one document -
    exactly what summarization and comparison need, since they must
    cover the whole document, not just parts similar to a query.

    Parameters
    ----------
    source_name : str
        Exact filename to filter by, e.g. "Unit4.pdf".

    Returns
    -------
    list[str]
        Chunk texts, ordered by their original chunk_id.
    """
    vectorstore = get_vectorstore()
    results = vectorstore._collection.get(
        where={"source": source_name},
        include=["documents", "metadatas"],
    )

    # Pair each document with its chunk_id so we can sort back into
    # original reading order (ChromaDB does not guarantee stored order).
    paired = list(zip(results["documents"], results["metadatas"]))
    paired.sort(key=lambda pair: pair[1].get("chunk_id", 0))

    return [doc for doc, _ in paired]


def clear_vectorstore() -> None:
    """
    Deletes ALL stored chunks and recreates an empty collection.

    Used by app.py (Phase 14) so that clicking "Process Documents"
    starts fresh each time - only the newly uploaded PDFs become
    searchable, instead of accumulating every document ever uploaded
    across every session.
    """
    vectorstore = get_vectorstore()
    vectorstore.reset_collection()
