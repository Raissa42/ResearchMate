"""
utils/text_splitter.py
=========================================================================
PHASE 7 - Text Splitter

Job of this file: take the raw text extracted from a PDF (one big string)
and break it into smaller overlapping "chunks" that are the right size
for embedding and semantic search.

We use LangChain's RecursiveCharacterTextSplitter, which is smart about
WHERE it cuts text - it tries to split on paragraph breaks first, then
sentences, then words, only cutting mid-word as a last resort. This keeps
chunks semantically coherent instead of chopping ideas in half randomly.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------------------------------------------
# Chunking settings - tuned for research paper / lecture note style text
# -------------------------------------------------------------------
CHUNK_SIZE = 1000       # characters per chunk (~150-200 words)
CHUNK_OVERLAP = 150     # characters shared between consecutive chunks


def split_text(text: str, source_name: str) -> list[dict]:
    """
    Splits a single document's text into overlapping chunks.

    Parameters
    ----------
    text : str
        The full raw text of one document (from pdf_loader.py).
    source_name : str
        The filename this text came from, e.g. "Unit4_HDL.pdf".
        We attach this to every chunk so we can always trace an
        answer back to which document it came from.

    Returns
    -------
    list[dict]
        A list of chunk dictionaries, each shaped like:
            {
                "text": "...chunk content...",
                "source": "Unit4_HDL.pdf",
                "chunk_id": 0
            }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries these in order
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "text": chunk_text,
            "source": source_name,
            "chunk_id": i,
        })

    return chunks


def split_multiple_documents(documents: dict) -> list[dict]:
    """
    Splits ALL uploaded documents into one combined list of chunks.

    Parameters
    ----------
    documents : dict
        Output of pdf_loader.load_multiple_pdfs(), shaped like:
            {"Unit4_HDL.pdf": "full text...", "Unit6.pdf": "full text..."}

    Returns
    -------
    list[dict]
        A single flat list of chunk dictionaries across ALL documents,
        ready to be passed into the embedding step (Phase 8).
    """
    all_chunks = []

    for source_name, text in documents.items():
        doc_chunks = split_text(text, source_name)
        all_chunks.extend(doc_chunks)
        print(f"[text_splitter] {source_name}: {len(doc_chunks)} chunks created")

    return all_chunks
