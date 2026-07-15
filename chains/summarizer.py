"""
chains/summarizer.py
=========================================================================
PHASE 13 (part 1) - Summarization Pipeline

Job of this file: summarize ONE document, or ALL uploaded documents,
using their FULL content - not a similarity-searched subset.

-------------------------------------------------------------------------
WHY THIS CAN'T JUST REUSE THE STANDARD RAG PIPELINE
-------------------------------------------------------------------------
The standard RAG pipeline (Phase 12) retrieves chunks that are most
SIMILAR IN MEANING to a question. But "summarize this paper" isn't
really a question with a specific meaning to search for - it needs
ALL of the paper's content, not just a few chunks that happen to be
"closest" to the word "summarize".

So instead of searching, we simply fetch every chunk belonging to a
document (via chroma_store.get_all_chunks_for_source) and hand the
WHOLE thing to Gemini with summarization instructions.
"""

from chains.gemini_client import get_chat_model
from vectorstore.chroma_store import list_all_sources, get_all_chunks_for_source


SUMMARY_PROMPT_TEMPLATE = """You are ResearchMate, an AI research assistant.

Summarize the following document in a clear, well-organized way.
Cover the main points, key findings, and overall purpose of the document.
Base your summary STRICTLY on the content given below - do not add
information from outside knowledge.

Document: {source_name}

Content:
{content}

Summary:"""


def summarize_document(source_name: str) -> str:
    """
    Summarizes ONE document using its full stored content.

    Parameters
    ----------
    source_name : str
        Exact filename, e.g. "Unit4.pdf".

    Returns
    -------
    str
        Gemini's summary of that document.
    """
    chunks = get_all_chunks_for_source(source_name)

    if not chunks:
        return f"No content found for {source_name}."

    full_content = "\n\n".join(chunks)

    llm = get_chat_model()
    prompt = SUMMARY_PROMPT_TEMPLATE.format(source_name=source_name, content=full_content)
    response = llm.invoke(prompt)

    return response.content


def summarize_all_documents() -> dict:
    """
    Summarizes EVERY document currently stored in the vectorstore.

    Returns
    -------
    dict
        {
            "summaries": {"Unit4.pdf": "...", "Unit6.pdf": "..."},
            "sources": ["Unit4.pdf", "Unit6.pdf"]
        }
    """
    sources = list_all_sources()

    if not sources:
        return {
            "summaries": {},
            "sources": [],
        }

    summaries = {}
    for source_name in sources:
        summaries[source_name] = summarize_document(source_name)

    return {
        "summaries": summaries,
        "sources": sources,
    }
