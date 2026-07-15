"""
chains/comparator.py
=========================================================================
PHASE 13 (part 2) - Comparison Pipeline

Job of this file: compare TWO specific documents - similarities,
differences, methodology, conclusions - using their full content.
"""

from chains.gemini_client import get_chat_model
from vectorstore.chroma_store import get_all_chunks_for_source


COMPARISON_PROMPT_TEMPLATE = """You are ResearchMate, an AI research assistant.

Compare the following two documents. Discuss:
- Key similarities between them
- Key differences between them
- How their approaches, methods, or conclusions differ (if applicable)

Base your comparison STRICTLY on the content given below - do not add
information from outside knowledge. If the documents don't have enough
information to compare on some point, say so honestly instead of
guessing.

Document A: {source_a}
Content A:
{content_a}

Document B: {source_b}
Content B:
{content_b}

Comparison:"""


def compare_documents(source_a: str, source_b: str) -> str:
    """
    Compares two documents using their full stored content.

    Parameters
    ----------
    source_a : str
        Exact filename of the first document, e.g. "Unit4.pdf".
    source_b : str
        Exact filename of the second document, e.g. "Unit6.pdf".

    Returns
    -------
    str
        Gemini's comparison of the two documents.
    """
    chunks_a = get_all_chunks_for_source(source_a)
    chunks_b = get_all_chunks_for_source(source_b)

    if not chunks_a:
        return f"No content found for {source_a}."
    if not chunks_b:
        return f"No content found for {source_b}."

    content_a = "\n\n".join(chunks_a)
    content_b = "\n\n".join(chunks_b)

    llm = get_chat_model()
    prompt = COMPARISON_PROMPT_TEMPLATE.format(
        source_a=source_a, content_a=content_a,
        source_b=source_b, content_b=content_b,
    )
    response = llm.invoke(prompt)

    return response.content
