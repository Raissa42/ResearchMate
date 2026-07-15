"""
agents/router.py
=========================================================================
PHASE 13 (part 3) - The AI Agent (Dynamic Pipeline Router)

Job of this file: look at the user's question, decide WHICH pipeline
should handle it, run that pipeline, and return a consistent result
shape regardless of which pipeline actually ran.

-------------------------------------------------------------------------
HOW THE ROUTING DECISION WORKS (rule-based, for now)
-------------------------------------------------------------------------
We use simple KEYWORD matching to detect intent. This is:
    - Free (no extra API call)
    - Instant
    - Easy to explain and defend in a viva
    - Good enough for clearly-phrased requests

A more advanced version could ask Gemini itself to classify intent
(see the note at the bottom of this file) - but rule-based routing
is the right STARTING point, and is exactly how many real production
systems begin before adding LLM-based classification later.
"""

from chains.rag_chain import run_rag_pipeline
from chains.summarizer import summarize_all_documents, summarize_document
from chains.comparator import compare_documents
from vectorstore.chroma_store import list_all_sources


# -------------------------------------------------------------------
# Keyword sets used for rule-based intent classification.
# Lowercase, since we lowercase the question before checking.
# -------------------------------------------------------------------
SUMMARIZE_KEYWORDS = ["summarize", "summary", "summarise", "overview of"]
COMPARE_KEYWORDS = ["compare", "comparison", "difference between", "similarities between", "vs "]
EXTRACT_KEYWORDS = ["list all", "extract", "list the", "find all methodologies", "find all conclusions"]


def classify_intent(question: str) -> str:
    """
    Classifies a question into one of: "summarize", "compare",
    "extract", or "qa" (standard question-answering, the default).

    Parameters
    ----------
    question : str
        The user's typed question.

    Returns
    -------
    str
        One of "summarize", "compare", "extract", "qa".
    """
    lowered = question.lower()

    if any(keyword in lowered for keyword in SUMMARIZE_KEYWORDS):
        return "summarize"

    if any(keyword in lowered for keyword in COMPARE_KEYWORDS):
        return "compare"

    if any(keyword in lowered for keyword in EXTRACT_KEYWORDS):
        return "extract"

    return "qa"


import re


def _find_mentioned_sources(question: str) -> list[str]:
    """
    Looks for known document filenames mentioned inside the question
    text, so the Comparison pipeline knows WHICH two documents to use
    without asking the user to type exact filenames awkwardly.

    Matches both the full filename and the filename without its
    extension (e.g. both "Unit4.pdf" and "Unit4" would match), using
    WORD BOUNDARIES so a short name like "B" doesn't accidentally
    match inside an unrelated word like "between".
    """
    lowered_question = question.lower()
    all_sources = list_all_sources()

    mentioned = []
    for source in all_sources:
        name_without_ext = source.rsplit(".", 1)[0].lower()

        full_pattern = r"\b" + re.escape(source.lower()) + r"\b"
        name_pattern = r"\b" + re.escape(name_without_ext) + r"\b"

        if re.search(full_pattern, lowered_question) or re.search(name_pattern, lowered_question):
            mentioned.append(source)

    return mentioned


def route_question(question: str, top_k: int = 4) -> dict:
    """
    THE AGENT'S MAIN ENTRY POINT.

    Classifies the question's intent, runs the matching pipeline, and
    returns a result in a CONSISTENT shape - so app.py (Phase 14) can
    display any answer the same way, regardless of which pipeline
    actually produced it.

    Parameters
    ----------
    question : str
        The user's typed question.
    top_k : int
        Retrieval depth for the standard "qa" pipeline (Phase 15 setting,
        user-adjustable in the sidebar). Does not affect summarize/compare,
        which always use full document content, or extract, which always
        uses a wider fixed net (8) since it needs broader coverage.

    Returns
    -------
    dict
        {
            "answer": "...",
            "sources": [...],
            "pipeline_used": "qa" | "summarize" | "compare" | "extract"
        }
    """
    intent = classify_intent(question)

    if intent == "summarize":
        # "Summarize this paper" (singular) vs "summarize all papers" (plural) -
        # if exactly one source is mentioned by name, summarize just that one.
        mentioned = _find_mentioned_sources(question)
        if len(mentioned) == 1:
            answer = summarize_document(mentioned[0])
            return {"answer": answer, "sources": mentioned, "pipeline_used": "summarize"}

        result = summarize_all_documents()
        if not result["sources"]:
            return {
                "answer": "No documents have been uploaded and processed yet.",
                "sources": [],
                "pipeline_used": "summarize",
            }

        combined_answer = "\n\n".join(
            f"**{source}**\n{summary}" for source, summary in result["summaries"].items()
        )
        return {"answer": combined_answer, "sources": result["sources"], "pipeline_used": "summarize"}

    if intent == "compare":
        mentioned = _find_mentioned_sources(question)
        all_sources = list_all_sources()

        # If exactly 2 documents exist in total, "compare them" is
        # unambiguous - there's only one possible pair - so we don't
        # need the user to name them explicitly.
        if len(mentioned) < 2 and len(all_sources) == 2:
            mentioned = all_sources

        if len(mentioned) < 2:
            available = ", ".join(all_sources) or "no documents uploaded yet"
            return {
                "answer": (
                    "I need you to mention two specific documents to compare. "
                    f"Available documents: {available}."
                ),
                "sources": [],
                "pipeline_used": "compare",
            }

        source_a, source_b = mentioned[0], mentioned[1]
        answer = compare_documents(source_a, source_b)
        return {"answer": answer, "sources": [source_a, source_b], "pipeline_used": "compare"}

    if intent == "extract":
        # Extraction (e.g. "list all methodologies") reuses the standard
        # RAG pipeline, but with a WIDER retrieval net (higher top_k),
        # since extraction questions often need info spread across many
        # chunks and documents, not just the 4 most similar ones.
        result = run_rag_pipeline(question, top_k=8)
        return {"answer": result["answer"], "sources": result["sources"], "pipeline_used": "extract"}

    # Default: standard question-answering pipeline
    result = run_rag_pipeline(question, top_k=top_k)
    return {"answer": result["answer"], "sources": result["sources"], "pipeline_used": "qa"}


# -------------------------------------------------------------------
# NOTE for your report/viva: an LLM-based classifier alternative
# -------------------------------------------------------------------
# Instead of (or in addition to) keyword matching, you could ask
# Gemini itself to classify intent, e.g.:
#
#   classification_prompt = f'''Classify this question into exactly one
#   word - summarize, compare, extract, or qa - and respond with ONLY
#   that word: "{question}"'''
#
# This would handle more varied phrasing (e.g. "give me the gist of
# all these papers" wouldn't match our keyword list, but an LLM
# classifier would likely still catch it as "summarize"). The tradeoff
# is one extra API call per question. A good extension to mention in
# your final report as "future work".
