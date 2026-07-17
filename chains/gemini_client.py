"""
chains/gemini_client.py
=========================================================================
PHASE 11 - Gemini Integration

Job of this file: wrap the Gemini CHAT model (not the embedding model -
that's a separate model, from Phase 8) and define the PROMPT TEMPLATE
that tells Gemini exactly how to behave as ResearchMate's answer engine.

-------------------------------------------------------------------------
WHY A PROMPT TEMPLATE MATTERS THIS MUCH FOR THIS PROJECT
-------------------------------------------------------------------------
Your project's core promise is: "answer ONLY from the uploaded documents,
and NEVER hallucinate." Gemini doesn't know this by default - it will
happily answer from its own general training knowledge unless we
EXPLICITLY instruct it not to, every single time, inside the prompt.

The prompt template below is doing the heavy lifting for that promise.
It:
    1. Gives Gemini the retrieved chunks as its ONLY source of truth
    2. Explicitly forbids answering from outside knowledge
    3. Tells it exactly what to say when the answer isn't found
    4. Asks it to mention which source document it used
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")


# -------------------------------------------------------------------
# THE CORE PROMPT TEMPLATE
# -------------------------------------------------------------------
# {context}  -> filled in with retriever.py's format_chunks_for_prompt()
# {question} -> filled in with the user's actual question
RAG_PROMPT_TEMPLATE = """You are ResearchMate, an AI research assistant that answers questions
STRICTLY based on the content of documents the user has uploaded.

RULES YOU MUST FOLLOW:
1. Answer ONLY using the information in the "Context" section below.
2. Do NOT use any outside knowledge, even if you know the answer from
   general training - only the given context counts as truth here.
3. If the answer is not present in the context, respond EXACTLY with:
   "I could not find this information inside the uploaded documents."
4. When you do answer, mention which document(s) the information came
   from, using the [Source: filename] labels given in the context.
5. Be clear and concise. Do not pad your answer with unnecessary text.

Context:
{context}

Question:
{question}

Answer:"""


def get_chat_model() -> ChatGoogleGenerativeAI:
    """
    Returns a configured Gemini chat model instance.

    Returns
    -------
    ChatGoogleGenerativeAI
        A LangChain-wrapped Gemini chat model, ready to call .invoke()
        on with a prompt string.
    """
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,  # low temperature = more focused, less "creative"
    )


def ask_gemini(context: str, question: str) -> str:
    """
    Sends a fully-formed RAG prompt (context + question) to Gemini and
    returns its plain text answer.

    Parameters
    ----------
    context : str
        The formatted, source-labeled chunk text from
        retriever.format_chunks_for_prompt().
    question : str
        The user's original question.

    Returns
    -------
    str
        Gemini's answer, grounded in the given context.
    """
    llm = get_chat_model()

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    response = llm.invoke(prompt)

    return response.content
