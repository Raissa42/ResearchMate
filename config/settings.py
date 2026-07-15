"""
config/settings.py
=========================================================================
Central configuration file.

Job of this file: load environment variables from .env ONCE, and expose
them as clean Python variables the rest of the app can import.

WHY THIS MATTERS:
Just having a .env file sitting in your folder does nothing by itself.
Something has to actually call load_dotenv() to read that file and
inject its values into the environment (os.environ), where libraries
like langchain_google_genai look for GOOGLE_API_KEY.

By putting load_dotenv() here, and importing THIS file early (e.g. at
the top of app.py), we guarantee the .env file gets loaded before any
other part of the app tries to use an API key.
"""

import os
from dotenv import load_dotenv

# Reads the .env file in your project root and loads its values into
# os.environ. If .env doesn't exist, this simply does nothing (no crash).
load_dotenv()

# -------------------------------------------------------------------
# Centralized settings - import these instead of calling os.getenv()
# scattered across different files.
# -------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "vectorstore/chroma_db")

# -------------------------------------------------------------------
# Fail loudly and clearly if the API key is missing, instead of letting
# a confusing pydantic/validation error surface deep inside a library.
# -------------------------------------------------------------------
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. \n"
        "Fix: 1) Copy .env.example to .env in your project root. \n"
        "     2) Open .env and paste your real Gemini API key. \n"
        "     3) Make sure .env is in the SAME folder as app.py."
    )
