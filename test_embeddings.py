"""
test_embeddings.py
=========================================================================
Standalone test script for Phase 8 - run this instead of pasting lines
into an interactive Python shell, to avoid module caching confusion.

Run with:
    python test_embeddings.py
"""

from utils.pdf_loader import load_pdf_text
from utils.text_splitter import split_text
from embeddings.embedder import embed_chunks

with open("data/test.pdf", "rb") as f:
    text = load_pdf_text(f)

print(f"Extracted {len(text)} characters from PDF.")

chunks = split_text(text, "test.pdf")
print(f"Created {len(chunks)} chunks.")

embedded_chunks = embed_chunks(chunks)

print(f"Chunk 0 text: {embedded_chunks[0]['text'][:80]}...")
print(f"Chunk 0 embedding length: {len(embedded_chunks[0]['embedding'])}")
print("SUCCESS: embeddings created correctly.")
