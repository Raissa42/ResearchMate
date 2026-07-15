"""
utils/pdf_loader.py
=========================================================================
PHASE 6 - PDF Loader

Job of this file: take uploaded PDF file(s) and turn them into plain
text that the rest of the pipeline (chunking, embeddings) can work with.

This file does NOT do chunking or embeddings - it only extracts raw text.
Keeping this responsibility separate makes each piece easy to test and
debug on its own.
"""

from pypdf import PdfReader


def load_pdf_text(uploaded_file) -> str:
    """
    Extracts all text from a single uploaded PDF file.

    Parameters
    ----------
    uploaded_file : a file-like object
        This is what Streamlit's st.file_uploader() gives us - it behaves
        like a normal Python file object, so pypdf can read it directly
        without needing to save it to disk first.

    Returns
    -------
    str
        The full extracted text of the PDF, with pages joined by
        double newlines. Returns an empty string if extraction fails
        or the PDF has no readable text (e.g. a scanned image PDF).
    """
    try:
        reader = PdfReader(uploaded_file)
    except Exception as error:
        # A corrupted or unsupported PDF should not crash the whole app.
        print(f"[pdf_loader] Failed to read PDF: {error}")
        return ""

    extracted_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # some pages (e.g. pure images) return None or ""
            extracted_pages.append(page_text)

    full_text = "\n\n".join(extracted_pages)
    return full_text


def load_multiple_pdfs(uploaded_files) -> dict:
    """
    Extracts text from multiple uploaded PDFs at once.

    Parameters
    ----------
    uploaded_files : list
        A list of file-like objects, exactly what Streamlit's
        st.file_uploader(accept_multiple_files=True) returns.

    Returns
    -------
    dict
        A dictionary mapping {filename: extracted_text}.
        Example:
            {
                "Unit4_HDL.pdf": "full text here...",
                "Unit6_Systems.pdf": "full text here..."
            }
        This mapping is what lets us later tell the user
        "this answer came from Unit4_HDL.pdf" - even without page numbers,
        we still track which DOCUMENT each chunk belongs to.
    """
    documents = {}

    for uploaded_file in uploaded_files:
        text = load_pdf_text(uploaded_file)

        if not text.strip():
            print(f"[pdf_loader] Warning: no text extracted from {uploaded_file.name}")
            continue

        documents[uploaded_file.name] = text

    return documents
