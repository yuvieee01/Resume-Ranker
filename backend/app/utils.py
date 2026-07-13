import os
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Pull plaintext out of a PDF byte stream."""
    try:
        return pdf_extract_text(BytesIO(file_bytes))
    except Exception:
        return ""


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """Return plaintext from an uploaded file based on its extension."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        return ""
