import os
import glob

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")


def preload_docs():
    """Ingest any PDFs found in ../docs/ at startup."""
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if not pdf_files:
        print("[preload] No PDFs found in docs/ — skipping preload")
        return

    # Import here to avoid circular import at module level
    from rag import ingest_pdf, list_documents

    already_loaded = set(list_documents())

    for pdf_path in pdf_files:
        doc_name = os.path.basename(pdf_path)
        if doc_name in already_loaded:
            print(f"[preload] Already loaded: {doc_name}")
            continue
        try:
            with open(pdf_path, "rb") as f:
                data = f.read()
            result = ingest_pdf(data, doc_name)
            print(f"[preload] {doc_name}: {result['chunks_created']} chunks ingested")
        except Exception as e:
            print(f"[preload] Failed to load {doc_name}: {e}")
