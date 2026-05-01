import fitz  # PyMuPDF
from typing import List, Dict, Any


def parse_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract text page-by-page from PDF bytes."""
    pages = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append({"text": text, "page_number": page_num + 1})
        doc.close()
    except Exception as e:
        print(f"[chunker] PDF parse error: {e}")
    return pages


def recursive_split(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """Split text recursively on separators until chunks are small enough."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        chunks.append(current)
                    # Part itself may be too large — recurse with remaining separators
                    remaining_seps = separators[separators.index(sep) + 1:]
                    if len(part) > chunk_size and remaining_seps:
                        chunks.extend(recursive_split(part, remaining_seps, chunk_size))
                    else:
                        current = part
            if current.strip():
                chunks.append(current)
            return chunks

    # No separator worked — hard split
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_pages(
    pages: List[Dict[str, Any]],
    source_doc: str,
    chunk_size: int = 512,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Chunk extracted pages into overlapping text segments with metadata."""
    separators = ["\n\n", "\n", ". ", " "]
    raw_chunks: List[Dict[str, Any]] = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        splits = recursive_split(text, separators, chunk_size)

        # Apply overlap by merging adjacent chunk tails
        overlapped: List[str] = []
        for i, chunk in enumerate(splits):
            if i > 0 and overlap > 0:
                prev = splits[i - 1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                chunk = tail + " " + chunk
            overlapped.append(chunk.strip())

        for chunk_text in overlapped:
            if chunk_text:
                raw_chunks.append({"text": chunk_text, "page_number": page_number})

    total = len(raw_chunks)
    result = []
    for idx, item in enumerate(raw_chunks):
        result.append(
            {
                "text": item["text"],
                "metadata": {
                    "source_doc": source_doc,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "page_number": item["page_number"],
                },
            }
        )
    return result
