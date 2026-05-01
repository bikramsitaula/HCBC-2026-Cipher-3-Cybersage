import os
import json
import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from chunker import parse_pdf, chunk_pages

CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "cybersage"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5

SYSTEM_PROMPT = """
You are CyberSage, a cybersecurity training assistant. You ONLY answer using the retrieved context below. Never speculate or add information not present in the context.

Always respond in this exact JSON format:
{
  "risk_summary": "One sentence describing the risk",
  "owasp_reference": "OWASP category and ID if found, otherwise N/A",
  "remediation": "Concrete steps to fix or mitigate",
  "confidence": "High if context directly answers, Medium if partial, Low if minimal",
  "source": "document name and page if available",
  "insufficient_info": false
}

If the context does not contain enough information, return insufficient_info: true and explain briefly in risk_summary what you could not find. Never fabricate an answer.
"""


def _get_openai_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


def _build_collection():
    api_key = _get_openai_key()
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if api_key:
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=EMBED_MODEL,
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    else:
        # Fallback: default embedding (no OpenAI key)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    return client, collection


_chroma_client, _collection = _build_collection()


def ingest_pdf(file_bytes: bytes, doc_name: str) -> Dict[str, Any]:
    """Parse, chunk, embed and store a PDF. Returns summary."""
    pages = parse_pdf(file_bytes)
    if not pages:
        return {"status": "error", "doc_name": doc_name, "chunks_created": 0, "detail": "No text extracted from PDF"}

    chunks = chunk_pages(pages, source_doc=doc_name)
    if not chunks:
        return {"status": "error", "doc_name": doc_name, "chunks_created": 0, "detail": "No chunks produced"}

    ids = [str(uuid.uuid4()) for _ in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Upsert in batches of 100 to avoid size limits
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        _collection.add(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    return {"status": "ok", "doc_name": doc_name, "chunks_created": len(chunks)}


def query_rag(question: str) -> Dict[str, Any]:
    """Retrieve relevant chunks and call LLM to generate structured answer."""
    api_key = _get_openai_key()
    if not api_key:
        return {
            "risk_summary": "API key not configured. Please set OPENAI_API_KEY in your environment.",
            "owasp_reference": "N/A",
            "remediation": "Set the OPENAI_API_KEY environment variable and restart the server.",
            "confidence": "Low",
            "source": "N/A",
            "insufficient_info": True,
        }

    try:
        results = _collection.query(query_texts=[question], n_results=TOP_K)
    except Exception as e:
        return {
            "risk_summary": f"Retrieval error: {str(e)}",
            "owasp_reference": "N/A",
            "remediation": "Ensure documents have been uploaded before querying.",
            "confidence": "Low",
            "source": "N/A",
            "insufficient_info": True,
        }

    documents: List[str] = results.get("documents", [[]])[0]
    metadatas: List[Dict] = results.get("metadatas", [[]])[0]

    if not documents:
        return {
            "risk_summary": "No relevant information found in the loaded documents.",
            "owasp_reference": "N/A",
            "remediation": "Upload relevant cybersecurity documents and try again.",
            "confidence": "Low",
            "source": "N/A",
            "insufficient_info": True,
        }

    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        source = meta.get("source_doc", "unknown")
        page = meta.get("page_number", "?")
        context_parts.append(f"[Source: {source}, Page {page}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "risk_summary": raw,
            "owasp_reference": "N/A",
            "remediation": "Could not parse structured response.",
            "confidence": "Low",
            "source": "N/A",
            "insufficient_info": False,
        }

    # Ensure all required fields present
    parsed.setdefault("risk_summary", "")
    parsed.setdefault("owasp_reference", "N/A")
    parsed.setdefault("remediation", "")
    parsed.setdefault("confidence", "Low")
    parsed.setdefault("source", "")
    parsed.setdefault("insufficient_info", False)

    return parsed


def list_documents() -> List[str]:
    """Return unique document names stored in the collection."""
    try:
        all_items = _collection.get(include=["metadatas"])
        metadatas = all_items.get("metadatas", [])
        docs = sorted({m.get("source_doc", "unknown") for m in metadatas if m})
        return docs
    except Exception:
        return []
