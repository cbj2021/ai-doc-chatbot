import os
import logging
import chromadb
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

def get_embeddings():
    return OllamaEmbeddings(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )

def get_chroma_client():
    return chromadb.PersistentClient(path=settings.CHROMA_DIR)

def store_chunks(doc_id: str, chunks: list[str]) -> int:
    """Embed chunks and store in ChromaDB."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=f"doc_{doc_id}")
    embeddings = get_embeddings()

    logger.info(f"Embedding {len(chunks)} chunks for doc {doc_id}...")
    vectors = embeddings.embed_documents(chunks)

    collection.add(
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
    )
    logger.info(f"Stored {len(chunks)} chunks in ChromaDB")
    return len(chunks)

def search_chunks(doc_id: str, query: str, top_k: int = None) -> list[str]:
    """Find most relevant chunks for a query."""
    top_k = top_k or settings.TOP_K_RESULTS
    client = get_chroma_client()

    try:
        collection = client.get_collection(name=f"doc_{doc_id}")
    except Exception:
        raise ValueError(f"Document {doc_id} not found. Please upload it first.")

    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
    )
    chunks = results["documents"][0] if results["documents"] else []
    logger.info(f"Retrieved {len(chunks)} relevant chunks for query")
    return chunks

def list_documents() -> list[str]:
    """List all document IDs stored in ChromaDB."""
    client = get_chroma_client()
    collections = client.list_collections()
    return [c.name.replace("doc_", "") for c in collections]

def delete_document(doc_id: str):
    """Delete a document and all its chunks."""
    client = get_chroma_client()
    client.delete_collection(name=f"doc_{doc_id}")
    logger.info(f"Deleted document {doc_id}")
