import logging
from langchain_ollama import OllamaLLM
from app.core.config import settings
from app.core.vector_store import search_chunks

logger = logging.getLogger(__name__)

def get_llm():
    return OllamaLLM(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
    )

def answer_question(doc_id: str, question: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Build prompt with context
    3. Generate answer with Llama 3
    """
    # Step 1 - Retrieve
    chunks = search_chunks(doc_id, question)
    if not chunks:
        return {
            "answer": "I could not find relevant information in the document.",
            "sources": [],
            "chunks_used": 0,
        }

    # Step 2 - Build prompt
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a helpful assistant that answers questions based only on the provided document context.
If the answer is not in the context, say "I don't have enough information in the document to answer that."

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    # Step 3 - Generate
    logger.info(f"Sending prompt to {settings.OLLAMA_MODEL}...")
    llm = get_llm()
    answer = llm.invoke(prompt)

    return {
        "answer": answer.strip(),
        "sources": chunks,
        "chunks_used": len(chunks),
    }
