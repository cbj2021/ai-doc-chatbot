import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.core.config import settings
from app.core.pdf_processor import extract_text, split_text
from app.core.vector_store import store_chunks, list_documents, delete_document
from app.core.rag import answer_question

logger = logging.getLogger(__name__)
router = APIRouter()

class QuestionRequest(BaseModel):
    doc_id: str
    question: str

class QuestionResponse(BaseModel):
    answer: str
    chunks_used: int
    doc_id: str

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and process it into the vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    doc_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{file.filename}")

    # Save uploaded file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved PDF: {file_path}")

    # Process into vector store
    try:
        text = extract_text(file_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        chunks = split_text(text)
        num_chunks = store_chunks(doc_id, chunks)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_created": num_chunks,
        "message": f"Successfully processed. You can now ask questions using doc_id: {doc_id}",
    }

@router.post("/ask", response_model=QuestionResponse)
def ask_question(body: QuestionRequest):
    """Ask a question about an uploaded document."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = answer_question(body.doc_id, body.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"RAG failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

    return QuestionResponse(
        answer=result["answer"],
        chunks_used=result["chunks_used"],
        doc_id=body.doc_id,
    )

@router.get("/documents")
def get_documents():
    """List all uploaded documents."""
    docs = list_documents()
    return {"documents": docs, "count": len(docs)}

@router.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    """Delete a document from the vector store."""
    try:
        delete_document(doc_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": f"Document {doc_id} deleted successfully"}

@router.get("/health")
def health():
    return {"status": "ok", "model": settings.OLLAMA_MODEL}
