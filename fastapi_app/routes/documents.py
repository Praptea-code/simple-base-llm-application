"""Document management and RAG (Retrieval-Augmented Generation) system"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

router = APIRouter()

# Document store: {doc_id: {filename, content, uploaded_at, chunks}}
DOCUMENTS = {}


class QueryRequest(BaseModel):
    """Request model for document Q&A"""
    question: str = Field(..., description="Question to ask about the documents")
    provider: str = Field(default="gemini", description="LLM provider to use")
    doc_ids: Optional[List[str]] = Field(default=None, description="Specific document IDs to search (None = all)")
    max_context_length: int = Field(default=4000, description="Maximum context length to send to LLM")


class DocumentInfo(BaseModel):
    """Document metadata"""
    id: str
    filename: str
    size: int
    uploaded_at: str
    preview: str


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better retrieval"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def simple_search(query: str, documents: dict, top_k: int = 3) -> List[dict]:
    """
    Simple relevance search based on keyword matching.
    Returns top_k most relevant chunks.
    
    For production, consider using:
    - Sentence embeddings (sentence-transformers)
    - Vector database (ChromaDB, Pinecone, etc.)
    """
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for doc_id, doc_data in documents.items():
        for i, chunk in enumerate(doc_data.get("chunks", [])):
            chunk_lower = chunk.lower()
            # Score based on word overlap
            chunk_words = set(chunk_lower.split())
            overlap = len(query_words & chunk_words)
            
            # Bonus for exact phrase match
            if query.lower() in chunk_lower:
                overlap += 5
            
            if overlap > 0:
                scored_chunks.append({
                    "doc_id": doc_id,
                    "filename": doc_data["filename"],
                    "chunk_index": i,
                    "content": chunk,
                    "score": overlap
                })
    
    # Sort by score and return top_k
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to be used for Q&A.
    Supports .txt files.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are currently supported")
    
    content = await file.read()
    text_content = content.decode("utf-8")
    
    # Generate unique ID
    doc_id = str(uuid.uuid4())[:8]
    
    # Chunk the document for better retrieval
    chunks = chunk_text(text_content)
    
    DOCUMENTS[doc_id] = {
        "filename": file.filename,
        "content": text_content,
        "chunks": chunks,
        "size": len(content),
        "uploaded_at": datetime.now().isoformat()
    }
    
    return JSONResponse(content={
        "message": f"{file.filename} uploaded successfully",
        "doc_id": doc_id,
        "chunks": len(chunks),
        "size": len(content)
    })


@router.get("/")
async def list_documents():
    """
    List all uploaded documents with metadata.
    """
    documents = []
    for doc_id, doc_data in DOCUMENTS.items():
        documents.append({
            "id": doc_id,
            "filename": doc_data["filename"],
            "size": doc_data["size"],
            "chunks": len(doc_data.get("chunks", [])),
            "uploaded_at": doc_data["uploaded_at"],
            "preview": doc_data["content"][:200] + "..." if len(doc_data["content"]) > 200 else doc_data["content"]
        })
    
    return {
        "documents": documents,
        "total": len(documents)
    }


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    """
    Get a specific document by ID.
    """
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    
    doc_data = DOCUMENTS[doc_id]
    return {
        "id": doc_id,
        "filename": doc_data["filename"],
        "content": doc_data["content"],
        "size": doc_data["size"],
        "chunks": len(doc_data.get("chunks", [])),
        "uploaded_at": doc_data["uploaded_at"]
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document by ID.
    """
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    
    filename = DOCUMENTS[doc_id]["filename"]
    del DOCUMENTS[doc_id]
    
    return {"message": f"Document '{filename}' deleted successfully"}


@router.post("/query")
async def query_documents(request: QueryRequest):
    """
    Ask a question and get an AI-generated answer based on uploaded documents.
    
    This implements a simple RAG (Retrieval-Augmented Generation) pipeline:
    1. Search for relevant document chunks
    2. Build context from retrieved chunks
    3. Send question + context to LLM
    4. Return AI-generated answer with sources
    """
    if not DOCUMENTS:
        raise HTTPException(status_code=400, detail="No documents uploaded. Please upload documents first.")
    
    # Filter documents if specific IDs provided
    search_docs = DOCUMENTS
    if request.doc_ids:
        search_docs = {k: v for k, v in DOCUMENTS.items() if k in request.doc_ids}
        if not search_docs:
            raise HTTPException(status_code=404, detail="None of the specified documents were found")
    
    # Step 1: Retrieve relevant chunks
    relevant_chunks = simple_search(request.question, search_docs, top_k=5)
    
    if not relevant_chunks:
        return JSONResponse(content={
            "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
            "sources": [],
            "context_used": ""
        })
    
    # Step 2: Build context from chunks
    context_parts = []
    sources = []
    total_length = 0
    
    for chunk in relevant_chunks:
        if total_length + len(chunk["content"]) > request.max_context_length:
            break
        context_parts.append(f"[From {chunk['filename']}]:\n{chunk['content']}")
        total_length += len(chunk["content"])
        
        # Track sources
        source_info = {"filename": chunk["filename"], "doc_id": chunk["doc_id"]}
        if source_info not in sources:
            sources.append(source_info)
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Build prompt for LLM
    rag_prompt = f"""Based on the following context from uploaded documents, please answer the question. 
If the answer cannot be found in the context, say so clearly.

CONTEXT:
{context}

QUESTION: {request.question}

Please provide a clear, concise answer based only on the information in the context above."""

    # Step 4: Get answer from LLM
    try:
        from ..llm import PROVIDERS
        
        if request.provider not in PROVIDERS:
            available = ", ".join(PROVIDERS.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Provider '{request.provider}' not available. Choose from: {available}"
            )
        
        llm = PROVIDERS[request.provider]
        
        # Check if provider is configured
        if not llm.is_configured():
            raise HTTPException(
                status_code=503,
                detail=f"Provider '{request.provider}' is not configured. Please set the API key."
            )
        
        response = await llm.generate(prompt=rag_prompt, temperature=0.3)
        
        # Handle response (could be LLMResponse object or string)
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(context_parts),
            "provider": request.provider
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Fallback: return context without LLM answer
        return {
            "answer": f"Error generating AI answer: {str(e)}. Here's the relevant context I found:",
            "context": context,
            "sources": sources,
            "error": str(e)
        }


@router.get("/search/{query}")
async def search_documents(
    query: str,
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return")
):
    """
    Search documents for relevant chunks without generating an AI answer.
    Useful for previewing what context would be used.
    """
    if not DOCUMENTS:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    
    results = simple_search(query, DOCUMENTS, top_k=top_k)
    
    return {
        "query": query,
        "results": results,
        "total_found": len(results)
    }