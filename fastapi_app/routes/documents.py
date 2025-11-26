"""Document management and RAG (Retrieval-Augmented Generation) system with Redis"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import uuid
from datetime import datetime
import redis.asyncio as redis

from fastapi_app.config import settings
from fastapi_app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


class RedisDocumentStore:
    """Redis-based persistent document storage"""
    
    def __init__(self):
        self.redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379')
        self.redis_client: Optional[redis.Redis] = None
        self.doc_prefix = "doc:"
        self.chunk_prefix = "chunk:"
        self.doc_list_key = "documents:list"
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks
    
    async def store_document(self, filename: str, content: str):
        """Store document and its chunks in Redis"""
        if not self.redis_client:
            raise HTTPException(status_code=503, detail="Redis not connected")
        
        doc_id = str(uuid.uuid4())[:8]
        chunks = self.chunk_text(content)
        
        doc_data = {
            "id": doc_id,
            "filename": filename,
            "content": content,
            "size": len(content),
            "chunks_count": len(chunks),
            "uploaded_at": datetime.now().isoformat()
        }
        
        await self.redis_client.set(
            f"{self.doc_prefix}{doc_id}",
            json.dumps(doc_data),
            ex=86400 * 30  # 30 days
        )
        
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "content": chunk
            }
            await self.redis_client.set(
                f"{self.chunk_prefix}{doc_id}:{i}",
                json.dumps(chunk_data),
                ex=86400 * 30
            )
        
        await self.redis_client.sadd(self.doc_list_key, doc_id)
        logger.info(f"Stored document {doc_id} with {len(chunks)} chunks")
        
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "size": len(content)
        }
    
    async def get_document(self, doc_id: str):
        """Retrieve document by ID"""
        if not self.redis_client:
            raise HTTPException(status_code=503, detail="Redis not connected")
        
        doc_json = await self.redis_client.get(f"{self.doc_prefix}{doc_id}")
        return json.loads(doc_json) if doc_json else None
    
    async def list_documents(self):
        """List all documents"""
        if not self.redis_client:
            raise HTTPException(status_code=503, detail="Redis not connected")
        
        doc_ids = await self.redis_client.smembers(self.doc_list_key)
        documents = []
        
        for doc_id in doc_ids:
            doc_data = await self.get_document(doc_id)
            if doc_data:
                documents.append({
                    "id": doc_data["id"],
                    "filename": doc_data["filename"],
                    "size": doc_data["size"],
                    "chunks": doc_data["chunks_count"],
                    "uploaded_at": doc_data["uploaded_at"],
                    "preview": doc_data["content"][:200] + "..." if len(doc_data["content"]) > 200 else doc_data["content"]
                })
        
        return documents
    
    async def delete_document(self, doc_id: str):
        """Delete document and its chunks"""
        if not self.redis_client:
            raise HTTPException(status_code=503, detail="Redis not connected")
        
        doc_data = await self.get_document(doc_id)
        if not doc_data:
            return False
        
        await self.redis_client.delete(f"{self.doc_prefix}{doc_id}")
        
        chunks_count = doc_data.get("chunks_count", 0)
        for i in range(chunks_count):
            await self.redis_client.delete(f"{self.chunk_prefix}{doc_id}:{i}")
        
        await self.redis_client.srem(self.doc_list_key, doc_id)
        logger.info(f"Deleted document {doc_id}")
        return True
    
    async def search_chunks(self, query: str, doc_ids: Optional[List[str]] = None, top_k: int = 5):
        """Search for relevant chunks"""
        if not self.redis_client:
            raise HTTPException(status_code=503, detail="Redis not connected")
        
        search_ids = doc_ids if doc_ids else list(await self.redis_client.smembers(self.doc_list_key))
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for doc_id in search_ids:
            doc_data = await self.get_document(doc_id)
            if not doc_data:
                continue
                
            chunks_count = doc_data.get("chunks_count", 0)
            
            for i in range(chunks_count):
                chunk_json = await self.redis_client.get(f"{self.chunk_prefix}{doc_id}:{i}")
                if not chunk_json:
                    continue
                
                chunk_data = json.loads(chunk_json)
                chunk_lower = chunk_data["content"].lower()
                
                chunk_words = set(chunk_lower.split())
                overlap = len(query_words & chunk_words)
                
                if query.lower() in chunk_lower:
                    overlap += 5
                
                if overlap > 0:
                    scored_chunks.append({**chunk_data, "score": overlap})
        
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]


# Global instance
document_store = RedisDocumentStore()


class QueryRequest(BaseModel):
    """Request model for document Q&A"""
    question: str = Field(..., description="Question to ask")
    provider: str = Field(default="gemini", description="LLM provider")
    doc_ids: Optional[List[str]] = Field(default=None, description="Document IDs")
    max_context_length: int = Field(default=4000, description="Max context length")


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload document to Redis knowledge base"""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported")
    
    content = await file.read()
    text_content = content.decode("utf-8")
    
    result = await document_store.store_document(file.filename, text_content)
    
    return JSONResponse(content={
        "message": f"{file.filename} uploaded successfully",
        **result
    })


@router.get("/")
async def list_documents():
    """List all documents from Redis"""
    documents = await document_store.list_documents()
    return {
        "documents": documents,
        "total": len(documents)
    }


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    """Get document by ID from Redis"""
    doc_data = await document_store.get_document(doc_id)
    
    if not doc_data:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    
    return {
        "id": doc_data["id"],
        "filename": doc_data["filename"],
        "content": doc_data["content"],
        "size": doc_data["size"],
        "chunks": doc_data["chunks_count"],
        "uploaded_at": doc_data["uploaded_at"]
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document from Redis"""
    doc_data = await document_store.get_document(doc_id)
    
    if not doc_data:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    
    filename = doc_data["filename"]
    await document_store.delete_document(doc_id)
    
    return {"message": f"Document '{filename}' deleted successfully"}


@router.post("/query")
async def query_documents(request: QueryRequest):
    """Query documents using RAG"""
    documents = await document_store.list_documents()
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    
    search_docs = request.doc_ids if request.doc_ids else None
    relevant_chunks = await document_store.search_chunks(request.question, search_docs, top_k=5)
    
    if not relevant_chunks:
        return JSONResponse(content={
            "answer": "No relevant information found in documents",
            "sources": [],
            "context_used": ""
        })
    
    context_parts = []
    sources = []
    total_length = 0
    
    for chunk in relevant_chunks:
        if total_length + len(chunk["content"]) > request.max_context_length:
            break
        context_parts.append(f"[From {chunk['filename']}]:\n{chunk['content']}")
        total_length += len(chunk["content"])
        
        source_info = {"filename": chunk["filename"], "doc_id": chunk["doc_id"]}
        if source_info not in sources:
            sources.append(source_info)
    
    context = "\n\n".join(context_parts)
    
    rag_prompt = f"""Based on the following context from uploaded documents, answer the question.

CONTEXT:
{context}

QUESTION: {request.question}

Provide a clear answer based only on the context above."""

    try:
        from fastapi_app.llm import PROVIDERS
        
        if request.provider not in PROVIDERS:
            available = ", ".join(PROVIDERS.keys())
            raise HTTPException(status_code=400, detail=f"Provider '{request.provider}' not available. Choose from: {available}")
        
        llm = PROVIDERS[request.provider]
        
        if not llm.is_configured():
            raise HTTPException(status_code=503, detail=f"Provider '{request.provider}' not configured")
        
        response = await llm.generate(prompt=rag_prompt, temperature=0.3)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(context_parts),
            "provider": request.provider
        }
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "context": context,
            "sources": sources,
            "error": str(e)
        }


@router.get("/search/{query}")
async def search_documents(
    query: str,
    top_k: int = Query(default=5, ge=1, le=20)
):
    """Search documents"""
    documents = await document_store.list_documents()
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    
    results = await document_store.search_chunks(query, top_k=top_k)
    
    return {
        "query": query,
        "results": results,
        "total_found": len(results)
    }