"""Document management and RAG (Retrieval-Augmented Generation) system with Redis"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import uuid
from datetime import datetime
import redis.asyncio as redis
from enum import Enum

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
document_sessions = {}

class DocumentType(str, Enum):
    """Types of documents that can be created"""
    RUNBOOK = "runbook"
    TROUBLESHOOTING = "troubleshooting"
    ARCHITECTURE = "architecture"
    INCIDENT_REPORT = "incident_report"
    SOP = "sop"  # Standard Operating Procedure

class CreateDocumentRequest(BaseModel):
    """Request to create a new document"""
    document_type: DocumentType
    title: str = Field(..., min_length=3, max_length=200)
    initial_prompt: str = Field(..., min_length=10)
    provider: str = Field(default="gemini")
    doc_ids: Optional[List[str]] = Field(default=None, description="Specific documents to reference")

class ClarificationRequest(BaseModel):
    """Request for clarification during document creation"""
    session_id: str
    answer: str
    provider: str = Field(default="gemini")

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
@router.post("/create-document/start")
async def start_document_creation(request: CreateDocumentRequest):
    """
    Start an interactive document creation session.
    The AI will ask clarifying questions before generating the document.
    """
    documents = await document_store.list_documents()
    
    if not documents:
        raise HTTPException(
            status_code=400, 
            detail="No knowledge base documents available. Please upload documents first."
        )
    
    # Create session ID
    session_id = str(uuid.uuid4())[:12]
    
    # Get relevant context from KB
    search_docs = request.doc_ids if request.doc_ids else None
    relevant_chunks = await document_store.search_chunks(
        request.initial_prompt, 
        search_docs, 
        top_k=10
    )
    
    # Build context
    context_parts = []
    sources = []
    for chunk in relevant_chunks:
        context_parts.append(f"[{chunk['filename']}]:\n{chunk['content']}")
        source_info = {"filename": chunk["filename"], "doc_id": chunk["doc_id"]}
        if source_info not in sources:
            sources.append(source_info)
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Create clarification prompt
    clarification_prompt = f"""You are an expert SRE/DevOps documentation assistant. A user wants to create a {request.document_type.value} document.

KNOWLEDGE BASE CONTEXT:
{context}

USER'S REQUEST:
{request.title}
{request.initial_prompt}

YOUR TASK:
Based ONLY on the knowledge base context provided above, ask 3-5 specific clarifying questions to gather the information needed to create a comprehensive {request.document_type.value} document.

CRITICAL RULES:
1. ONLY ask questions about information that exists in the knowledge base context
2. If the knowledge base lacks necessary information, explicitly state what's missing
3. DO NOT assume or invent information
4. Ask specific, technical questions relevant to {request.document_type.value}
5. Focus on gaps that need to be filled from the existing knowledge base

Format your response as:
ANALYSIS: [Brief assessment of available information]
QUESTIONS:
1. [Question 1]
2. [Question 2]
3. [Question 3]
[etc.]

MISSING: [List any critical information not found in the knowledge base]"""

    try:
        from fastapi_app.llm import PROVIDERS
        
        if request.provider not in PROVIDERS:
            available = ", ".join(PROVIDERS.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Provider '{request.provider}' not available. Choose from: {available}"
            )
        
        llm = PROVIDERS[request.provider]
        
        if not llm.is_configured():
            raise HTTPException(
                status_code=503, 
                detail=f"Provider '{request.provider}' not configured"
            )
        
        response = await llm.generate(prompt=clarification_prompt, temperature=0.3)
        clarification_response = response.content if hasattr(response, 'content') else str(response)
        
        # Store session
        document_sessions[session_id] = {
            "document_type": request.document_type,
            "title": request.title,
            "initial_prompt": request.initial_prompt,
            "context": context,
            "sources": sources,
            "clarifications": [],
            "provider": request.provider,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "session_id": session_id,
            "status": "awaiting_clarification",
            "questions": clarification_response,
            "sources_available": sources,
            "message": "Please answer the clarifying questions to proceed with document creation."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting document creation: {str(e)}")


@router.post("/create-document/clarify")
async def provide_clarification(request: ClarificationRequest):
    """
    Provide answers to clarifying questions.
    The AI will determine if more questions are needed or if it can generate the document.
    """
    if request.session_id not in document_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session = document_sessions[request.session_id]
    session["clarifications"].append(request.answer)
    
    # Ask AI if we have enough information
    verification_prompt = f"""You are an expert SRE/DevOps documentation assistant.

DOCUMENT TYPE: {session['document_type'].value}
TITLE: {session['title']}
ORIGINAL REQUEST: {session['initial_prompt']}

KNOWLEDGE BASE CONTEXT:
{session['context']}

USER'S CLARIFICATIONS SO FAR:
{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(session['clarifications']))}

YOUR TASK:
Determine if you have enough information from the knowledge base and clarifications to create a complete {session['document_type'].value} document.

RESPOND IN THIS FORMAT:
STATUS: [READY or NEED_MORE]
REASONING: [Explain what you have or what's missing]
ADDITIONAL_QUESTIONS: [If NEED_MORE, list specific questions. If READY, write "None"]
MISSING_FROM_KB: [List any critical information not in the knowledge base]"""

    try:
        from fastapi_app.llm import PROVIDERS
        
        llm = PROVIDERS[request.provider]
        response = await llm.generate(prompt=verification_prompt, temperature=0.3)
        verification_response = response.content if hasattr(response, 'content') else str(response)
        
        # Parse response to determine status
        if "STATUS: READY" in verification_response:
            return {
                "session_id": request.session_id,
                "status": "ready_to_generate",
                "analysis": verification_response,
                "message": "Ready to generate document. Call /create-document/generate to create the final document."
            }
        else:
            return {
                "session_id": request.session_id,
                "status": "need_more_info",
                "questions": verification_response,
                "message": "Please provide additional information."
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing clarification: {str(e)}")


@router.post("/create-document/generate")
async def generate_document(session_id: str, provider: str = "gemini"):
    """
    Generate the final document based on the knowledge base and clarifications.
    This will ONLY use information from the KB - no hallucinations.
    """
    if session_id not in document_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session = document_sessions[session_id]
    
    # Create document generation prompt
    document_prompt = f"""You are an expert SRE/DevOps documentation assistant. Create a comprehensive {session['document_type'].value} document.

DOCUMENT TITLE: {session['title']}

ORIGINAL REQUEST: {session['initial_prompt']}

KNOWLEDGE BASE CONTEXT (YOUR ONLY SOURCE OF TRUTH):
{session['context']}

USER'S CLARIFICATIONS:
{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(session['clarifications']))}

YOUR TASK:
Create a professional {session['document_type'].value} document using ONLY information from the knowledge base context above.

CRITICAL RULES:
1. USE ONLY factual information from the knowledge base - NO assumptions or inventions
2. If information is missing, explicitly state "Information not available in knowledge base"
3. Cite which documents information comes from using [Source: filename]
4. Be specific and technical - this is for professional SREs/DevOps engineers
5. Include clear sections appropriate for a {session['document_type'].value}
6. DO NOT hallucinate or make up information

FORMAT FOR {session['document_type'].value.upper()}:
{get_document_template(session['document_type'])}

Generate the document now, following the format above."""

    try:
        from fastapi_app.llm import PROVIDERS
        
        llm = PROVIDERS[provider]
        response = await llm.generate(prompt=document_prompt, temperature=0.2, max_tokens=2000)
        generated_doc = response.content if hasattr(response, 'content') else str(response)
        
        # Store generated document in KB
        final_doc = f"""# {session['title']}

**Document Type:** {session['document_type'].value}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sources:** {', '.join(s['filename'] for s in session['sources'])}

---

{generated_doc}

---

**Knowledge Base Sources Used:**
{chr(10).join(f"- {s['filename']}" for s in session['sources'])}
"""
        
        # Save to document store
        result = await document_store.store_document(
            f"{session['title']}.txt",
            final_doc
        )
        
        # Clean up session
        del document_sessions[session_id]
        
        return {
            "status": "completed",
            "document": generated_doc,
            "doc_id": result["doc_id"],
            "sources": session['sources'],
            "message": "Document created successfully and added to knowledge base."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating document: {str(e)}")


def get_document_template(doc_type: DocumentType) -> str:
    """Returns the appropriate template structure for each document type"""
    templates = {
        DocumentType.RUNBOOK: """
# Title
## Overview
## Prerequisites
## Step-by-Step Procedure
## Rollback Procedure
## Validation Steps
## Troubleshooting
## References""",
        
        DocumentType.TROUBLESHOOTING: """
# Title
## Problem Description
## Symptoms
## Root Cause Analysis
## Resolution Steps
## Prevention
## Related Issues
## References""",
        
        DocumentType.ARCHITECTURE: """
# Title
## Overview
## System Components
## Data Flow
## Infrastructure Details
## Security Considerations
## Scalability & Performance
## Monitoring & Observability
## References""",
        
        DocumentType.INCIDENT_REPORT: """
# Title
## Incident Summary
## Timeline
## Impact
## Root Cause
## Resolution
## Action Items
## Lessons Learned
## References""",
        
        DocumentType.SOP: """
# Title
## Purpose
## Scope
## Responsibilities
## Procedure
## Frequency
## Documentation Requirements
## Review & Updates
## References"""
    }
    return templates.get(doc_type, "")


@router.delete("/create-document/session/{session_id}")
async def cancel_document_session(session_id: str):
    """Cancel an ongoing document creation session"""
    if session_id in document_sessions:
        del document_sessions[session_id]
        return {"message": "Session cancelled successfully"}
    raise HTTPException(status_code=404, detail="Session not found")