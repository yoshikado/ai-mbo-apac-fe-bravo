import os
import io
import uuid
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
from app.database import get_vector_collection

# Load environment configuration variables
load_dotenv()

app = FastAPI(
    title="Canonical AI Assistant API", 
    description="MVP Backend supporting serverless RAG, URL ingestion, and citation tracking.",
    version="0.1.0"
)

# Initialize Cloudflare Workers AI standard client wrapper
client = OpenAI(
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{os.getenv('CLOUDFLARE_ACCOUNT_ID')}/ai/v1",
    api_key=os.getenv("CLOUDFLARE_API_TOKEN")
)


# ==========================================
# 1. DATA VALIDATION SCHEMAS (PYDANTIC)
# ==========================================

class QueryRequest(BaseModel):
    query: str

class SourceModel(BaseModel):
    component: str
    source: str
    url: Optional[str] = None

class QueryResponse(BaseModel):
    interaction_id: str
    answer: str
    sources: List[SourceModel]

class IngestRequest(BaseModel):
    content: str
    source_name: str       # e.g., "MicroCloud Site Migration Guide"
    component: str         # e.g., "MicroCloud", "Juju", "Ceph"
    version: Optional[str] = "stable"

class URLIngestRequest(BaseModel):
    url: str
    component: str         # e.g., "Ubuntu Pro", "Licensing"
    version: Optional[str] = "stable"

class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int
    collection_size: int


# ==========================================
# 2. HELPER UTILITIES
# ==========================================

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> List[str]:
    """Splits a body of text into structural segments bounded by words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ==========================================
# 3. ROUTE ENDPOINTS
# ==========================================

@app.post("/query", response_model=QueryResponse)
async def query_assistant(payload: QueryRequest):
    """
    Accepts an engineering prompt, extracts matching documents from local disk vector store,
    presents context to Cloudflare serverless GPU, and gives back a response with clean sources.
    """
    interaction_id = str(uuid.uuid4())
    try:
        # 1. Initialize local persistent vector index storage database
        collection = get_vector_collection()
        
        # 2. Retrieve top matching documents 
        results = collection.query(
            query_texts=[payload.query],
            n_results=3
        )
        
        context_chunks = []
        sources = []
        seen_sources = set()
        
        # 3. Process documents and cleanly remove redundant source attributes
        if results and results['documents'] and results['documents'][0]:
            for idx, doc in enumerate(results['documents'][0]):
                context_chunks.append(doc)
                
                meta = results['metadatas'][0][idx] if results['metadatas'] else {}
                src_name = meta.get("source", "Documentation Baseline")
                comp_name = meta.get("component", "Unknown")
                src_url = meta.get("url", None)
                
                # Check uniqueness by grouping component, source string identifier, and the URL
                unique_key = f"{comp_name}::{src_name}::{src_url}"
                
                if unique_key not in seen_sources:
                    seen_sources.add(unique_key)
                    sources.append({
                        "component": comp_name,
                        "source": src_name,
                        "url": src_url
                    })
        
        context_str = "\n---\n".join(context_chunks) if context_chunks else "No relevant context located inside local index."
        
        # 4. Enforce accurate technical system behavioral patterns
        system_prompt = (
            "You are an expert Canonical Field Software Engineer. Answer the user's technical questions "
            "accurately using ONLY the provided documentation context below. If the context does not contain "
            "the answer, rely on precise technical facts about Canonical products (MicroCloud, Juju, LXD, Ceph).\n\n"
            f"DOCUMENTATION CONTEXT:\n{context_str}"
        )
        
        # 5. Execute inference request via verified Cloudflare Mixture-of-Experts architecture
        completion = client.chat.completions.create(
            model="@cf/google/gemma-4-26b-a4b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.query}
            ]
        )
        
        return QueryResponse(
            interaction_id=interaction_id,
            answer=completion.choices[0].message.content,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_raw_text(payload: IngestRequest):
    """Direct text data payload ingestion method."""
    try:
        collection = get_vector_collection()
        chunks = chunk_text(payload.content)
        
        documents, metadatas, ids = [], [], []
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "source": payload.source_name,
                "component": payload.component,
                "version": payload.version,
                "url": None
            })
            ids.append(f"{payload.component}_{uuid.uuid4().hex[:8]}_{idx}")
            
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            
        return IngestResponse(
            status="success", chunks_ingested=len(documents), collection_size=collection.count()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_from_url(payload: URLIngestRequest):
    """
    Downloads file strings from a live URL, handles runtime PDF stream processing
    transparently in memory, builds vector definitions, and registers metadata mapping.
    """
    try:
        # 1. Retrieve raw asset stream using asynchronous client block
        headers = {"User-Agent": "Mozilla/5.0 (Ubuntu; Linux x86_64)"}
        async with httpx.AsyncClient(follow_redirects=True) as web_client:
            response = await web_client.get(payload.url, headers=headers)
            
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to access web resource. Status: {response.status_code}")
            
        content_type = response.headers.get("content-type", "").lower()
        extracted_text = ""
        
        # 2. Check content-type headers or file syntax terminations
        if "application/pdf" in content_type or payload.url.endswith(".pdf"):
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            extracted_text = response.text

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="The targeted web document appears to hold no indexable text payload.")

        # 3. Initialize data pipeline structures
        collection = get_vector_collection()
        chunks = chunk_text(extracted_text)
        
        documents, metadatas, ids = [], [], []
        source_name = payload.url.split("/")[-1] or payload.url
        
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "source": source_name,
                "component": payload.component,
                "version": payload.version,
                "url": payload.url  # Save explicit URL endpoint address directly inside the block
            })
            ids.append(f"{payload.component}_{uuid.uuid4().hex[:8]}_{idx}")
            
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            
        return IngestResponse(
            status="success", chunks_ingested=len(documents), collection_size=collection.count()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))