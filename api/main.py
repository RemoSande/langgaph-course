import os
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from langchain.schema import Document
from typing import List
from api.models import DocumentModel, DocumentResponse
from graph.state import get_database, GraphState
from database.db import Database, PGVectorDatabase
from typing import List, Optional
from pydantic import BaseModel
from config.settings import settings
from ingestion.pdf_parser import parse_pdf
from ingestion.s3_handler import download_from_s3, list_s3_files

class SearchQuery(BaseModel):
    query: str
    k: int = 5

class UpdateDocument(BaseModel):
    content: str
    metadata: Optional[dict] = None

class TextDocument(BaseModel):
    content: str
    metadata: dict = {}

class S3BucketInfo(BaseModel):
    bucket_name: str
    folder_path: str

from graph.graph import app as graph_app

load_dotenv(find_dotenv())

app = FastAPI()
db = PGVectorDatabase(settings.DATABASE_URL, "documents")

@app.get("/")
async def root():
    return {"message": "Welcome to the LangChain FastAPI application!"}

@app.get("/welcome")
async def welcome():
    return {"message": "Welcome to our enhanced RAG system with LangGraph!"}

# Retrieves all env variables and raises an error if not found
def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found.")
    return value

# Load environment variables
load_dotenv()

# Initialize database
db = get_database()

@app.post("/documents/")
async def add_documents(documents: list[DocumentModel]):
    try:
        docs = [
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "digest": doc.generate_digest()
                } if doc.metadata else {"digest": doc.generate_digest()}
            )
            for doc in documents
        ]
        ids = await db.store_documents(docs)
        return {"message": "Documents added successfully", "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/ids")
async def get_all_ids():
    try:
        ids = await db.get_all_ids()
        return ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/", response_model=list[DocumentResponse])
async def get_documents_by_ids(ids: list[str]):
    try:
        documents = await db.get_documents_by_ids(ids)
        if not documents:
            raise HTTPException(status_code=404, detail="One or more IDs not found")
        return [DocumentResponse(page_content=doc['content'], metadata=doc['metadata']) for doc in documents]
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/")
async def delete_documents(ids: list[str]):
    try:
        await db.delete_documents(ids)
        return {"message": f"{len(ids)} documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def quick_response(query: SearchQuery):
    state = GraphState(question=query.query, client_topics=[])
    result = await graph_app.ainvoke(state)
    return {"response": result.generation}

@app.get("/health")
async def health_check():
    try:
        is_healthy = await db.check_health()
        if is_healthy:
            return {"status": "healthy", "database": "connected"}
        else:
            raise HTTPException(status_code=500, detail="Database health check failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/db-test")
async def db_test():
    try:
        # Test document insertion
        test_doc = {"content": "Test document", "metadata": {"test": True}}
        doc_id = await db.store_documents([test_doc])
        
        # Test document retrieval
        retrieved_doc = await db.get_documents_by_ids(doc_id)
        
        # Test document deletion
        await db.delete_documents(doc_id)
        
        return {"status": "success", "message": "Database operations successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database test failed: {str(e)}")

@app.get("/search/")
async def search_documents(query: str, k: int = 5):
    try:
        results = await db.retrieve_documents(query, k=k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/documents/{doc_id}")
async def update_document(doc_id: str, update: UpdateDocument):
    try:
        await db.update_document(doc_id, update.content, update.metadata)
        return {"message": "Document updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/text")
async def ingest_text(document: TextDocument):
    try:
        doc_ids = await db.store_documents([{"content": document.content, "metadata": document.metadata}])
        return {"message": "Document ingested successfully", "id": doc_ids[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/s3_pdf")
async def ingest_s3_pdf(s3_info: S3BucketInfo):
    try:
        pdf_files = list_s3_files(s3_info.bucket_name, s3_info.folder_path)
        ingested_docs = []
        for pdf_file in pdf_files:
            local_path = download_from_s3(s3_info.bucket_name, pdf_file)
            content = parse_pdf(local_path)
            doc_ids = await db.store_documents([{"content": content, "metadata": {"source": pdf_file}}])
            ingested_docs.append({"file": pdf_file, "id": doc_ids[0]})
        return {"message": "PDF documents ingested successfully", "ingested_documents": ingested_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
