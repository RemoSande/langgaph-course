import os
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from langchain.schema import Document
from typing import List
from models import DocumentModel, DocumentResponse
from database.db import PGVectorDatabase
from graph.state import get_database, GraphState
from graph.graph import app as graph_app

load_dotenv(find_dotenv())

app = FastAPI()

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
                metadata=(
                    {**doc.metadata, "digest": doc.generate_digest()}
                    if doc.metadata
                    else {"digest": doc.generate_digest()}
                ),
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
        return documents
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
async def quick_response(msg: str, client_topics: List[str] = []):
    state = GraphState(question=msg, client_topics=client_topics)
    result = await graph_app.ainvoke(state)
    return {"response": result.generation}

@app.get("/health")
async def health_check():
    try:
        is_healthy = await db.check_health()
        if is_healthy:
            return {"status": "healthy"}
        else:
            raise HTTPException(status_code=500, detail="Database health check failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
