from fastapi import APIRouter, HTTPException
from typing import List
from app.models.document import DocumentModel, DocumentResponse
from app.services.pgvector_service import get_pgvector_service
from app.core.config import settings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

router = APIRouter()

embeddings = OpenAIEmbeddings()

pgvector_service = get_pgvector_service(
    connection_string=settings.DATABASE_URL,
    collection_name="testcollection",
    embedding_function=embeddings,
    use_async=settings.USE_ASYNC
)

@router.post("/", response_model=dict)
async def add_documents(documents: List[DocumentModel]):
    try:
        docs = [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "digest": doc.generate_digest()}
            )
            for doc in documents
        ]
        if settings.USE_ASYNC:
            result = await pgvector_service.aadd_documents(docs)
        else:
            result = pgvector_service.add_documents(docs)
        return {"message": "Documents added successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ids", response_model=List[str])
async def get_all_ids():
    try:
        if settings.USE_ASYNC:
            ids = await pgvector_service.aget_all_ids()
        else:
            ids = pgvector_service.get_all_ids()
        return ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/by-ids", response_model=List[DocumentResponse])
async def get_documents_by_ids(ids: List[str]):
    try:
        if settings.USE_ASYNC:
            documents = await pgvector_service.aget_documents_by_ids(ids)
        else:
            documents = pgvector_service.get_documents_by_ids(ids)
        return [DocumentResponse(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/")
async def delete_documents(ids: List[str]):
    try:
        if settings.USE_ASYNC:
            await pgvector_service.adelete(ids=ids)
        else:
            pgvector_service.delete(ids=ids)
        return {"message": f"{len(ids)} documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))