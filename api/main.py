import os
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from models import DocumentModel, DocumentResponse
from database.db import PGVectorDatabase
from graph.state import get_database

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

try:
    # Get environment variables
    OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
    CONNECTION_STRING = get_env_variable("DATABASE_URL")

    embeddings = OpenAIEmbeddings()
    db = get_database()
    retriever = db.store.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

except ValueError as e:
    raise HTTPException(status_code=500, detail=str(e))
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

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
async def quick_response(msg: str):
    result = await chain.ainvoke(msg)
    return result

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
