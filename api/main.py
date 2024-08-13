from fastapi import FastAPI
from api.routes import query, ingestion

app = FastAPI()

app.include_router(query.router)
app.include_router(ingestion.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API"}