from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from database.db import Database
from database.pgvector_store import AsyncPGVector
from langchain_openai import OpenAIEmbeddings
import os
from functools import lru_cache

@lru_cache()
def get_database() -> Database:
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL environment variable is not set. Please set it in your environment or .env file.")
    return PGVectorDatabase(connection_string, "rag_collection")

class GraphState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        client_topics: list of client topics
        _db: private database instance
    """
    
    question: str
    generation: str = ""
    web_search: bool = False
    documents: List[Dict[str, Any]] = []
    client_topics: List[str] = []
    _db: Optional[Database] = None

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db
