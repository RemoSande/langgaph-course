from typing import List, Dict, Any
from pydantic import BaseModel, Field
from database.db import Database, InMemoryDatabase

class GraphState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        client_topics: list of client topics
        db: database instance
    """
    
    question: str
    generation: str = ""
    web_search: bool = False
    documents: List[Dict[str, Any]] = []
    client_topics: List[str] = []
    db: Database = Field(default_factory=InMemoryDatabase)
