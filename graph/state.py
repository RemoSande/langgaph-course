from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel


class GraphState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        client_topics: list of client topics
    """
    
    question: str
    generation: str = ""
    web_search: bool = False
    documents: List[Dict[str, Any]] = []
    client_topics: List[str] = []
