from typing import List, Dict, Any
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    async def store_documents(self, documents: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        pass

class InMemoryDatabase(Database):
    def __init__(self):
        self.documents = []

    async def store_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.documents.extend(documents)

    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # This is a simple implementation. In a real-world scenario,
        # you'd want to implement proper vector search here.
        return self.documents[:k]

# You can add more database implementations here in the future
