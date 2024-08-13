from typing import List, Dict, Any
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from database.pgvector_store import AsyncPgVector

class Database(ABC):
    @abstractmethod
    async def store_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        pass

    @abstractmethod
    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_all_ids(self) -> List[str]:
        pass

    @abstractmethod
    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_documents(self, ids: List[str]) -> None:
        pass

class PGVectorDatabase(Database):
    def __init__(self, connection_string: str, collection_name: str):
        self.embeddings = OpenAIEmbeddings()
        self.store = AsyncPgVector(
            connection_string=connection_string,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

    async def store_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        docs = [Document(page_content=doc['content'], metadata=doc.get('metadata', {})) for doc in documents]
        return await self.store.aadd_documents(docs)

    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        results = await self.store.asimilarity_search_with_score(query, k=k)
        return [{'content': doc.page_content, 'metadata': doc.metadata, 'score': score} for doc, score in results]

    async def get_all_ids(self) -> List[str]:
        return await self.store.get_all_ids()

    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        docs = await self.store.get_documents_by_ids(ids)
        return [{'content': doc.page_content, 'metadata': doc.metadata} for doc in docs]

    async def delete_documents(self, ids: List[str]) -> None:
        await self.store.delete(ids=ids)

# Keep the InMemoryDatabase for testing purposes
class InMemoryDatabase(Database):
    def __init__(self):
        self.documents = []

    async def store_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        ids = [f"doc_{i}" for i in range(len(self.documents), len(self.documents) + len(documents))]
        self.documents.extend(zip(ids, documents))
        return ids

    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return [doc for _, doc in self.documents[:k]]

    async def get_all_ids(self) -> List[str]:
        return [id for id, _ in self.documents]

    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        return [doc for id, doc in self.documents if id in ids]

    async def delete_documents(self, ids: List[str]) -> None:
        self.documents = [(id, doc) for id, doc in self.documents if id not in ids]
