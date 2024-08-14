from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from database.pgvector_store import AsyncPGVector
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    @abstractmethod
    async def update_document(self, id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        pass

class PGVectorDatabase(Database):
    def __init__(self, connection_string: str, collection_name: str):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.store = None

    @asynccontextmanager
    async def get_store(self):
        if self.store is None:
            self.store = await AsyncPgVector.create(
                connection_string=self.connection_string,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        try:
            yield self.store
        except Exception as e:
            logger.error(f"Database operation failed: {str(e)}")
            raise

    async def store_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        try:
            docs = [Document(page_content=doc['content'], metadata=doc.get('metadata', {})) for doc in documents]
            async with self.get_store() as store:
                return await store.aadd_documents(docs)
        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}")
            raise

    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            async with self.get_store() as store:
                results = await store.asimilarity_search_with_score(query, k=k)
            return [{'content': doc.page_content, 'metadata': doc.metadata, 'score': score} for doc, score in results]
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            raise

    async def get_all_ids(self) -> List[str]:
        try:
            async with self.get_store() as store:
                return await store.get_all_ids()
        except Exception as e:
            logger.error(f"Failed to get all IDs: {str(e)}")
            raise

    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        try:
            async with self.get_store() as store:
                docs = await store.get_documents_by_ids(ids)
            return [{'content': doc.page_content, 'metadata': doc.metadata} for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {str(e)}")
            raise

    async def delete_documents(self, ids: List[str]) -> None:
        try:
            async with self.get_store() as store:
                await store.delete(ids=ids)
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise

    async def update_document(self, id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            async with self.get_store() as store:
                await store.delete(ids=[id])
                doc = Document(page_content=content, metadata=metadata or {})
                await store.aadd_documents([doc])
        except Exception as e:
            logger.error(f"Failed to update document: {str(e)}")
            raise

    async def check_health(self) -> bool:
        try:
            async with self.get_store() as store:
                await store.get_all_ids()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

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
