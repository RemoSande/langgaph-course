from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from database.pgvector_store import AsyncPGVector
import logging
from contextlib import asynccontextmanager
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Base class for database exceptions"""
    pass

class DocumentNotFoundError(DatabaseError):
    """Raised when a document is not found"""
    pass

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

    async def safe_get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        try:
            documents = await self.get_documents_by_ids(ids)
            if not documents:
                raise DocumentNotFoundError(f"No documents found for ids: {ids}")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise DatabaseError(f"Failed to retrieve documents: {str(e)}")

class PGVectorDatabase(Database):
    def __init__(self, connection_string: str, collection_name: str):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.store = None

    async def get_store(self):
        if self.store is None:
            self.store = await AsyncPGVector.create(
                connection_string=self.connection_string,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        return self.store

    async def store_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        docs = [Document(page_content=doc['page_content'], metadata=doc.get('metadata', {})) for doc in documents]
        store = await self.get_store()
        return await store.aadd_documents(docs)

    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        store = await self.get_store()
        results = await store.asimilarity_search_with_score(query, k=k)
        return [{'content': doc.page_content, 'metadata': doc.metadata, 'score': score} for doc, score in results]

    async def get_all_ids(self) -> List[str]:
        store = await self.get_store()
        return await store.get_all_ids()

    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        store = await self.get_store()
        docs = await store.get_documents_by_ids(ids)
        return [{'content': doc.page_content, 'metadata': doc.metadata} for doc in docs]

    async def delete_documents(self, ids: List[str]) -> None:
        store = await self.get_store()
        await store.adelete(ids=ids)

    async def update_document(self, id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        store = await self.get_store()
        await store.adelete(ids=[id])
        doc = Document(page_content=content, metadata=metadata or {})
        await store.aadd_documents([doc])

    async def check_health(self) -> bool:
        try:
            store = await self.get_store()
            await store.get_all_ids()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

# Keep the InMemoryDatabase for testing purposes
class PGVectorDatabase(Database):
    def __init__(self, connection_string: str, collection_name: str):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.store = None

    @asynccontextmanager
    async def get_store(self):
        if self.store is None:
            self.store = await AsyncPGVector.create(
                connection_string=self.connection_string,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        try:
            yield self.store
        finally:
            if self.store:
                await self.store.aclose()

    async def store_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        docs = [Document(page_content=doc['content'], metadata=doc.get('metadata', {})) for doc in documents]
        async with self.get_store() as store:
            return await store.aadd_documents(docs)

    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        async with self.get_store() as store:
            results = await store.asimilarity_search_with_score(query, k=k)
        return [{'content': doc.page_content, 'metadata': doc.metadata, 'score': score} for doc, score in results]

    async def get_all_ids(self) -> List[str]:
        async with self.get_store() as store:
            return await store.get_all_ids()

    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        async with self.get_store() as store:
            docs = await store.get_documents_by_ids(ids)
        return [{'content': doc.page_content, 'metadata': doc.metadata} for doc in docs]

    async def delete_documents(self, ids: List[str]) -> None:
        async with self.get_store() as store:
            await store.delete(ids=ids)

    async def update_document(self, id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        async with self.get_store() as store:
            await store.delete(ids=[id])
            doc = Document(page_content=content, metadata=metadata or {})
            await store.aadd_documents([doc])

    async def check_health(self) -> bool:
        try:
            async with self.get_store() as store:
                await store.get_all_ids()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

    async def create_custom_table(self, table_name: str):
        async with self.get_store() as store:
            async with store._engine.begin() as conn:
                await conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT
                    )
                """))

    async def insert_test_data(self, table_name: str, data: List[Dict[str, Any]]):
        async with self.get_store() as store:
            async with store._engine.begin() as conn:
                for item in data:
                    await conn.execute(text(f"""
                        INSERT INTO {table_name} (name, description)
                        VALUES (:name, :description)
                    """), item)

    async def fetch_all_from_table(self, table_name: str) -> List[Dict[str, Any]]:
        async with self.get_store() as store:
            async with store._engine.begin() as conn:
                result = await conn.execute(text(f"SELECT * FROM {table_name}"))
                return [dict(row) for row in result]

    async def drop_table(self, table_name: str):
        async with self.get_store() as store:
            async with store._engine.begin() as conn:
                await conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))

class InMemoryDatabase(Database):
    def __init__(self):
        self.documents = {}

    async def store_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        # Implement store_documents method
        pass

    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Implement retrieve_documents method
        pass

    async def get_all_ids(self) -> List[str]:
        # Implement get_all_ids method
        pass

    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        # Implement get_documents_by_ids method
        pass

    async def delete_documents(self, ids: List[str]) -> None:
        # Implement delete_documents method
        pass

    async def update_document(self, id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # Implement update_document method
        pass

    async def check_health(self) -> bool:
        # Implement check_health method
        return True
