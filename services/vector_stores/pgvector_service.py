from typing import List, Dict, Optional, Any
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain.indexes import SQLRecordManager, index
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from langchain_core.runnables.config import run_in_executor
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

class ExtendedPGVector(PGVector):
    """
    Extended PGVector class with additional functionality for document management.
    """
    def __init__(self, connection_string: str, collection_name: str, embedding_function: Optional[Embeddings] = None, *args, **kwargs):
        """
        Initialize ExtendedPGVector with a database connection, embedding function 
        and collection name.
        """
        embedding_function = embedding_function or OpenAIEmbeddings()
        super().__init__(connection_string=connection_string, collection_name=collection_name, embedding_function=embedding_function, *args, **kwargs)
        self.engine = create_engine(connection_string)
        self.record_manager = SQLRecordManager(f"pgvector/{collection_name}", db_url=connection_string)
        self.record_manager.create_schema()

    def get_all_ids(self) -> List[str]:
        """Retrieve all custom IDs from the embedding store."""
        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore.custom_id).all()
            return [result[0] for result in results if result[0] is not None]

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """Retrieve documents by their custom IDs."""
        with Session(self._bind) as session:
            results = (
                session.query(self.EmbeddingStore)
                .filter(self.EmbeddingStore.custom_id.in_(ids))
                .all()
            )
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results
                if result.custom_id in ids
            ]

    def update_collection(self, docs: List[Document], cleanup_mode: str = "incremental") -> Dict[str, int]:
        """
        Update the collection with new documents.
        
        :param cleanup_mode: "incremental" (default), "full", or "none"
        """
        result = index(
            docs,
            self.record_manager,
            self,
            cleanup=cleanup_mode,
            source_id_key="digest"
        )
        return result

    def add_documents(self, docs: List[Document]) -> Dict[str, int]:
        """Add new documents to the collection without cleanup."""
        return self.update_collection(docs, cleanup_mode="none")

    def replace_collection(self, docs: List[Document]) -> Dict[str, int]:
        """Replace the entire collection with new documents."""
        return self.update_collection(docs, cleanup_mode="full")
    

class AsyncPGVector(ExtendedPGVector):
    """
    Asynchronous version of ExtendedPGVector.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = self.as_retriever()
        
    async def aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return await run_in_executor(None, self.retriever.get_relevant_documents, query, **kwargs)
        
    async def aget_all_ids(self) -> List[str]:
        return await run_in_executor(None, self.get_all_ids)

    async def aget_documents_by_ids(self, ids: List[str]) -> List[Document]:
        return await run_in_executor(None, self.get_documents_by_ids, ids)

    async def aupdate_collection(self, docs: List[Document], cleanup_mode: str = "incremental") -> Dict[str, int]:
        return await run_in_executor(None, self.update_collection, docs, cleanup_mode)

    async def aadd_documents(self, docs: List[Document]) -> Dict[str, int]:
        return await self.aupdate_collection(docs, cleanup_mode="none")

    async def areplace_collection(self, docs: List[Document]) -> Dict[str, int]:
        return await self.aupdate_collection(docs, cleanup_mode="full")

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        await run_in_executor(None, super().delete, ids, **kwargs)

def get_pgvector_service(connection_string: str, collection_name: str, embedding_function: Optional[Embeddings] = None, use_async: bool = False):
    """
    Factory function to get either a synchronous or asynchronous PGVector service.
    
    :param connection_string: Database connection string
    :param collection_name: Name of the collection
    :param embedding_function: Custom embedding function 
    :param use_async: If True, returns AsyncPGVector, otherwise ExtendedPGVector
    """
    if use_async:
        return AsyncPGVector(connection_string=connection_string, collection_name=collection_name, embedding_function=embedding_function)
    else:
        return ExtendedPGVector(connection_string=connection_string, collection_name=collection_name, embedding_function=embedding_function)
