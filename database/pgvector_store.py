import asyncio
from typing import Any, List, Optional

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from sqlalchemy.orm import Session

class ExtendedPgVector(PGVector):
    def get_all_ids(self) -> List[str]:
        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore.custom_id).all()
            return [result[0] for result in results if result[0] is not None]

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
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

class AsyncPgVector(ExtendedPgVector):
    """
    An asynchronous wrapper for the ExtendedPgVector class.
    This class provides asynchronous versions of the methods in ExtendedPgVector,
    allowing for non-blocking database operations.
    """
    async def get_all_ids(self) -> List[str]:
        """
        Asynchronously retrieve all document IDs from the database.
        
        Returns:
            List[str]: A list of all document IDs.
        """
        return await run_in_executor(None, super().get_all_ids)

    async def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Asynchronously retrieve documents by their IDs.
        
        Args:
            ids (List[str]): A list of document IDs to retrieve.
        
        Returns:
            List[Document]: A list of Document objects corresponding to the given IDs.
        """
        return await run_in_executor(None, super().get_documents_by_ids, ids)

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        collection_only: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Asynchronously delete documents from the database.
        
        Args:
            ids (Optional[List[str]]): A list of document IDs to delete. If None, all documents are deleted.
            collection_only (bool): If True, only delete the collection, not the documents.
            **kwargs: Additional keyword arguments to pass to the delete method.
        """
        await run_in_executor(None, super().delete, ids, collection_only, **kwargs)

    async def aadd_documents(self, documents: List[Document]) -> List[str]:
        """
        Asynchronously add documents to the database.
        
        Args:
            documents (List[Document]): A list of Document objects to add to the database.
        
        Returns:
            List[str]: A list of IDs of the added documents.
        """
        return await run_in_executor(None, self.add_documents, documents)
