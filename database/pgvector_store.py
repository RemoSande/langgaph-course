import logging
from typing import Any, List, Optional
from urllib.parse import urlparse

from langchain_postgres import PGVector
from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

logger = logging.getLogger(__name__)

class AsyncPGVector(PGVector):
    """
    An asynchronous implementation of PGVector using langchain_postgres.
    This class provides asynchronous methods for vector store operations,
    allowing for non-blocking database interactions.
    """

    @classmethod
    async def create(cls, connection_string: str, collection_name: str, embedding_function: Any):
        """
        Asynchronously create and initialize a new AsyncPGVector instance.

        Args:
            connection_string (str): The database connection string.
            collection_name (str): The name of the collection to use.
            embedding_function (Any): The function to use for creating embeddings.

        Returns:
            AsyncPGVector: An initialized AsyncPGVector instance.
        """
        # Convert connection string to use postgresql+psycopg://
        parsed = urlparse(connection_string)
        new_connection_string = f"postgresql+psycopg://{parsed.netloc}{parsed.path}"

        return cls(
            connection_string=new_connection_string,
            collection_name=collection_name,
            embedding_function=embedding_function,
            async_mode=True
        )

    async def get_all_ids(self) -> List[str]:
        """
        Asynchronously retrieve all document IDs from the database.
        
        Returns:
            List[str]: A list of all document IDs.
        """
        try:
            async with AsyncSession(self._engine) as session:
                result = await session.execute(select(self.EmbeddingStore.custom_id))
                return [row[0] for row in result if row[0] is not None]
        except Exception as e:
            logger.error(f"Error retrieving all IDs: {e}")
            raise

    async def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Asynchronously retrieve documents by their IDs.
        
        Args:
            ids (List[str]): A list of document IDs to retrieve.
        
        Returns:
            List[Document]: A list of Document objects corresponding to the given IDs.
        """
        try:
            async with AsyncSession(self._engine) as session:
                result = await session.execute(
                    select(self.EmbeddingStore).filter(self.EmbeddingStore.custom_id.in_(ids))
                )
                rows = result.all()
                return [
                    Document(page_content=row[0].document, metadata=row[0].cmetadata or {})
                    for row in rows if row[0].custom_id in ids
                ]
        except Exception as e:
            logger.error(f"Error retrieving documents by IDs: {e}")
            raise

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Asynchronously delete documents from the database.
        
        Args:
            ids (Optional[List[str]]): A list of document IDs to delete. If None, all documents are deleted.
            **kwargs: Additional keyword arguments to pass to the delete method.
        """
        try:
            await super().adelete(ids=ids, **kwargs)
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    async def aadd_documents(self, documents: List[Document]) -> List[str]:
        """
        Asynchronously add documents to the database.
        
        Args:
            documents (List[Document]): A list of Document objects to add to the database.
        
        Returns:
            List[str]: A list of IDs of the added documents.
        """
        try:
            return await super().aadd_documents(documents)
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """
        Asynchronously perform a similarity search.

        Args:
            query (str): The query string.
            k (int): The number of results to return.
            **kwargs: Additional arguments to pass to the similarity search.

        Returns:
            List[Document]: A list of similar documents.
        """
        try:
            return await super().asimilarity_search(query, k=k, **kwargs)
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()

    async def aclose(self):
        """Asynchronously close the database connection."""
        if hasattr(self, '_engine'):
            await self._engine.dispose()
