from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

class PGVectorDatabase:
    def __init__(self, connection_string: str, collection_name: str):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()

    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        vectorstore = PGVector(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        docs = await vectorstore.asimilarity_search_with_relevance_scores(query, k=k)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc, _ in docs]
