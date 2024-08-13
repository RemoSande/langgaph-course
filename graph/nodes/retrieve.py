from typing import Any, Dict
from graph.state import GraphState

async def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state.question

    documents = await state.db.retrieve_documents(question, k=5)
    return {"documents": documents, "question": question}

