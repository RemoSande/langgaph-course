from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from graph.graph import app as graph_app

router = APIRouter()

class QueryInput(BaseModel):
    question: str
    client_topics: List[str]

@router.post("/query")
async def query(input: QueryInput):
    try:
        result = await graph_app.ainvoke({
            "question": input.question,
            "client_topics": input.client_topics
        })
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))