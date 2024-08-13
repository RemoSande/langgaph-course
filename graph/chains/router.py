from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# client_topics = ['agents', 'prompt engineering', 'adversarial attacks']

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to 
Use the vectorstore for questions on the following topics: {client_topics}. For all other topics, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Given the topics: {client_topics}, route this question: {question}"),
        ]
    )

question_router = route_prompt | structured_llm_router


