import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from api.main import app
from database.db import get_database
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env file
load_dotenv()

# Set up test database URL
os.environ['DATABASE_URL'] = os.getenv('TEST_DATABASE_URL', 'postgresql+psycopg://test_user:test_password@localhost:5434/test_db')

client = TestClient(app)

@pytest.fixture(scope="module")
def test_app():
    # Set up any test-specific configurations here
    return app

@pytest.fixture(autouse=True)
async def setup_test_environment():
    # Ensure the DATABASE_URL is set for each test
    os.environ['DATABASE_URL'] = os.getenv('TEST_DATABASE_URL', 'postgresql+psycopg://test_user:test_password@localhost:5434/test_db')
    
    # Clear the test database before each test
    db = await get_database()
    await db.delete_documents(await db.get_all_ids())
    
    yield
    
    # Clean up after tests
    await db.delete_documents(await db.get_all_ids())

@pytest.mark.asyncio
async def test_ingest_document(test_app):
    test_documents = [
        {
            "page_content": "This is a test document about AI.",
            "metadata": {
                "source": "book",
                "author": "Jane Smith",
                "page": 42
            }
        },
        {
            "page_content": "Another test document about machine learning.",
            "metadata": {
                "source": "web",
                "url": "https://example.com/ml",
                "date": "2023-06-15"
            }
        }
    ]
    
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        response = await ac.post("/documents/", json=test_documents)
    
    assert response.status_code == 200
    result = response.json()
    assert "ids" in result
    assert len(result["ids"]) == 2  # We sent 2 documents

    # Now let's try to retrieve these documents
    for doc_id in result["ids"]:
        retrieve_response = await ac.get(f"/documents/?ids={doc_id}")
        assert retrieve_response.status_code == 200
        retrieved_doc = retrieve_response.json()[0]
        assert "page_content" in retrieved_doc
        assert "metadata" in retrieved_doc

@pytest.mark.asyncio
async def test_retrieve_document(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        # First, ingest a document
        ingest_response = await ac.post("/documents/", json=[{"page_content": "Test document"}])
        doc_id = ingest_response.json()["ids"][0]

        # Then, retrieve it
        response = await ac.get(f"/documents/?ids={doc_id}")
    assert response.status_code == 200
    assert response.json()[0]["page_content"] == "Test document"

@pytest.mark.asyncio
async def test_database_connection():
    db = get_database()
    assert await db.check_health() == True

@pytest.mark.asyncio
async def test_delete_document(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        # First, ingest a document
        ingest_response = await ac.post("/documents/", json=[{"page_content": "Test document to delete"}])
        doc_id = ingest_response.json()["ids"][0]

        # Then, delete it
        delete_response = await ac.delete(f"/documents/?ids={doc_id}")
        assert delete_response.status_code == 200

        # Try to retrieve it (should fail)
        get_response = await ac.get(f"/documents/?ids={doc_id}")
        assert get_response.status_code == 404

@pytest.mark.asyncio
async def test_chat(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        response = await ac.post("/chat/", json={"msg": "Hello, how are you?", "client_topics": []})
    assert response.status_code == 200
    assert "response" in response.json()

@pytest.mark.asyncio
async def test_health_check(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
