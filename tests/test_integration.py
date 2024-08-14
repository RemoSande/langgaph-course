import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from api.main import app
from database.db import get_database
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up test database URL
os.environ['DATABASE_URL'] = os.getenv('TEST_DATABASE_URL', 'postgresql+psycopg://test_user:test_password@test_db:5432/test_db')

client = TestClient(app)

@pytest.fixture(scope="module")
def test_app():
    # Set up any test-specific configurations here
    return app

@pytest.mark.asyncio
async def test_ingest_document(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        response = await ac.post("/documents/", json=[{"page_content": "Test document"}])
    assert response.status_code == 200
    assert "ids" in response.json()

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
