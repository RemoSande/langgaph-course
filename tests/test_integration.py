import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from api.main import app
from database.db import get_database

client = TestClient(app)

@pytest.mark.asyncio
async def test_ingest_document():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/documents/", json=[{"page_content": "Test document"}])
    assert response.status_code == 200
    assert "ids" in response.json()

@pytest.mark.asyncio
async def test_retrieve_document():
    # First, ingest a document
    async with AsyncClient(app=app, base_url="http://test") as ac:
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
async def test_delete_document():
    async with AsyncClient(app=app, base_url="http://test") as ac:
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
async def test_chat():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/chat/", json={"msg": "Hello, how are you?", "client_topics": []})
    assert response.status_code == 200
    assert "response" in response.json()

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
