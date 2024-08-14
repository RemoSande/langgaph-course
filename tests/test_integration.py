import pytest
from fastapi.testclient import TestClient
from api.main import app
from database.db import get_database

client = TestClient(app)

def test_ingest_document():
    response = client.post("/documents/", json={"page_content": "Test document"})
    assert response.status_code == 200
    assert "id" in response.json()

def test_retrieve_document():
    # First, ingest a document
    ingest_response = client.post("/documents/", json={"page_content": "Test document"})
    doc_id = ingest_response.json()["id"]

    # Then, retrieve it
    response = client.get(f"/documents/?ids={doc_id}")
    assert response.status_code == 200
    assert response.json()[0]["page_content"] == "Test document"

def test_database_connection():
    db = get_database()
    assert db.check_health() == True

def test_delete_document():
    # First, ingest a document
    ingest_response = client.post("/documents/", json={"page_content": "Test document to delete"})
    doc_id = ingest_response.json()["id"]

    # Then, delete it
    delete_response = client.delete(f"/documents/?ids={doc_id}")
    assert delete_response.status_code == 200

    # Try to retrieve it (should fail)
    get_response = client.get(f"/documents/?ids={doc_id}")
    assert get_response.status_code == 404
