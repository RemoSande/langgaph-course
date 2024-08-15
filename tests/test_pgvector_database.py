import pytest
from database.db import PGVectorDatabase
from config.settings import settings

@pytest.mark.asyncio
async def test_pgvector_store_and_retrieve():
    # Initialize the database with test credentials
    db = PGVectorDatabase(settings.TEST_DATABASE_URL, "test_collection")

    # Create sample documents
    sample_docs = [
        {"content": "This is a test document 1", "metadata": {"source": "test1"}},
        {"content": "This is a test document 2", "metadata": {"source": "test2"}},
    ]

    doc_ids = []  # Initialize doc_ids here
    try:
        # Store the documents
        doc_ids = await db.store_documents(sample_docs)

        # Verify that we got the correct number of IDs back
        assert len(doc_ids) == len(sample_docs)

        # Retrieve all document IDs
        all_ids = await db.get_all_ids()
        assert set(doc_ids).issubset(set(all_ids))

        # Retrieve the documents to verify they were stored correctly
        retrieved_docs = await db.get_documents_by_ids(doc_ids)

        # Verify the content of the retrieved documents
        assert len(retrieved_docs) == len(sample_docs)
        for original, retrieved in zip(sample_docs, retrieved_docs):
            assert original["content"] == retrieved["content"]
            assert original["metadata"] == retrieved["metadata"]

        # Test similarity search
        search_result = await db.retrieve_documents("test document", k=2)
        assert len(search_result) == 2
        assert all("content" in doc and "metadata" in doc and "score" in doc for doc in search_result)

    finally:
        # Clean up: delete the test documents
        if doc_ids:
            await db.delete_documents(doc_ids)

    # Verify documents were deleted
    remaining_docs = await db.get_documents_by_ids(doc_ids)
    assert len(remaining_docs) == 0

@pytest.mark.asyncio
async def test_pgvector_update_document():
    db = PGVectorDatabase(settings.TEST_DATABASE_URL, "test_collection")

    # Create and store a document
    doc = {"content": "Original content", "metadata": {"source": "test"}}
    [doc_id] = await db.store_documents([doc])

    try:
        # Update the document
        new_content = "Updated content"
        await db.update_document(doc_id, new_content, {"source": "test_updated"})

        # Retrieve and verify the updated document
        [updated_doc] = await db.get_documents_by_ids([doc_id])
        assert updated_doc["content"] == new_content
        assert updated_doc["metadata"]["source"] == "test_updated"

    finally:
        # Clean up
        await db.delete_documents([doc_id])

@pytest.mark.asyncio
async def test_pgvector_health_check():
    db = PGVectorDatabase(settings.TEST_DATABASE_URL, "test_collection")
    
    # Check database health
    health_status = await db.check_health()
    assert health_status == True

@pytest.mark.asyncio
async def test_create_and_populate_custom_table():
    db = PGVectorDatabase(settings.TEST_DATABASE_URL, "test_collection")
    table_name = "test_items"
    
    try:
        # Create a new table
        await db.create_custom_table(table_name)
        
        # Insert test data
        test_data = [
            {"name": "Item 1", "description": "This is item 1"},
            {"name": "Item 2", "description": "This is item 2"},
            {"name": "Item 3", "description": "This is item 3"},
        ]
        await db.insert_test_data(table_name, test_data)
        
        # Fetch and verify the data
        fetched_data = await db.fetch_all_from_table(table_name)
        
        assert len(fetched_data) == len(test_data)
        for i, item in enumerate(fetched_data):
            assert item['name'] == test_data[i]['name']
            assert item['description'] == test_data[i]['description']
        
    finally:
        # Clean up: drop the test table
        await db.drop_table(table_name)

    # Verify the table was dropped
    try:
        await db.fetch_all_from_table(table_name)
        pytest.fail(f"The '{table_name}' table should not exist after dropping it")
    except Exception:
        pass  # Expected behavior
