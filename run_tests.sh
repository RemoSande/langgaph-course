#!/bin/bash
set -e

# Start the test database
docker-compose up -d test_db

# Wait for the database to be ready
echo "Waiting for test database to be ready..."
sleep 10

# Run the tests
pytest tests/

# Shut down the test database
docker-compose down
