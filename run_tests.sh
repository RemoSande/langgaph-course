#!/bin/bash
set -e

# Start the test database and API service
docker-compose up -d test_db api

# Wait for the services to be ready
echo "Waiting for services to be ready..."
sleep 15

# Run the tests
pytest tests/test_integration.py

# Shut down the services
docker-compose down
