import pytest


import sys
from pathlib import Path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from model_router import get_painting_with_forward
from model_services import  get_seen_list

@pytest.mark.asyncio
async def test_get_painting_with_forward():
    prev_seen = get_seen_list()
    print(f"Previously seen paintings: {prev_seen}")
    painting = await get_painting_with_forward()
    after_seen = get_seen_list()
    print(f"Seen paintings after forward: {after_seen}")
    assert len(after_seen) == len(prev_seen) + 1
    print(painting)

    assert painting is not None
    assert hasattr(painting, 'id')
    assert hasattr(painting, 'title')
    assert hasattr(painting, 'artist')
    assert hasattr(painting, 'year')
    assert hasattr(painting, 'genre')
    assert hasattr(painting, 'movement')
    assert hasattr(painting, 'tags')
    assert hasattr(painting, 'imageUrl')
    assert hasattr(painting, 'vector')
    assert len(painting.vector) == 17


import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from model_router import model_router, GroundTruthLabel

# Include your router in a test app
from fastapi import FastAPI

app_for_testing = FastAPI()
app_for_testing.include_router(model_router)

client = TestClient(app_for_testing)

# --- Test data ---
sample_label = {
    "id": "painting_123",
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5,
               0.6, 0.7, 0.8, 0.9, 1.0,
               0.11, 0.12, 0.13, 0.14, 0.15,
               0.16, 0.17]
}


def test_upload_label_invalid_data():
    # Missing vector
    invalid_label = {"id": "painting_456"}

    response = client.post("/label", json=invalid_label)
    assert response.status_code == 422  # FastAPI validation error

def test_upload_label_wrong_vector_type():
    # vector should be list of floats
    invalid_label = {"id": "painting_789", "vector": "not_a_list"}

    response = client.post("/label", json=invalid_label)
    assert response.status_code == 422
