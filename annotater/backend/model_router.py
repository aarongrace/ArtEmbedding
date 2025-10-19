from pydantic import BaseModel, conlist
from typing import Dict, List
from fastapi import APIRouter

from model_services import *

from pydantic import BaseModel, conlist
from typing import List


from model_cache import get_cache

class PaintingData(BaseModel):
    id: str
    title: str
    artist: str
    year: str

    # --- Raw WikiArt metadata ---
    genre: List[str]
    movement: List[str]
    tags: List[str]

    # --- Image & Model output ---
    imageUrl: str                 # base64-encoded full image
    vector: List[float]  # 17-dim vector from model

class GroundTruthLabel(BaseModel):
    id: str
    vector: List[float]

model_router = APIRouter()
@model_router.get("/painting", response_model=PaintingData)
async def get_painting_with_forward():
    cache = get_cache()
    image_id, predictions = cache.get_embedding()
    image_entry = get_metadata_by_id(image_id)
    image_url_mounted = get_image_path(image_id, local=False)
    painting_data = PaintingData(
        id=image_id,
        title=image_entry.get("title", "Unknown Title"),
        artist=image_entry.get("artist", "Unknown Artist"),
        year=image_entry.get("year", "Unknown Year"),
        genre=image_entry.get("genres", []),
        # note that "styles" field in metadata corresponds to "movement"
        movement=image_entry.get("styles", []),
        tags=image_entry.get("tags", []),
        imageUrl=image_url_mounted,
        vector=predictions
    )
    return painting_data

@model_router.post("/set_demo_mode")
async def set_demo_mode_in_services_file(value: bool):
    set_demo_mode(value)
    return {"demo_mode": value}

@model_router.post("/set_fetch_large_images")
async def set_fetch_large_images(value: bool):
    cache = get_cache()
    cache.set_fetch_large_images(value)
    return {"fetch_large_images": value}

@model_router.post("/set_fetch_test_images")
async def set_fetch_test_images(value: bool):
    cache = get_cache()
    cache.set_fetch_test_images(value)
    return {"fetch_test_images": value}

@model_router.post("/label")
async def upload_label(label: GroundTruthLabel):
    add_to_labels_list(label.id, label.vector)
    cache = get_cache()
    cache.add_to_backprop_queue(label.id, label.vector)
    return {"status": "ok"}

@model_router.post("/save_checkpoint")
async def save_model_checkpoint():
    from embed_model import save_progress, get_model_and_processor
    from datetime import datetime
    model, _ = get_model_and_processor()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_progress(model, f"model_session_{timestamp}")
    return {"status": "model saved"}