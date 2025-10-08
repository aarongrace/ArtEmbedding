from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


import sys
from pathlib import Path

from fastapi.staticfiles import StaticFiles
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from model_router import model_router
from model_services import PAINTINGS_DIR


# import logging
# import setup_logging

# setup_logging.setup_logging()

# logger = logging.getLogger(__name__)



app = FastAPI(title="Art Embeddings Backend", version="0.1.0")

app.mount("/paintings", StaticFiles(directory=PAINTINGS_DIR, html=True), name="paintings")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router, prefix="/model")

@app.get("/")
async def welcome() -> dict:
    return {"msg": "Welcome to the Art Embedding Annotater Backend!"}
