from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


import sys
from pathlib import Path

from fastapi.staticfiles import StaticFiles
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

ROOT_DIR = Path(__file__).resolve().parents[2]  # 2 levels above
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model_router import model_router
from model_services import PAINTINGS_DIR


# import logging
# import setup_logging

# setup_logging.setup_logging()

# logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from model_services import ensure_user_state, USER_NAME
    from embed_model import get_model_and_processor
    # --- Startup code ---
    ensure_user_state(USER_NAME)   # Ensure default user exists
    get_model_and_processor()      # Preload model
    print(f"Startup complete: user {USER_NAME} ensured and model loaded.")
    
    yield  # Everything after this is shutdown code

    # --- Optional shutdown code ---
    print("Shutdown complete.")


app = FastAPI(title="Art Embeddings Backend", version="0.1.0", lifespan=lifespan)

app.mount("/paintings", StaticFiles(directory=PAINTINGS_DIR, html=True), name="paintings")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router, prefix="/model")



# --- Startup event ---
@app.on_event("startup")
async def startup_event():
    # Ensure default user exists
    ensure_user_state(USER_NAME)
    # You could also preload the model here if desired
    print(f"Startup complete: ensured user {USER_NAME} exists.")

@app.get("/")
async def welcome() -> dict:
    return {"msg": "Welcome to the Art Embedding Annotater Backend!"}
