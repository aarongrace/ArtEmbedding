import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
from model_cache import get_cache



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
    get_cache() # Initialize cache
    
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

@app.get("/")
async def welcome() -> dict:
    return {"msg": "Welcome to the Art Embedding Annotater Backend!"}
