"""FastAPI application entry point with lifespan, logging, and middleware."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import HealthResponse
from app.routers import chat, movies
from app.services.database import DatabaseService
from app.services.llm import OllamaService

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = DatabaseService(settings.db_path)
    llm = OllamaService()

    movies.init_router(db)
    chat.init_router(db, llm)

    app.state.db = db
    app.state.llm = llm

    logger.info(
        "Application started: db=%s  model=%s",
        settings.db_path, settings.ollama_model,
    )
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="Movie AI Agent",
    description=(
        "A conversational REST API that answers movie questions using the "
        "TMDB 5000 + MovieLens dataset, powered by a local LLM (Ollama phi3:mini)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.0f ms)",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Check database connectivity and Ollama availability."""
    db: DatabaseService = app.state.db
    llm: OllamaService = app.state.llm

    db_ok = db.health_check()
    ollama_status = await llm.health_check()

    overall = "healthy" if db_ok and ollama_status.get("model_loaded") else "degraded"
    return HealthResponse(status=overall, database=db_ok, ollama=ollama_status)


app.include_router(movies.router)
app.include_router(chat.router)
