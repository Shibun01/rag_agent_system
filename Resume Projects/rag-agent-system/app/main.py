import sys
import os

# Allow running from inside the app/ directory or from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import setup_logging, get_logger, get_settings
from app.api.v1.router import router as v1_router

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Agent System...")
    # Initialize vector store, DB connections, etc.
    yield
    logger.info("Shutting down RAG Agent System...")


app = FastAPI(
    title="RAG Agent System",
    description=(
        "Production-grade RAG & Agents API covering: Naive RAG, Advanced RAG, "
        "Corrective RAG (CRAG), Self-RAG, GraphRAG, HyDE, Multi-query RAG, "
        "ReAct Agents, Plan-Execute, Reflection, Supervisor, and Multi-agent orchestration."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins_list,
    allow_methods=settings.cors_allow_methods_list,
    allow_headers=settings.cors_allow_headers_list,
)

app.include_router(v1_router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return JSONResponse({"status": "ok", "service": "rag-agent-system"})


@app.get("/health", tags=["Health"])
async def health():
    return JSONResponse({"status": "healthy"})
