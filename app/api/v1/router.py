from fastapi import APIRouter
from app.api.v1.endpoints import rag, agents, documents, chat

router = APIRouter()

router.include_router(rag.router,       prefix="/rag",       tags=["RAG"])
router.include_router(agents.router,    prefix="/agents",    tags=["Agents"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(chat.router,      prefix="/chat",      tags=["Chat"])
