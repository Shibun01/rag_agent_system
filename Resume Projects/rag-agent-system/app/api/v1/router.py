from fastapi import APIRouter, Depends
from app.api.dependencies import require_api_key
from app.api.v1.endpoints import agents, analytics, chat, documents, rag

router = APIRouter(dependencies=[Depends(require_api_key)])

router.include_router(rag.router,       prefix="/rag",       tags=["RAG"])
router.include_router(agents.router,    prefix="/agents",    tags=["Agents"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(chat.router,      prefix="/chat",      tags=["Chat"])
router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
