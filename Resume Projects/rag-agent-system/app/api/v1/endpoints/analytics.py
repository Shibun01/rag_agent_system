from fastapi import APIRouter, Depends

from app.api.dependencies import get_tenant_id
from app.models.schemas import FeedbackRequest, FeedbackResponse, QueryLogResponse
from app.services.analytics import create_feedback, list_feedback, list_query_logs

router = APIRouter()


@router.get("/queries", response_model=list[QueryLogResponse], summary="List query logs")
async def query_logs(limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    return await list_query_logs(limit, tenant_id=tenant_id)


@router.post("/feedback", response_model=FeedbackResponse, summary="Record user feedback")
async def submit_feedback(req: FeedbackRequest, tenant_id: str = Depends(get_tenant_id)):
    return await create_feedback(
        tenant_id=tenant_id,
        query_log_id=req.query_log_id,
        rating=req.rating,
        comment=req.comment,
        query=req.query,
        strategy=req.strategy,
        collection_name=req.collection_name,
    )


@router.get("/feedback", response_model=list[FeedbackResponse], summary="List feedback")
async def feedback(limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    return await list_feedback(limit, tenant_id=tenant_id)
