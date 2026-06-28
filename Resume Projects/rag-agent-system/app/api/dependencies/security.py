from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config.settings import get_settings
from app.core.tenancy import normalize_tenant_id

settings = get_settings()


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not settings.auth_enabled:
        return

    allowed_keys = settings.api_keys_list
    if not allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key authentication is enabled but no API keys are configured.",
        )

    if x_api_key not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


async def get_tenant_id(x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID")) -> str:
    tenant_id = x_tenant_id or settings.default_tenant_id
    if settings.require_tenant_header and not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Tenant-ID header is required.",
        )

    try:
        return normalize_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc