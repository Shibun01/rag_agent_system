from __future__ import annotations

import re

from app.config.settings import get_settings

settings = get_settings()
_TENANT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_DELIMITER = "__"


def normalize_tenant_id(value: str) -> str:
    tenant_id = value.strip().lower()
    if not tenant_id or not _TENANT_PATTERN.fullmatch(tenant_id):
        raise ValueError("Tenant IDs must match ^[a-z0-9][a-z0-9_-]{0,63}$")
    return tenant_id


def scope_collection_name(collection_name: str, tenant_id: str) -> str:
    if tenant_id == settings.default_tenant_id:
        return collection_name
    prefix = f"{tenant_id}{_DELIMITER}"
    return collection_name if collection_name.startswith(prefix) else f"{prefix}{collection_name}"


def is_tenant_collection(collection_name: str, tenant_id: str) -> bool:
    if tenant_id == settings.default_tenant_id:
        return _DELIMITER not in collection_name
    return collection_name.startswith(f"{tenant_id}{_DELIMITER}")


def public_collection_name(collection_name: str, tenant_id: str) -> str:
    if tenant_id == settings.default_tenant_id:
        return collection_name
    prefix = f"{tenant_id}{_DELIMITER}"
    return collection_name[len(prefix):] if collection_name.startswith(prefix) else collection_name