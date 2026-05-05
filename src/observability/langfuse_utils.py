from typing import Any, Dict, Optional
import uuid

try:
    from langfuse import get_client, propagate_attributes
    LANGFUSE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    get_client = None
    propagate_attributes = None
    LANGFUSE_AVAILABLE = False


def init_langfuse_client(public_key: Optional[str] = None, secret_key: Optional[str] = None, host: Optional[str] = None):
    """Initialize and return the Langfuse client (best-effort).

    This is a small helper to centralize client creation for notebooks and tests.
    """
    if not LANGFUSE_AVAILABLE:
        return None
    try:
        client = get_client()
        return client
    except Exception:
        return None


def create_trace_id(client: Any, seed: Optional[str] = None) -> str:
    if client is None:
        return seed or str(uuid.uuid4())
    if hasattr(client, "create_trace_id"):
        try:
            return client.create_trace_id(seed=seed or str(uuid.uuid4()))
        except Exception:
            return seed or str(uuid.uuid4())
    return seed or str(uuid.uuid4())
