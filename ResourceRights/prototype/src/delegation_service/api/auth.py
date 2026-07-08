"""Authentication utilities for extracting caller identity."""

from fastapi import Header, HTTPException

# In production, this would introspect the Globus token
# For now, accept a header with the caller's identity


async def get_caller_identity(
    x_caller_identity: str | None = Header(default=None, alias="X-Caller-Identity"),
    authorization: str | None = Header(default=None),
) -> str:
    """
    Extract caller identity from request.

    For MVP: accept X-Caller-Identity header directly.
    Production: introspect the Authorization bearer token via Globus Auth.
    """
    if x_caller_identity:
        return x_caller_identity

    if authorization and authorization.startswith("Bearer "):
        # TODO: Introspect token via Globus Auth to get identity
        # For now, just reject if no X-Caller-Identity provided
        raise HTTPException(
            status_code=501,
            detail="Token introspection not yet implemented. Use X-Caller-Identity header.",
        )

    raise HTTPException(
        status_code=401,
        detail="Missing authentication. Provide X-Caller-Identity header or Bearer token.",
    )
