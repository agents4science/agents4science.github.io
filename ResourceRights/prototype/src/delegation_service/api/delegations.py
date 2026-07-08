"""Delegation API endpoints."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from delegation_service.adapters.storage import StorageAdapter
from delegation_service.api.auth import get_caller_identity
from delegation_service.database import get_db
from delegation_service.models.database import Delegation
from delegation_service.models.schemas import (
    DelegationCreate,
    DelegationResponse,
    DelegationUpdate,
)

router = APIRouter(prefix="/delegations", tags=["delegations"])

storage_adapter = StorageAdapter()


def get_adapter(resource_type: str):
    if resource_type == "storage":
        return storage_adapter
    raise HTTPException(status_code=400, detail=f"Unsupported resource type: {resource_type}")


@router.post("", response_model=DelegationResponse, status_code=201)
async def create_delegation(
    body: DelegationCreate,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_caller_identity),
):
    """Create a new delegation."""
    adapter = get_adapter(body.resource_type)

    # TODO: Validate caller has authority to delegate
    # For MVP, allow any authenticated caller to create root delegations

    delegation_id = str(ULID())

    # Provision enforcement
    enforcement_ref = await adapter.provision(
        grantee=body.grantee,
        resource_id=body.resource_id,
        scope=body.scope,
        expires_at=body.expires_at,
    )

    # Determine initial consumed value based on resource type
    if body.resource_type == "storage":
        consumed = {"bytes": 0}
    elif body.resource_type == "compute":
        consumed = {"node_hours": 0}
    else:
        consumed = {}

    delegation = Delegation(
        delegation_id=delegation_id,
        parent_id=None,  # TODO: support hierarchical delegation
        delegator=caller,
        grantee=body.grantee,
        resource_type=body.resource_type,
        resource_id=body.resource_id,
        scope=body.scope,
        quota=body.quota,
        consumed=consumed,
        enforcement_ref=enforcement_ref,
        expires_at=body.expires_at,
    )

    db.add(delegation)
    await db.flush()
    await db.refresh(delegation)

    return delegation


@router.get("", response_model=list[DelegationResponse])
async def list_delegations(
    grantee: str | None = Query(default=None),
    delegator: str | None = Query(default=None),
    resource_id: str | None = Query(default=None),
    include_revoked: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_caller_identity),
):
    """List delegations with optional filters."""
    query = select(Delegation)

    if grantee:
        query = query.where(Delegation.grantee == grantee)
    if delegator:
        query = query.where(Delegation.delegator == delegator)
    if resource_id:
        query = query.where(Delegation.resource_id == resource_id)
    if not include_revoked:
        query = query.where(Delegation.revoked == False)  # noqa: E712

    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{delegation_id}", response_model=DelegationResponse)
async def get_delegation(
    delegation_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_caller_identity),
):
    """Get a specific delegation."""
    result = await db.execute(
        select(Delegation).where(Delegation.delegation_id == delegation_id)
    )
    delegation = result.scalar_one_or_none()

    if not delegation:
        raise HTTPException(status_code=404, detail="Delegation not found")

    # TODO: Check caller is delegator, grantee, or ancestor
    return delegation


@router.patch("/{delegation_id}", response_model=DelegationResponse)
async def update_delegation(
    delegation_id: str,
    body: DelegationUpdate,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_caller_identity),
):
    """Update a delegation (quota or expiration)."""
    result = await db.execute(
        select(Delegation).where(Delegation.delegation_id == delegation_id)
    )
    delegation = result.scalar_one_or_none()

    if not delegation:
        raise HTTPException(status_code=404, detail="Delegation not found")

    if delegation.revoked:
        raise HTTPException(status_code=400, detail="Cannot update revoked delegation")

    # TODO: Check caller is delegator or ancestor

    if body.quota is not None:
        delegation.quota = body.quota
        # If quota increased and currently suspended, restore access
        if delegation.suspended:
            adapter = get_adapter(delegation.resource_type)
            await adapter.update(
                delegation.enforcement_ref,
                delegation.resource_id,
                suspended=False,
            )
            delegation.suspended = False

    if body.expires_at is not None:
        delegation.expires_at = body.expires_at

    await db.flush()
    await db.refresh(delegation)
    return delegation


@router.delete("/{delegation_id}", status_code=204)
async def revoke_delegation(
    delegation_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_caller_identity),
):
    """Revoke a delegation and all its children."""
    result = await db.execute(
        select(Delegation).where(Delegation.delegation_id == delegation_id)
    )
    delegation = result.scalar_one_or_none()

    if not delegation:
        raise HTTPException(status_code=404, detail="Delegation not found")

    if delegation.revoked:
        return  # Already revoked

    # TODO: Check caller is delegator or ancestor

    await _revoke_recursive(db, delegation)


async def _revoke_recursive(db: AsyncSession, delegation: Delegation) -> None:
    """Revoke a delegation and all its descendants."""
    # First revoke children
    result = await db.execute(
        select(Delegation).where(
            Delegation.parent_id == delegation.delegation_id,
            Delegation.revoked == False,  # noqa: E712
        )
    )
    children = result.scalars().all()

    for child in children:
        await _revoke_recursive(db, child)

    # Revoke enforcement
    if delegation.enforcement_ref:
        adapter = get_adapter(delegation.resource_type)
        await adapter.revoke(delegation.enforcement_ref, delegation.resource_id)

    # Mark as revoked
    delegation.revoked = True
    delegation.revoked_at = datetime.now(UTC)
