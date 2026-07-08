"""Minimal FastAPI app using JSON file storage. No database required."""

from datetime import datetime

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
from ulid import ULID

from delegation_service.storage import FileStore

app = FastAPI(title="Delegation Service (Simple)", version="0.1.0")
store = FileStore("delegations.json")


class DelegationCreate(BaseModel):
    grantee: str
    resource_type: str
    resource_id: str
    scope: dict
    quota: dict | None = None
    expires_at: datetime | None = None


class DelegationUpdate(BaseModel):
    quota: dict | None = None
    expires_at: datetime | None = None


def get_caller(x_caller_identity: str | None = Header(default=None)) -> str:
    if not x_caller_identity:
        raise HTTPException(401, "Missing X-Caller-Identity header")
    return x_caller_identity


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/delegations", status_code=201)
def create_delegation(
    body: DelegationCreate,
    x_caller_identity: str | None = Header(default=None),
):
    caller = get_caller(x_caller_identity)

    if body.resource_type == "storage":
        consumed = {"bytes": 0}
    elif body.resource_type == "compute":
        consumed = {"node_hours": 0}
    else:
        consumed = {}

    delegation = {
        "delegation_id": str(ULID()),
        "parent_id": None,
        "delegator": caller,
        "grantee": body.grantee,
        "resource_type": body.resource_type,
        "resource_id": body.resource_id,
        "scope": body.scope,
        "quota": body.quota,
        "consumed": consumed,
        "suspended": False,
        "enforcement_ref": f"dry-run-acl-{body.grantee[-8:]}",
        "expires_at": body.expires_at.isoformat() if body.expires_at else None,
        "revoked": False,
        "revoked_at": None,
    }

    return store.create(delegation)


@app.get("/v1/delegations")
def list_delegations(
    grantee: str | None = Query(default=None),
    delegator: str | None = Query(default=None),
    resource_id: str | None = Query(default=None),
    include_revoked: bool = Query(default=False),
    x_caller_identity: str | None = Header(default=None),
):
    get_caller(x_caller_identity)
    return store.list(grantee, delegator, resource_id, include_revoked)


@app.get("/v1/delegations/{delegation_id}")
def get_delegation(
    delegation_id: str,
    x_caller_identity: str | None = Header(default=None),
):
    get_caller(x_caller_identity)
    d = store.get(delegation_id)
    if not d:
        raise HTTPException(404, "Delegation not found")
    return d


@app.patch("/v1/delegations/{delegation_id}")
def update_delegation(
    delegation_id: str,
    body: DelegationUpdate,
    x_caller_identity: str | None = Header(default=None),
):
    get_caller(x_caller_identity)
    d = store.get(delegation_id)
    if not d:
        raise HTTPException(404, "Delegation not found")
    if d.get("revoked"):
        raise HTTPException(400, "Cannot update revoked delegation")

    updates = {}
    if body.quota is not None:
        updates["quota"] = body.quota
    if body.expires_at is not None:
        updates["expires_at"] = body.expires_at.isoformat()

    return store.update(delegation_id, updates)


@app.delete("/v1/delegations/{delegation_id}", status_code=204)
def revoke_delegation(
    delegation_id: str,
    x_caller_identity: str | None = Header(default=None),
):
    get_caller(x_caller_identity)
    d = store.get(delegation_id)
    if not d:
        raise HTTPException(404, "Delegation not found")

    _revoke_recursive(delegation_id)


def _revoke_recursive(delegation_id: str) -> None:
    for child in store.get_children(delegation_id):
        _revoke_recursive(child["delegation_id"])
    store.delete(delegation_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
