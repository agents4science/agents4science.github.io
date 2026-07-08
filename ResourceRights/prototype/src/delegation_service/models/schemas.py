"""Pydantic schemas for API request/response validation."""

from datetime import datetime

from pydantic import BaseModel, Field


class StorageScope(BaseModel):
    path: str
    operations: list[str] = Field(default_factory=lambda: ["read", "write"])


class ComputeScope(BaseModel):
    queues: list[str]
    max_nodes_per_job: int | None = None
    max_walltime_hours: int | None = None


class StorageQuota(BaseModel):
    bytes: int


class ComputeQuota(BaseModel):
    node_hours: float


class DelegationCreate(BaseModel):
    grantee: str
    resource_type: str = Field(pattern="^(storage|compute)$")
    resource_id: str
    scope: dict
    quota: dict | None = None
    expires_at: datetime | None = None


class DelegationUpdate(BaseModel):
    quota: dict | None = None
    expires_at: datetime | None = None


class DelegationResponse(BaseModel):
    delegation_id: str
    parent_id: str | None
    delegator: str
    grantee: str
    resource_type: str
    resource_id: str
    scope: dict
    quota: dict | None
    consumed: dict
    suspended: bool
    enforcement_ref: str | None = None
    expires_at: datetime | None
    revoked: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TransferRegister(BaseModel):
    task_id: str
    destination_path: str


class TransferWebhook(BaseModel):
    task_id: str
    status: str
    bytes_transferred: int | None = None


class ComputeWebhook(BaseModel):
    job_id: str
    delegation_id: str
    status: str
    nodes: int
    walltime_seconds: int | None = None
    node_hours_charged: float | None = None
