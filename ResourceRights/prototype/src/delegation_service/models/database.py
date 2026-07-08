"""SQLAlchemy database models."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Delegation(Base):
    """Core delegation record."""

    __tablename__ = "delegations"

    delegation_id: Mapped[str] = mapped_column(String(26), primary_key=True)
    parent_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("delegations.delegation_id"), nullable=True
    )

    delegator: Mapped[str] = mapped_column(Text, nullable=False)
    grantee: Mapped[str] = mapped_column(Text, nullable=False)

    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[dict] = mapped_column(JSON, nullable=False)

    quota: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    consumed: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    suspended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    enforcement_ref: Mapped[str | None] = mapped_column(Text, nullable=True)

    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    parent: Mapped["Delegation | None"] = relationship(
        "Delegation", remote_side=[delegation_id], back_populates="children"
    )
    children: Mapped[list["Delegation"]] = relationship("Delegation", back_populates="parent")

    storage_tasks: Mapped[list["StorageTask"]] = relationship(
        "StorageTask", back_populates="delegation"
    )
    compute_jobs: Mapped[list["ComputeJob"]] = relationship(
        "ComputeJob", back_populates="delegation"
    )

    __table_args__ = (
        Index("idx_delegations_grantee", "grantee"),
        Index("idx_delegations_parent", "parent_id"),
        Index("idx_delegations_type", "resource_type"),
        Index("idx_delegations_expires", "expires_at", postgresql_where=~revoked),
    )


class StorageTask(Base):
    """Track Globus transfer tasks for consumption accounting."""

    __tablename__ = "storage_tasks"

    task_id: Mapped[str] = mapped_column(Text, primary_key=True)
    delegation_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("delegations.delegation_id"), nullable=False
    )

    bytes_transferred: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    delegation: Mapped[Delegation] = relationship("Delegation", back_populates="storage_tasks")


class ComputeJob(Base):
    """Track compute jobs for consumption accounting."""

    __tablename__ = "compute_jobs"

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)
    delegation_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("delegations.delegation_id"), nullable=False
    )

    nodes: Mapped[int] = mapped_column(Integer, nullable=False)
    walltime_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    node_hours_charged: Mapped[float | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    delegation: Mapped[Delegation] = relationship("Delegation", back_populates="compute_jobs")
