"""Base adapter interface for resource type enforcement."""

from abc import ABC, abstractmethod
from datetime import datetime


class ResourceAdapter(ABC):
    """Abstract base class for resource type adapters."""

    @abstractmethod
    async def provision(
        self,
        grantee: str,
        resource_id: str,
        scope: dict,
        expires_at: datetime | None,
    ) -> str:
        """
        Provision enforcement for a delegation.

        Returns an enforcement_ref (e.g., ACL ID) to store with the delegation.
        """
        ...

    @abstractmethod
    async def update(
        self,
        enforcement_ref: str,
        resource_id: str,
        suspended: bool,
    ) -> None:
        """Update enforcement (e.g., downgrade to read-only on quota exhaustion)."""
        ...

    @abstractmethod
    async def revoke(self, enforcement_ref: str, resource_id: str) -> None:
        """Remove enforcement (delete ACL, etc.)."""
        ...
