"""Resource type adapters for enforcement provisioning."""

from delegation_service.adapters.base import ResourceAdapter
from delegation_service.adapters.storage import StorageAdapter

__all__ = ["ResourceAdapter", "StorageAdapter"]
