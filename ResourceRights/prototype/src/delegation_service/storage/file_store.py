"""Simple JSON file storage for local development."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FileStore:
    """Store delegations in a JSON file with sequential search."""

    def __init__(self, path: str = "delegations.json"):
        self.path = Path(path)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self.path.write_text("[]")

    def _load(self) -> list[dict[str, Any]]:
        return json.loads(self.path.read_text())

    def _save(self, data: list[dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.rename(self.path)

    def create(self, delegation: dict[str, Any]) -> dict[str, Any]:
        data = self._load()
        delegation["created_at"] = datetime.utcnow().isoformat()
        data.append(delegation)
        self._save(data)
        return delegation

    def get(self, delegation_id: str) -> dict[str, Any] | None:
        for d in self._load():
            if d["delegation_id"] == delegation_id:
                return d
        return None

    def list(
        self,
        grantee: str | None = None,
        delegator: str | None = None,
        resource_id: str | None = None,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        results = []
        for d in self._load():
            if not include_revoked and d.get("revoked"):
                continue
            if grantee and d.get("grantee") != grantee:
                continue
            if delegator and d.get("delegator") != delegator:
                continue
            if resource_id and d.get("resource_id") != resource_id:
                continue
            results.append(d)
        return results

    def update(self, delegation_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        data = self._load()
        for d in data:
            if d["delegation_id"] == delegation_id:
                d.update(updates)
                self._save(data)
                return d
        return None

    def delete(self, delegation_id: str) -> bool:
        """Mark as revoked (don't actually delete)."""
        return self.update(delegation_id, {
            "revoked": True,
            "revoked_at": datetime.utcnow().isoformat(),
        }) is not None

    def get_children(self, parent_id: str) -> list[dict[str, Any]]:
        return [d for d in self._load()
                if d.get("parent_id") == parent_id and not d.get("revoked")]
