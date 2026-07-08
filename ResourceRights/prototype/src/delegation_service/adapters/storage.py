"""Storage adapter using Globus for ACL management."""

import logging
from datetime import datetime

from globus_sdk import ConfidentialAppAuthClient, TransferClient
from globus_sdk.authorizers import ClientCredentialsAuthorizer

from delegation_service.adapters.base import ResourceAdapter
from delegation_service.config import settings

logger = logging.getLogger(__name__)

TRANSFER_SCOPE = "urn:globus:auth:scope:transfer.api.globus.org:all"


class StorageAdapter(ResourceAdapter):
    """Manages Globus ACLs for storage delegations."""

    def __init__(self) -> None:
        self._transfer_client: TransferClient | None = None

    def _get_transfer_client(self) -> TransferClient:
        if self._transfer_client is None:
            auth_client = ConfidentialAppAuthClient(
                settings.globus_client_id,
                settings.globus_client_secret,
            )
            authorizer = ClientCredentialsAuthorizer(auth_client, TRANSFER_SCOPE)
            self._transfer_client = TransferClient(authorizer=authorizer)
        return self._transfer_client

    async def provision(
        self,
        grantee: str,
        resource_id: str,
        scope: dict,
        expires_at: datetime | None,
    ) -> str:
        """Create a Globus ACL rule for the grantee."""
        path = scope.get("path", "/")
        operations = scope.get("operations", ["read"])
        permissions = "rw" if "write" in operations else "r"

        if settings.globus_dry_run:
            acl_id = f"dry-run-acl-{grantee[-8:]}"
            logger.info(
                f"[DRY RUN] Would create ACL: collection={resource_id}, "
                f"principal={grantee}, path={path}, permissions={permissions}"
            )
            return acl_id

        tc = self._get_transfer_client()

        rule_data = {
            "DATA_TYPE": "access",
            "principal_type": "identity",
            "principal": self._extract_identity_id(grantee),
            "path": path,
            "permissions": permissions,
        }

        result = tc.add_endpoint_acl_rule(resource_id, rule_data)
        acl_id = result["access_id"]
        logger.info(f"Created ACL {acl_id} on {resource_id} for {grantee}")
        return acl_id

    async def update(
        self,
        enforcement_ref: str,
        resource_id: str,
        suspended: bool,
    ) -> None:
        """Update ACL permissions (e.g., downgrade to read-only on suspension)."""
        permissions = "r" if suspended else "rw"

        if settings.globus_dry_run:
            logger.info(
                f"[DRY RUN] Would update ACL {enforcement_ref}: permissions={permissions}"
            )
            return

        tc = self._get_transfer_client()
        tc.update_endpoint_acl_rule(resource_id, enforcement_ref, {"permissions": permissions})
        logger.info(f"Updated ACL {enforcement_ref} to permissions={permissions}")

    async def revoke(self, enforcement_ref: str, resource_id: str) -> None:
        """Delete the Globus ACL rule."""
        if settings.globus_dry_run:
            logger.info(f"[DRY RUN] Would delete ACL {enforcement_ref} from {resource_id}")
            return

        tc = self._get_transfer_client()
        tc.delete_endpoint_acl_rule(resource_id, enforcement_ref)
        logger.info(f"Deleted ACL {enforcement_ref} from {resource_id}")

    @staticmethod
    def _extract_identity_id(grantee: str) -> str:
        """Extract the UUID from a Globus identity URN."""
        if grantee.startswith("urn:globus:auth:identity:"):
            return grantee.replace("urn:globus:auth:identity:", "")
        return grantee
