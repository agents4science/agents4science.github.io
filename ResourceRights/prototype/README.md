# Delegation Service

Resource rights delegation service for scientific cyberinfrastructure.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests (uses in-memory SQLite, no database setup needed)
pytest

# Run the service (requires PostgreSQL)
export DELEGATION_DATABASE_URL="postgresql+asyncpg://localhost/delegation_service"
export DELEGATION_GLOBUS_DRY_RUN=true  # Skip actual Globus calls
uvicorn delegation_service.main:app --reload
```

## API

Base URL: `http://localhost:8000/v1`

### Create Delegation

```bash
curl -X POST http://localhost:8000/v1/delegations \
  -H "Content-Type: application/json" \
  -H "X-Caller-Identity: urn:globus:auth:identity:pi-uuid" \
  -d '{
    "grantee": "urn:globus:auth:identity:agent-uuid",
    "resource_type": "storage",
    "resource_id": "eagle-collection-uuid",
    "scope": {"path": "/projects/myproject", "operations": ["read", "write"]},
    "quota": {"bytes": 1099511627776},
    "expires_at": "2026-12-01T00:00:00Z"
  }'
```

### List Delegations

```bash
curl http://localhost:8000/v1/delegations \
  -H "X-Caller-Identity: urn:globus:auth:identity:pi-uuid"
```

### Get Delegation

```bash
curl http://localhost:8000/v1/delegations/{delegation_id} \
  -H "X-Caller-Identity: urn:globus:auth:identity:pi-uuid"
```

### Update Delegation

```bash
curl -X PATCH http://localhost:8000/v1/delegations/{delegation_id} \
  -H "Content-Type: application/json" \
  -H "X-Caller-Identity: urn:globus:auth:identity:pi-uuid" \
  -d '{"quota": {"bytes": 2199023255552}}'
```

### Revoke Delegation

```bash
curl -X DELETE http://localhost:8000/v1/delegations/{delegation_id} \
  -H "X-Caller-Identity: urn:globus:auth:identity:pi-uuid"
```

## Configuration

Environment variables (prefix: `DELEGATION_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://localhost/delegation_service` | Database connection string |
| `GLOBUS_CLIENT_ID` | (empty) | Globus application client ID |
| `GLOBUS_CLIENT_SECRET` | (empty) | Globus application client secret |
| `GLOBUS_DRY_RUN` | `true` | Skip actual Globus API calls |
| `API_PREFIX` | `/v1` | API route prefix |

## Development Status

MVP implementation with:
- [x] Core data model
- [x] CRUD API for delegations
- [x] Storage adapter (Globus ACL provisioning)
- [x] Basic auth via X-Caller-Identity header
- [ ] Globus Auth token introspection
- [ ] Hierarchical delegation validation
- [ ] Compute adapter
- [ ] Webhook endpoints for consumption tracking
- [ ] Background workers (expiration, quota alerts)
