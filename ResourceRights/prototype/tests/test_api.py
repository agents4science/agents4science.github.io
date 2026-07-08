"""Basic API tests using in-memory SQLite with async support."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from delegation_service.database import get_db
from delegation_service.main import app
from delegation_service.models.database import Base

SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite://"


@pytest.fixture
async def db_session():
    engine = create_async_engine(SQLALCHEMY_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def client(db_session):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_create_delegation(client):
    response = await client.post(
        "/v1/delegations",
        json={
            "grantee": "urn:globus:auth:identity:agent-123",
            "resource_type": "storage",
            "resource_id": "eagle-collection-uuid",
            "scope": {"path": "/projects/test", "operations": ["read", "write"]},
            "quota": {"bytes": 1099511627776},
        },
        headers={"X-Caller-Identity": "urn:globus:auth:identity:pi-456"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["grantee"] == "urn:globus:auth:identity:agent-123"
    assert data["delegator"] == "urn:globus:auth:identity:pi-456"
    assert data["resource_type"] == "storage"
    assert data["quota"] == {"bytes": 1099511627776}
    assert data["consumed"] == {"bytes": 0}
    assert data["suspended"] is False
    assert data["revoked"] is False


async def test_list_delegations(client):
    # Create a delegation first
    await client.post(
        "/v1/delegations",
        json={
            "grantee": "urn:globus:auth:identity:agent-123",
            "resource_type": "storage",
            "resource_id": "eagle-uuid",
            "scope": {"path": "/test", "operations": ["read"]},
        },
        headers={"X-Caller-Identity": "urn:globus:auth:identity:pi-456"},
    )

    response = await client.get(
        "/v1/delegations",
        headers={"X-Caller-Identity": "urn:globus:auth:identity:pi-456"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1


async def test_get_delegation(client):
    # Create a delegation first
    create_response = await client.post(
        "/v1/delegations",
        json={
            "grantee": "urn:globus:auth:identity:agent-123",
            "resource_type": "storage",
            "resource_id": "eagle-uuid",
            "scope": {"path": "/test", "operations": ["read"]},
        },
        headers={"X-Caller-Identity": "urn:globus:auth:identity:pi-456"},
    )
    delegation_id = create_response.json()["delegation_id"]

    response = await client.get(
        f"/v1/delegations/{delegation_id}",
        headers={"X-Caller-Identity": "urn:globus:auth:identity:pi-456"},
    )
    assert response.status_code == 200
    assert response.json()["delegation_id"] == delegation_id


async def test_revoke_delegation(client):
    # Create a delegation first
    create_response = await client.post(
        "/v1/delegations",
        json={
            "grantee": "urn:globus:auth:identity:agent-123",
            "resource_type": "storage",
            "resource_id": "eagle-uuid",
            "scope": {"path": "/test", "operations": ["read"]},
        },
        headers={"X-Caller-Identity": "urn:globus:auth:identity:pi-456"},
    )
    delegation_id = create_response.json()["delegation_id"]

    # Revoke it
    response = await client.delete(
        f"/v1/delegations/{delegation_id}",
        headers={"X-Caller-Identity": "urn:globus:auth:identity:pi-456"},
    )
    assert response.status_code == 204

    # Verify it's revoked
    get_response = await client.get(
        f"/v1/delegations/{delegation_id}",
        headers={"X-Caller-Identity": "urn:globus:auth:identity:pi-456"},
    )
    assert get_response.json()["revoked"] is True


async def test_missing_auth(client):
    response = await client.post(
        "/v1/delegations",
        json={
            "grantee": "urn:globus:auth:identity:agent-123",
            "resource_type": "storage",
            "resource_id": "eagle-uuid",
            "scope": {"path": "/test", "operations": ["read"]},
        },
    )
    assert response.status_code == 401
