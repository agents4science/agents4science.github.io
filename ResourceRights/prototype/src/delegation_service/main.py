"""FastAPI application entry point."""

import logging

from fastapi import FastAPI

from delegation_service.api import delegations_router
from delegation_service.config import settings

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Delegation Service",
    description="Resource rights delegation service for scientific cyberinfrastructure",
    version="0.1.0",
)

app.include_router(delegations_router, prefix=settings.api_prefix)


@app.get("/health")
async def health():
    return {"status": "ok"}
