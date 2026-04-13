"""
API v1 — aggregates all routers under /api/v1.
"""

from fastapi import APIRouter

from app.api.v1.routers import generate, health, jobs, models

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(health.router)
api_v1_router.include_router(models.router)
api_v1_router.include_router(generate.router)
api_v1_router.include_router(jobs.router)
