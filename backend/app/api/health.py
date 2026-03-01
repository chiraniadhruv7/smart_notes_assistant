"""
Health check endpoint for container orchestration.
"""

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Basic health check for load balancers and Docker health checks.
    Returns 200 if the API is running.
    """
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }
