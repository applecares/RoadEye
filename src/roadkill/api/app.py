"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from roadkill.dashboard import router as dashboard_router
from roadkill.database import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    logger.info("Initialising database...")
    init_db()
    logger.info("Database initialised")
    yield
    # Shutdown
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RoadEye",
        description="Wildlife roadkill detection and mapping platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Mount static files
    static_dir = Path(__file__).parent.parent / "dashboard" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include routers
    app.include_router(dashboard_router)

    return app


# Create default app instance
app = create_app()
