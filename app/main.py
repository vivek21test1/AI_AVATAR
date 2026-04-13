"""
Application factory.

`create_app()` is the single place where:
  • The FastAPI instance is constructed
  • Routers are registered (via the v1 aggregator)
  • Domain exception handlers are wired up
  • Logging is configured
  • Startup/shutdown hooks run

`app` at module level lets uvicorn discover it as `app.main:app`.
"""

from fastapi import FastAPI

from app.api.v1 import api_v1_router
from app.core.exceptions import register_exception_handlers
from app.core.logging_config import configure_logging


def create_app() -> FastAPI:
    configure_logging()

    application = FastAPI(
        title="AI Avatar Generation API",
        description=(
            "Generate 3D avatars from a single image using offline-capable models: "
            "TripoSR · Zero123++ · CRM · Wonder3D · InstantMesh · LAM.\n\n"
            "All model weights are downloaded once and cached locally — "
            "no network access is needed during inference."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Register all API routes
    application.include_router(api_v1_router)

    # Map domain exceptions → HTTP responses
    register_exception_handlers(application)

    return application


app = create_app()
