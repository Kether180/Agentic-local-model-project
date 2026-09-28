"""App factory."""

from core.logging import setup_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config.settings import get_api_settings
from api.routers import documents_router, responses_router, search_router
from api.routers.openapi import DESCRIPTION, TAGS


def create_app() -> FastAPI:
    setup_logging()
    settings = get_api_settings()

    app = FastAPI(
        title=settings.title,
        summary="Ingest PDFs, then search or ask questions about them.",
        description=DESCRIPTION,
        openapi_tags=TAGS,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents_router)
    app.include_router(responses_router)
    app.include_router(search_router)

    @app.get(
        "/health",
        tags=["health"],
        summary="Liveness probe",
        response_description='Always `{"status": "ok"}` when serving',
    )
    def health() -> dict[str, str]:
        """Whether the API process is up.

        Does not check Postgres, RabbitMQ, MinIO or the model provider — compose has its own
        healthchecks for those, and this one is used to decide whether the app itself is serving.
        """
        return {"status": "ok"}

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
