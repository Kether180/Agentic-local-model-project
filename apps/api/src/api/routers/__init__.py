from api.routers.documents import router as documents_router
from api.routers.responses import router as responses_router
from api.routers.search import router as search_router

__all__ = ["documents_router", "responses_router", "search_router"]
