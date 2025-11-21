from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from contextlib import asynccontextmanager
from .config import settings
from .logging_config import setup_logging, get_logger
from .routes import llm
from fastapi_app.routes import documents

# Setup logging
setup_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    log_dir=settings.log_dir,
)

# Get logger for this module
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app startup and shutdown events
    """
    # Startup: Perform health checks for all LLM providers
    logger.info("Starting application - performing health checks for LLM providers")
    try:
        from .routes.llm import perform_health_checks_for_all_providers

        await perform_health_checks_for_all_providers()
        logger.info("Health checks completed for all LLM providers")
    except Exception as e:
        logger.error(f"Failed to perform health checks on startup: {e}", exc_info=True)

    yield

    # Shutdown: Cleanup WebSocket connections and other resources
    logger.info("Shutting down application")
    try:
        from .routes.llm import close_all_websockets

        await close_all_websockets()
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}", exc_info=True)


app = FastAPI(title="FastAPI LLM App", version="1.0.0", lifespan=lifespan)

# Include LLM routes
app.include_router(llm.router)
app.include_router(documents.router, prefix="/api/documents")

# Serve static files (HTML UI)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Serve the LLM UI"""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "Welcome to FastAPI LLM App", "ui": "/static/index.html"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from .routes.llm import llm_health

    # Check LLM health (without performing actual API calls)
    llm_health_result = await llm_health(provider=None, check=False)

    # Determine overall status
    overall_status = "healthy"
    if llm_health_result.get("status") == "unhealthy":
        overall_status = "unhealthy"
    elif llm_health_result.get("status") == "not_configured":
        overall_status = "degraded"

    return {
        "status": overall_status,
        "llm": llm_health_result,
    }
