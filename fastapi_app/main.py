from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from contextlib import asynccontextmanager
from fastapi_app.config import settings
from fastapi_app.logging_config import setup_logging, get_logger
from fastapi_app.routes import llm
from fastapi_app.routes import documents

# Setup logging
setup_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    log_dir=settings.log_dir,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting application")
    
    # Initialize Redis document store
    try:
        await documents.document_store.connect()
        logger.info("Redis document store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
    
    # Perform LLM health checks
    try:
        from fastapi_app.routes.llm import perform_health_checks_for_all_providers
        await perform_health_checks_for_all_providers()
        logger.info("Health checks completed")
    except Exception as e:
        logger.error(f"Failed to perform health checks: {e}", exc_info=True)

    yield

    # Shutdown
    logger.info("Shutting down application")
    
    # Disconnect Redis
    try:
        await documents.document_store.disconnect()
        logger.info("Redis disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting Redis: {e}")
    
    # Close WebSocket connections
    try:
        from fastapi_app.routes.llm import close_all_websockets
        await close_all_websockets()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


app = FastAPI(title="FastAPI LLM App", version="1.0.0", lifespan=lifespan)

# Include routes
app.include_router(llm.router)
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Serve the UI"""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "Welcome to FastAPI LLM App"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from fastapi_app.routes.llm import llm_health

    llm_health_result = await llm_health(provider=None, check=False)

    overall_status = "healthy"
    if llm_health_result.get("status") == "unhealthy":
        overall_status = "unhealthy"
    elif llm_health_result.get("status") == "not_configured":
        overall_status = "degraded"

    return {
        "status": overall_status,
        "llm": llm_health_result,
    }