import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.face import router as face_router
from app.api.videos import router as videos_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db import init_db
from app.services.app_state import set_services
from app.services.service_registry import ServiceRegistry


settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Initializing database and model services")
    await init_db()
    services = ServiceRegistry()
    await asyncio.to_thread(services.load_models)
    set_services(services)
    logger.info("Initialization completed")
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(videos_router)
app.include_router(face_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
