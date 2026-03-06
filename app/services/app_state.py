from app.services.pipeline_service import VideoPipelineService
from app.services.service_registry import ServiceRegistry

_services: ServiceRegistry | None = None
_pipeline: VideoPipelineService | None = None


def set_services(services: ServiceRegistry) -> None:
    global _services, _pipeline
    _services = services
    _pipeline = VideoPipelineService(services)


def get_services() -> ServiceRegistry:
    if _services is None:
        raise RuntimeError("Services are not initialized")
    return _services


def get_pipeline() -> VideoPipelineService:
    if _pipeline is None:
        raise RuntimeError("Pipeline service is not initialized")
    return _pipeline
