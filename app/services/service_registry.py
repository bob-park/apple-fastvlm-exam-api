from app.services.embedding_service import EmbeddingService
from app.services.fastvlm_service import FastVLMService
from app.services.ffmpeg_service import FFmpegService
from app.services.insightface_service import InsightFaceService
from app.services.storage import StorageService


class ServiceRegistry:
    def __init__(self) -> None:
        self.storage = StorageService()
        self.ffmpeg = FFmpegService()
        self.embedder = EmbeddingService()
        self.fastvlm = FastVLMService()
        self.insightface = InsightFaceService()

    def load_models(self) -> None:
        self.embedder.load()
        self.fastvlm.load()
        self.insightface.load()
