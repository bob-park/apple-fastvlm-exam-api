from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.device import resolve_device


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        local_dir = Path(self.settings.text_embed_local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        device = resolve_device(self.settings.inference_device)
        self._model = SentenceTransformer(
            self.settings.text_embed_model_id,
            cache_folder=str(local_dir),
            device=device,
        )

    def embed_text(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError("Embedding model not initialized")
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()
