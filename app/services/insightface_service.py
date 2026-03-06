from pathlib import Path

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

from app.core.config import get_settings
from app.core.device import resolve_device


class InsightFaceService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_dir = Path(self.settings.insightface_model_dir)
        self._app: FaceAnalysis | None = None

    def load(self) -> None:
        device = resolve_device(self.settings.inference_device)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._app = FaceAnalysis(name=self.settings.insightface_model_name, root=str(self.model_dir))
        ctx_id = 0 if device == "cuda" else -1
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect_faces(self, image_path: Path) -> list[dict]:
        if self._app is None:
            raise RuntimeError("InsightFace model not initialized")
        image = np.array(Image.open(image_path).convert("RGB"))
        faces = self._app.get(image)
        result = []
        for face in faces:
            bbox = [float(value) for value in face.bbox.tolist()]
            embedding = [float(value) for value in face.embedding.tolist()]
            result.append({"bbox": bbox, "embedding": embedding})
        return result
