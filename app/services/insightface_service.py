from pathlib import Path
import logging

import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from PIL import Image

from app.core.config import get_settings
from app.core.device import resolve_device


logger = logging.getLogger(__name__)


class InsightFaceService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_dir = Path(self.settings.insightface_model_dir)
        self._app: FaceAnalysis | None = None

    def load(self) -> None:
        device = resolve_device(self.settings.inference_device)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        providers = self._resolve_ort_providers(device)
        self._app = FaceAnalysis(
            name=self.settings.insightface_model_name,
            root=str(self.model_dir),
            providers=providers,
        )
        ctx_id = 0 if device == "cuda" else -1
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info(
            "InsightFace initialized (device=%s, providers=%s, available=%s)",
            device,
            providers,
            ort.get_available_providers(),
        )

    def _resolve_ort_providers(self, device: str) -> list[str]:
        available = set(ort.get_available_providers())
        if device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if device == "cuda":
            logger.warning(
                "CUDA requested but CUDAExecutionProvider is unavailable. Falling back to CPU providers: %s",
                sorted(available),
            )

        if "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

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
