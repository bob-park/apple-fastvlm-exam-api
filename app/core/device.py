import logging

import torch


logger = logging.getLogger(__name__)


def resolve_device(preferred: str) -> str:
    device = preferred.lower().strip()
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError("INFERENCE_DEVICE must be one of: auto, cpu, gpu")

    if device == "cpu":
        return "cpu"

    cuda_available = torch.cuda.is_available()
    if device == "gpu":
        if not cuda_available:
            logger.warning("INFERENCE_DEVICE=gpu but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        return "cuda"

    return "cuda" if cuda_available else "cpu"
