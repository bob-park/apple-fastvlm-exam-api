import uuid

from fastapi import APIRouter, File, UploadFile

from app.schemas import FaceBox, FaceDetectResponse
from app.services.app_state import get_services


router = APIRouter(prefix="/face", tags=["face"])


@router.post("/detect", response_model=FaceDetectResponse)
async def detect_face(file: UploadFile = File(...)) -> FaceDetectResponse:
    services = get_services()
    payload = await file.read()
    saved_path = await services.storage.save_image_bytes(f"face_detect_{uuid.uuid4()}.jpg", payload)
    faces = services.insightface.detect_faces(saved_path)
    boxes = [
        FaceBox(x1=face["bbox"][0], y1=face["bbox"][1], x2=face["bbox"][2], y2=face["bbox"][3])
        for face in faces
    ]
    return FaceDetectResponse(faces=boxes)
