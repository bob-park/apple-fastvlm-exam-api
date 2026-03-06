import uuid

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.utils import cosine_similarity
from app.db import get_db_session
from app.models import Face
from app.schemas import FaceBox, FaceDetectItem, FaceDetectResponse, FaceMatchInfo
from app.services.app_state import get_services


router = APIRouter(prefix="/face", tags=["face"])
settings = get_settings()


@router.post("/detect", response_model=FaceDetectResponse)
async def detect_face(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
) -> FaceDetectResponse:
    services = get_services()
    payload = await file.read()
    saved_path = await services.storage.save_image_bytes(f"face_detect_{uuid.uuid4()}.jpg", payload)
    faces = services.insightface.detect_faces(saved_path)
    system_faces = (await session.execute(select(Face))).scalars().all()

    items: list[FaceDetectItem] = []
    for face in faces:
        bbox = face["bbox"]
        embedding = face["embedding"]
        match_info: FaceMatchInfo | None = None

        best_id = None
        best_alias = None
        best_score = 0.0
        for system_face in system_faces:
            score = cosine_similarity(system_face.embedding, embedding)
            if score >= settings.face_similarity_threshold and score > best_score:
                best_id = system_face.id
                best_alias = system_face.alias
                best_score = score
        if best_id is not None:
            match_info = FaceMatchInfo(id=best_id, alias=best_alias or "unknown", similarity=best_score)

        items.append(
            FaceDetectItem(
                box=FaceBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
                match=match_info,
            )
        )

    return FaceDetectResponse(faces=items)
