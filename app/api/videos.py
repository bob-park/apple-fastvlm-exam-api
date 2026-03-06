import asyncio
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import get_db_session
from app.models import FaceTrack, TextSegment, Video
from app.schemas import (
    FaceSearchItem,
    FaceSearchResponse,
    TextSearchItem,
    TextSearchRequest,
    TextSearchResponse,
    VideoIngestResponse,
    VideoItem,
    VideoListResponse,
    VideoSummary,
)
from app.services.app_state import get_pipeline, get_services


router = APIRouter(prefix="/videos", tags=["videos"])
settings = get_settings()


@router.post("/ingest", response_model=VideoIngestResponse, status_code=202)
async def ingest_video(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
) -> VideoIngestResponse:
    services = get_services()
    stored_path = await services.storage.save_video(file)

    video = Video(
        original_filename=file.filename or stored_path.name,
        stored_path=str(stored_path),
        status="uploaded",
    )
    session.add(video)
    await session.commit()
    await session.refresh(video)

    pipeline = get_pipeline()
    asyncio.create_task(pipeline.process_video(video.id))

    return VideoIngestResponse(
        id=video.id,
        original_filename=video.original_filename,
        status=video.status,
        created_at=video.created_at,
    )


@router.get("", response_model=VideoListResponse)
async def list_videos(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=settings.page_size_default, ge=1),
    session: AsyncSession = Depends(get_db_session),
) -> VideoListResponse:
    size = min(size, settings.page_size_max)
    offset = (page - 1) * size

    total = await session.scalar(select(func.count()).select_from(Video))
    rows = (
        await session.execute(
            select(Video)
            .order_by(Video.created_at.desc())
            .offset(offset)
            .limit(size)
        )
    ).scalars().all()

    items = [
        VideoItem(
            id=row.id,
            original_filename=row.original_filename,
            duration_seconds=row.duration_seconds,
            status=row.status,
            processing_started_at=row.processing_started_at,
            processing_finished_at=row.processing_finished_at,
            created_at=row.created_at,
        )
        for row in rows
    ]
    return VideoListResponse(page=page, size=size, total=int(total or 0), items=items)


@router.post("/texts", response_model=TextSearchResponse)
async def search_text_segments(
    payload: TextSearchRequest,
    session: AsyncSession = Depends(get_db_session),
) -> TextSearchResponse:
    services = get_services()
    query_embedding = services.embedder.embed_text(payload.query)

    distance = TextSegment.embedding.cosine_distance(query_embedding).label("distance")
    rows = (
        await session.execute(
            select(TextSegment, Video, distance)
            .join(Video, Video.id == TextSegment.video_id)
            .where(distance <= (1.0 - payload.threshold))
            .order_by(distance)
            .limit(payload.limit)
        )
    ).all()

    items = []
    for segment, video, distance_value in rows:
        similarity = max(0.0, 1.0 - float(distance_value))
        items.append(
            TextSearchItem(
                video_id=segment.video_id,
                video=VideoSummary(
                    id=video.id,
                    original_filename=video.original_filename,
                    duration_seconds=video.duration_seconds,
                    status=video.status,
                    created_at=video.created_at,
                ),
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                text=segment.text,
                similarity=similarity,
            )
        )
    return TextSearchResponse(items=items)


@router.post("/faces", response_model=FaceSearchResponse)
async def search_faces(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.7, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=200),
    session: AsyncSession = Depends(get_db_session),
) -> FaceSearchResponse:
    services = get_services()
    payload = await file.read()
    query_path = await services.storage.save_image_bytes(f"face_search_{uuid.uuid4()}.jpg", payload)
    faces = services.insightface.detect_faces(query_path)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected from query image")

    query_embedding = faces[0]["embedding"]
    distance = FaceTrack.embedding.cosine_distance(query_embedding).label("distance")
    rows = (
        await session.execute(
            select(FaceTrack, Video, distance)
            .join(Video, Video.id == FaceTrack.video_id)
            .where(distance <= (1.0 - threshold))
            .order_by(distance)
            .limit(limit)
        )
    ).all()

    items = [
        FaceSearchItem(
            video_id=track.video_id,
            video=VideoSummary(
                id=video.id,
                original_filename=video.original_filename,
                duration_seconds=video.duration_seconds,
                status=video.status,
                created_at=video.created_at,
            ),
            alias=track.alias,
            start_sec=track.start_sec,
            end_sec=track.end_sec,
            similarity=max(0.0, 1.0 - float(distance_value)),
        )
        for track, video, distance_value in rows
    ]
    return FaceSearchResponse(items=items)
