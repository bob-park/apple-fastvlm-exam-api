import asyncio
import mimetypes
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import get_db_session
from app.models import CatalogImage, Face, FaceTrack, TextSegment, Video
from app.schemas import (
    CatalogImageItem,
    CatalogImageListResponse,
    FaceTrackItem,
    FaceTrackUpdateRequest,
    FaceSearchItem,
    FaceSearchResponse,
    TextSegmentItem,
    TextSearchItem,
    TextSearchRequest,
    TextSearchResponse,
    VideoDetailResponse,
    VideoIngestResponse,
    VideoItem,
    VideoListResponse,
    VideoSummary,
)
from app.services.app_state import get_pipeline, get_services


router = APIRouter(prefix="/videos", tags=["videos"])
settings = get_settings()


def _parse_range_header(range_header: str, file_size: int) -> Optional[Tuple[int, int]]:
    if not range_header:
        return None
    if not range_header.startswith("bytes="):
        return None
    try:
        start_str, end_str = range_header.replace("bytes=", "", 1).split("-", 1)
    except ValueError:
        return None
    if start_str == "" and end_str == "":
        return None
    if start_str == "":
        try:
            suffix_length = int(end_str)
        except ValueError:
            return None
        if suffix_length <= 0:
            return None
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
    else:
        try:
            start = int(start_str)
            end = int(end_str) if end_str != "" else file_size - 1
        except ValueError:
            return None
    if start >= file_size or end < start:
        return None
    return start, min(end, file_size - 1)


async def _iter_file_range(path: Path, start: int, end: int, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    remaining = end - start + 1
    async with aiofiles.open(path, "rb") as file_handle:
        await file_handle.seek(start)
        while remaining > 0:
            chunk = await file_handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _pick_catalog_image_id(
    catalog_images: list[CatalogImage],
    start_sec: int,
    end_sec: int,
) -> int | None:
    if not catalog_images:
        return None
    in_range = [image for image in catalog_images if start_sec <= image.timestamp_sec <= end_sec]
    if in_range:
        return in_range[0].id
    target = start_sec
    best = min(catalog_images, key=lambda image: abs(image.timestamp_sec - target))
    return best.id


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


@router.get("/{video_id}", response_model=VideoDetailResponse)
async def get_video_detail(
    video_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> VideoDetailResponse:
    video = await session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    catalog_image_count = await session.scalar(
        select(func.count()).select_from(CatalogImage).where(CatalogImage.video_id == video_id)
    )
    text_segment_count = await session.scalar(
        select(func.count()).select_from(TextSegment).where(TextSegment.video_id == video_id)
    )
    face_track_count = await session.scalar(
        select(func.count()).select_from(FaceTrack).where(FaceTrack.video_id == video_id)
    )
    face_tracks = (
        await session.execute(
            select(FaceTrack)
            .where(FaceTrack.video_id == video_id)
            .order_by(FaceTrack.start_sec.asc())
        )
    ).scalars().all()
    text_segments = (
        await session.execute(
            select(TextSegment)
            .where(TextSegment.video_id == video_id)
            .order_by(TextSegment.start_sec.asc())
        )
    ).scalars().all()
    face_ids = {track.face_id for track in face_tracks}
    faces = (
        await session.execute(select(Face).where(Face.id.in_(face_ids)))
    ).scalars().all() if face_ids else []
    face_lookup = {face.id: face for face in faces}
    catalog_images = (
        await session.execute(
            select(CatalogImage)
            .where(CatalogImage.video_id == video_id)
            .order_by(CatalogImage.timestamp_sec.asc())
        )
    ).scalars().all()

    return VideoDetailResponse(
        id=video.id,
        original_filename=video.original_filename,
        duration_seconds=video.duration_seconds,
        status=video.status,
        processing_started_at=video.processing_started_at,
        processing_finished_at=video.processing_finished_at,
        created_at=video.created_at,
        stream_url=str(request.url_for("stream_video", video_id=str(video_id))),
        catalog_image_count=int(catalog_image_count or 0),
        text_segment_count=int(text_segment_count or 0),
        face_track_count=int(face_track_count or 0),
        face_tracks=[
            FaceTrackItem(
                id=track.id,
                face_id=track.face_id,
                alias=face_lookup.get(track.face_id).alias if track.face_id in face_lookup else "unknown",
                start_sec=track.start_sec,
                end_sec=track.end_sec,
                created_at=track.created_at,
                image_url=(
                    str(
                        request.url_for(
                            "get_catalog_image",
                            video_id=str(video_id),
                            image_id=str(image_id),
                        )
                    )
                    if (image_id := _pick_catalog_image_id(catalog_images, track.start_sec, track.end_sec))
                    else None
                ),
            )
            for track in face_tracks
        ],
        text_segments=[
            TextSegmentItem(
                id=segment.id,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                text=segment.text,
                created_at=segment.created_at,
                image_url=(
                    str(
                        request.url_for(
                            "get_catalog_image",
                            video_id=str(video_id),
                            image_id=str(image_id),
                        )
                    )
                    if (image_id := _pick_catalog_image_id(catalog_images, segment.start_sec, segment.end_sec))
                    else None
                ),
            )
            for segment in text_segments
        ],
    )


@router.patch("/{video_id}/faces/{face_track_id}", response_model=FaceTrackItem)
async def update_face_track_alias(
    video_id: uuid.UUID,
    face_track_id: int,
    payload: FaceTrackUpdateRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> FaceTrackItem:
    track = (
        await session.execute(
            select(FaceTrack).where(
                FaceTrack.id == face_track_id,
                FaceTrack.video_id == video_id,
            )
        )
    ).scalars().first()
    if not track:
        raise HTTPException(status_code=404, detail="Face track not found")

    face = await session.get(Face, track.face_id)
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    face.alias = payload.alias
    await session.commit()
    await session.refresh(face)

    catalog_images = (
        await session.execute(
            select(CatalogImage)
            .where(CatalogImage.video_id == video_id)
            .order_by(CatalogImage.timestamp_sec.asc())
        )
    ).scalars().all()
    image_id = _pick_catalog_image_id(catalog_images, track.start_sec, track.end_sec)
    return FaceTrackItem(
        id=track.id,
        face_id=track.face_id,
        alias=face.alias,
        start_sec=track.start_sec,
        end_sec=track.end_sec,
        created_at=track.created_at,
        image_url=(
            str(
                request.url_for(
                    "get_catalog_image",
                    video_id=str(video_id),
                    image_id=str(image_id),
                )
            )
            if image_id
            else None
        ),
    )


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    video = await session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(video.stored_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range") or ""
    byte_range = _parse_range_header(range_header, file_size)
    content_type, _ = mimetypes.guess_type(str(video_path))
    media_type = content_type or "application/octet-stream"

    if byte_range is None and range_header:
        return Response(
            status_code=416,
            headers={
                "Content-Range": f"bytes */{file_size}",
                "Accept-Ranges": "bytes",
            },
        )

    if byte_range is None:
        return FileResponse(
            path=video_path,
            media_type=media_type,
            filename=video.original_filename,
            headers={"Accept-Ranges": "bytes"},
        )

    start, end = byte_range
    content_length = end - start + 1
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
    }
    return StreamingResponse(
        _iter_file_range(video_path, start, end),
        status_code=206,
        media_type=media_type,
        headers=headers,
    )


@router.get("/{video_id}/catalog-images", response_model=CatalogImageListResponse)
async def list_catalog_images(
    video_id: uuid.UUID,
    request: Request,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=settings.page_size_default, ge=1),
    session: AsyncSession = Depends(get_db_session),
) -> CatalogImageListResponse:
    size = min(size, settings.page_size_max)
    offset = (page - 1) * size

    video = await session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    total = await session.scalar(
        select(func.count()).select_from(CatalogImage).where(CatalogImage.video_id == video_id)
    )
    rows = (
        await session.execute(
            select(CatalogImage)
            .where(CatalogImage.video_id == video_id)
            .order_by(CatalogImage.timestamp_sec.asc())
            .offset(offset)
            .limit(size)
        )
    ).scalars().all()

    items = [
        CatalogImageItem(
            id=row.id,
            timestamp_sec=row.timestamp_sec,
            created_at=row.created_at,
            image_url=str(
                request.url_for(
                    "get_catalog_image",
                    video_id=str(video_id),
                    image_id=str(row.id),
                )
            ),
        )
        for row in rows
    ]
    return CatalogImageListResponse(page=page, size=size, total=int(total or 0), items=items)


@router.get("/{video_id}/catalog-images/{image_id}", name="get_catalog_image")
async def get_catalog_image(
    video_id: uuid.UUID,
    image_id: int,
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    image = (
        await session.execute(
            select(CatalogImage).where(
                CatalogImage.id == image_id,
                CatalogImage.video_id == video_id,
            )
        )
    ).scalars().first()
    if not image:
        raise HTTPException(status_code=404, detail="Catalog image not found")

    image_path = Path(image.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Catalog image file missing")

    content_type, _ = mimetypes.guess_type(str(image_path))
    media_type = content_type or "image/jpeg"
    return FileResponse(path=image_path, media_type=media_type, filename=image_path.name)


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
    distance = Face.embedding.cosine_distance(query_embedding).label("distance")
    rows = (
        await session.execute(
            select(FaceTrack, Video, Face, distance)
            .join(Face, Face.id == FaceTrack.face_id)
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
            alias=face.alias,
            start_sec=track.start_sec,
            end_sec=track.end_sec,
            similarity=max(0.0, 1.0 - float(distance_value)),
        )
        for track, video, face, distance_value in rows
    ]
    return FaceSearchResponse(items=items)
