import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class VideoIngestResponse(BaseModel):
    id: uuid.UUID
    original_filename: str
    status: str
    created_at: datetime


class VideoItem(BaseModel):
    id: uuid.UUID
    original_filename: str
    duration_seconds: float | None
    status: str
    processing_started_at: datetime | None
    processing_finished_at: datetime | None
    created_at: datetime


class FaceTrackItem(BaseModel):
    id: int
    face_id: int
    alias: str
    start_sec: int
    end_sec: int
    created_at: datetime
    image_url: str | None


class FaceTrackUpdateRequest(BaseModel):
    alias: str = Field(min_length=1, max_length=128)


class TextSegmentItem(BaseModel):
    id: int
    start_sec: int
    end_sec: int
    text: str
    created_at: datetime
    image_url: str | None


class VideoDetailResponse(BaseModel):
    id: uuid.UUID
    original_filename: str
    duration_seconds: float | None
    status: str
    processing_started_at: datetime | None
    processing_finished_at: datetime | None
    created_at: datetime
    stream_url: str
    catalog_image_count: int
    text_segment_count: int
    face_track_count: int
    face_tracks: list[FaceTrackItem]
    text_segments: list[TextSegmentItem]


class VideoListResponse(BaseModel):
    page: int
    size: int
    total: int
    items: list[VideoItem]


class FaceBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class FaceMatchInfo(BaseModel):
    id: int
    alias: str
    similarity: float


class FaceDetectItem(BaseModel):
    box: FaceBox
    match: FaceMatchInfo | None


class FaceDetectResponse(BaseModel):
    faces: list[FaceDetectItem]


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=200)


class VideoSummary(BaseModel):
    id: uuid.UUID
    original_filename: str
    duration_seconds: float | None
    status: str
    created_at: datetime


class TextSearchItem(BaseModel):
    video_id: uuid.UUID
    video: VideoSummary
    start_sec: int
    end_sec: int
    text: str
    similarity: float


class TextSearchResponse(BaseModel):
    items: list[TextSearchItem]


class FaceSearchItem(BaseModel):
    video_id: uuid.UUID
    video: VideoSummary
    alias: str
    start_sec: int
    end_sec: int
    similarity: float


class FaceSearchResponse(BaseModel):
    items: list[FaceSearchItem]


class CatalogImageItem(BaseModel):
    id: int
    timestamp_sec: int
    created_at: datetime
    image_url: str


class CatalogImageListResponse(BaseModel):
    page: int
    size: int
    total: int
    items: list[CatalogImageItem]
