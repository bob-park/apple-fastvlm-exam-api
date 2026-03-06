import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_filename: Mapped[str] = mapped_column(String(512))
    stored_path: Mapped[str] = mapped_column(String(1024), unique=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(64), default="uploaded")
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    catalog_images: Mapped[list["CatalogImage"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    text_segments: Mapped[list["TextSegment"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    face_tracks: Mapped[list["FaceTrack"]] = relationship(back_populates="video", cascade="all, delete-orphan")


class CatalogImage(Base):
    __tablename__ = "catalog_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    timestamp_sec: Mapped[int] = mapped_column(Integer, index=True)
    image_path: Mapped[str] = mapped_column(String(1024), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped[Video] = relationship(back_populates="catalog_images")
    captions: Mapped[list["ImageCaption"]] = relationship(back_populates="catalog_image", cascade="all, delete-orphan")
    detected_faces: Mapped[list["DetectedFace"]] = relationship(back_populates="catalog_image", cascade="all, delete-orphan")


class ImageCaption(Base):
    __tablename__ = "image_captions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    catalog_image_id: Mapped[int] = mapped_column(ForeignKey("catalog_images.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    catalog_image: Mapped[CatalogImage] = relationship(back_populates="captions")


class TextSegment(Base):
    __tablename__ = "text_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    start_sec: Mapped[int] = mapped_column(Integer)
    end_sec: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped[Video] = relationship(back_populates="text_segments")


class FaceTrack(Base):
    __tablename__ = "face_tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    alias: Mapped[str] = mapped_column(String(128), index=True)
    start_sec: Mapped[int] = mapped_column(Integer)
    end_sec: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped[Video] = relationship(back_populates="face_tracks")
    detected_faces: Mapped[list["DetectedFace"]] = relationship(back_populates="face_track")


class DetectedFace(Base):
    __tablename__ = "detected_faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    catalog_image_id: Mapped[int] = mapped_column(ForeignKey("catalog_images.id", ondelete="CASCADE"), index=True)
    face_track_id: Mapped[int | None] = mapped_column(ForeignKey("face_tracks.id", ondelete="SET NULL"), index=True)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    embedding: Mapped[list[float]] = mapped_column(Vector(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    catalog_image: Mapped[CatalogImage] = relationship(back_populates="detected_faces")
    face_track: Mapped[FaceTrack | None] = relationship(back_populates="detected_faces")
