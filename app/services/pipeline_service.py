import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.core.utils import average_vectors, cosine_similarity
from app.db import SessionLocal
from app.models import CatalogImage, DetectedFace, FaceTrack, ImageCaption, TextSegment, Video
from app.services.service_registry import ServiceRegistry


logger = logging.getLogger(__name__)


@dataclass
class TrackBuffer:
    alias: str
    start_sec: int
    end_sec: int
    vectors: list[list[float]]

    @property
    def centroid(self) -> list[float]:
        return average_vectors(self.vectors)


class VideoPipelineService:
    def __init__(self, services: ServiceRegistry) -> None:
        self.settings = get_settings()
        self.services = services

    async def process_video(self, video_id: uuid.UUID) -> None:
        async with SessionLocal() as session:
            video = await session.get(Video, video_id)
            if video is None:
                logger.error("Video not found: %s", video_id)
                return

            video.status = "processing"
            video.processing_started_at = datetime.now(timezone.utc)
            await session.commit()

            try:
                await self._run_pipeline(session, video)
                video.status = "completed"
            except Exception:
                logger.exception("Video processing failed for %s", video_id)
                video.status = "failed"
            finally:
                video.processing_finished_at = datetime.now(timezone.utc)
                await session.commit()

    async def _run_pipeline(self, session, video: Video) -> None:
        video_path = Path(video.stored_path)
        frame_dir = self.services.storage.video_frame_dir(str(video.id))

        video.duration_seconds = self.services.ffmpeg.probe_duration(video_path)
        frames = self.services.ffmpeg.extract_frames_every_second(video_path, frame_dir)
        total_frames = len(frames)
        logger.debug("Catalog image extraction started: video_id=%s total_frames=%d", video.id, total_frames)
        await session.commit()

        catalog_images: list[CatalogImage] = []
        for idx, frame_path in enumerate(frames):
            item = CatalogImage(
                video_id=video.id,
                timestamp_sec=idx,
                image_path=str(frame_path),
            )
            session.add(item)
            catalog_images.append(item)
            logger.debug(
                "Catalog image extraction progress: video_id=%s done=%d/%d (%.2f%%) ts=%ds",
                video.id,
                idx + 1,
                total_frames,
                self._progress_percent(idx + 1, total_frames),
                idx,
            )
        await session.commit()
        logger.debug("Catalog image extraction completed: video_id=%s total_frames=%d", video.id, total_frames)

        await self._generate_text_segments(session, video.id, catalog_images)
        await self._extract_faces(session, video.id, catalog_images)

    async def _generate_text_segments(self, session, video_id: uuid.UUID, images: list[CatalogImage]) -> None:
        captions: list[tuple[CatalogImage, str, list[float]]] = []
        total_images = len(images)
        logger.debug("Catalog text generation started: video_id=%s total_images=%d", video_id, total_images)
        for idx, image in enumerate(images):
            text = self.services.fastvlm.describe_image(Path(image.image_path))
            embedding = self.services.embedder.embed_text(text)
            captions.append((image, text, embedding))
            session.add(ImageCaption(catalog_image_id=image.id, text=text, embedding=embedding))
            logger.debug(
                "Catalog text generation progress: video_id=%s done=%d/%d (%.2f%%) ts=%ds",
                video_id,
                idx + 1,
                total_images,
                self._progress_percent(idx + 1, total_images),
                image.timestamp_sec,
            )
        await session.commit()
        logger.debug("Catalog text generation completed: video_id=%s total_images=%d", video_id, total_images)

        threshold = self.settings.text_similarity_threshold
        buffer_items: list[tuple[int, str, list[float]]] = []
        for image, text, embedding in captions:
            if not buffer_items:
                buffer_items.append((image.timestamp_sec, text, embedding))
                continue

            current_centroid = average_vectors([item[2] for item in buffer_items])
            score = cosine_similarity(current_centroid, embedding)
            if score >= threshold and image.timestamp_sec <= buffer_items[-1][0] + 1:
                buffer_items.append((image.timestamp_sec, text, embedding))
            else:
                self._flush_text_segment(session, video_id, buffer_items)
                buffer_items = [(image.timestamp_sec, text, embedding)]

        if buffer_items:
            self._flush_text_segment(session, video_id, buffer_items)
        await session.commit()

    def _flush_text_segment(self, session, video_id: uuid.UUID, items: list[tuple[int, str, list[float]]]) -> None:
        timestamps = [item[0] for item in items]
        texts = [item[1] for item in items]
        vectors = [item[2] for item in items]
        segment = TextSegment(
            video_id=video_id,
            start_sec=min(timestamps),
            end_sec=max(timestamps),
            text=texts[0],
            embedding=average_vectors(vectors),
        )
        session.add(segment)

    async def _extract_faces(self, session, video_id: uuid.UUID, images: list[CatalogImage]) -> None:
        threshold = self.settings.face_similarity_threshold
        tracks: list[TrackBuffer] = []
        pending_faces: list[tuple[int, int, list[float], list[float]]] = []
        total_images = len(images)
        total_detected_faces = 0
        logger.debug("Face extraction started: video_id=%s total_images=%d", video_id, total_images)

        for idx, image in enumerate(images):
            faces = self.services.insightface.detect_faces(Path(image.image_path))
            total_detected_faces += len(faces)
            for face in faces:
                bbox: list[float] = face["bbox"]
                embedding: list[float] = face["embedding"]

                match_idx = self._match_track(tracks, embedding, threshold)
                if match_idx is None:
                    match_idx = len(tracks)
                    tracks.append(
                        TrackBuffer(
                            alias=f"face{match_idx + 1}",
                            start_sec=image.timestamp_sec,
                            end_sec=image.timestamp_sec,
                            vectors=[embedding],
                        )
                    )
                else:
                    track = tracks[match_idx]
                    track.end_sec = max(track.end_sec, image.timestamp_sec)
                    track.vectors.append(embedding)

                pending_faces.append((image.id, match_idx, bbox, embedding))
            logger.debug(
                "Face extraction progress: video_id=%s done=%d/%d (%.2f%%) ts=%ds detected_faces=%d tracks=%d",
                video_id,
                idx + 1,
                total_images,
                self._progress_percent(idx + 1, total_images),
                image.timestamp_sec,
                total_detected_faces,
                len(tracks),
            )

        if not tracks:
            logger.debug("Face extraction completed: video_id=%s detected_faces=0 tracks=0", video_id)
            return

        db_tracks: list[FaceTrack] = []
        for track in tracks:
            db_track = FaceTrack(
                video_id=video_id,
                alias=track.alias,
                start_sec=track.start_sec,
                end_sec=track.end_sec,
                embedding=track.centroid,
            )
            session.add(db_track)
            db_tracks.append(db_track)
        await session.flush()

        for catalog_image_id, track_idx, bbox, embedding in pending_faces:
            session.add(
                DetectedFace(
                    catalog_image_id=catalog_image_id,
                    face_track_id=db_tracks[track_idx].id,
                    x1=bbox[0],
                    y1=bbox[1],
                    x2=bbox[2],
                    y2=bbox[3],
                    embedding=embedding,
                )
            )
        await session.commit()
        logger.debug(
            "Face extraction completed: video_id=%s detected_faces=%d tracks=%d",
            video_id,
            total_detected_faces,
            len(tracks),
        )

    def _match_track(self, tracks: list[TrackBuffer], embedding: list[float], threshold: float) -> int | None:
        best_idx = None
        best_score = 0.0
        for idx, track in enumerate(tracks):
            score = cosine_similarity(track.centroid, embedding)
            if score >= threshold and score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

    @staticmethod
    def _progress_percent(done: int, total: int) -> float:
        if total <= 0:
            return 100.0
        return (done / total) * 100.0

    async def list_videos(self, page: int, size: int) -> tuple[list[Video], int]:
        offset = (page - 1) * size
        async with SessionLocal() as session:
            total = len((await session.execute(select(Video.id))).scalars().all())
            rows = (
                await session.execute(
                    select(Video)
                    .order_by(Video.created_at.desc())
                    .offset(offset)
                    .limit(size)
                )
            ).scalars().all()
            return rows, total
