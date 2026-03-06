import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.core.config import get_settings


class StorageService:
    def __init__(self) -> None:
        settings = get_settings()
        self.root = settings.storage_root
        self.video_dir = self.root / "videos"
        self.frame_dir = self.root / "frames"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir.mkdir(parents=True, exist_ok=True)

    async def save_video(self, upload_file: UploadFile) -> Path:
        suffix = Path(upload_file.filename or "").suffix or ".mp4"
        target = self.video_dir / f"{uuid.uuid4()}{suffix}"
        async with aiofiles.open(target, "wb") as out_file:
            while chunk := await upload_file.read(1024 * 1024):
                await out_file.write(chunk)
        await upload_file.close()
        return target

    async def save_image_bytes(self, filename: str, payload: bytes) -> Path:
        target = self.root / "queries" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "wb") as out_file:
            await out_file.write(payload)
        return target

    def video_frame_dir(self, video_id: str) -> Path:
        target = self.frame_dir / video_id
        target.mkdir(parents=True, exist_ok=True)
        return target
