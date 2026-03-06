import json
import subprocess
from pathlib import Path


class FFmpegService:
    def probe_duration(self, video_path: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        return float(payload["format"]["duration"])

    def extract_frames_every_second(self, video_path: Path, output_dir: Path) -> list[Path]:
        output_pattern = output_dir / "frame_%06d.jpg"
        # Keep aspect ratio, cap to 1080p, and never upscale lower-resolution inputs.
        scale_filter = "scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1,{scale_filter}",
            str(output_pattern),
        ]
        subprocess.run(command, capture_output=True, text=True, check=True)
        return sorted(output_dir.glob("frame_*.jpg"))
