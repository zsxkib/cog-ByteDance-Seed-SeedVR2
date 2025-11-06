import shutil
import subprocess
from pathlib import Path

import torch


def cut_videos(video: torch.Tensor, sp_size: int) -> torch.Tensor:
    if video.size(1) > 121:
        video = video[:, :121]
    total = video.size(1)
    if total <= 4 * sp_size:
        padding = [video[:, -1].unsqueeze(1)] * (4 * sp_size - total + 1)
        padding = torch.cat(padding, dim=1)
        return torch.cat([video, padding], dim=1)
    if (total - 1) % (4 * sp_size) == 0:
        return video
    padding = [video[:, -1].unsqueeze(1)] * (4 * sp_size - ((total - 1) % (4 * sp_size)))
    padding = torch.cat(padding, dim=1)
    return torch.cat([video, padding], dim=1)


def mux_audio_stream(src_media: Path, video_only: Path, output_path: Path) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("[WARN] ffmpeg missing; returning video without audio passthrough.")
        video_only.replace(output_path)
        return output_path

    ffmpeg_cmd = [
        ffmpeg_path,
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(src_media),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        str(output_path),
    ]

    result = subprocess.run(ffmpeg_cmd, check=False, capture_output=True)
    if result.returncode == 0:
        video_only.unlink(missing_ok=True)
        return output_path

    stderr = result.stderr.decode("utf-8", errors="ignore") if result.stderr else ""
    print("[WARN] Audio mux failed, returning video-only result.")
    if stderr:
        print(stderr)
    video_only.replace(output_path)
    return output_path
