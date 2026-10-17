"""Local, timestamp-aware decoding. No network or label access."""
from __future__ import annotations

from pathlib import Path
import base64
import io
import time

import av
import numpy as np
from PIL import Image, ImageOps


def iter_frames(path: str | Path, deadline: float | None = None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("File contains no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        origin = None
        previous = -1.0
        for frame in container.decode(stream):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Local video analysis exceeded its time budget")
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Video frame has no presentation timestamp")
            timestamp = float(frame.pts * frame.time_base)
            if origin is None:
                origin = timestamp
            timestamp -= origin
            if timestamp < previous:
                raise ValueError("Non-monotonic video timestamps")
            previous = timestamp
            yield timestamp, frame


def frame_image(frame):
    image = frame.to_image()
    # PyAV exposes the display-matrix rotation; older files may have no matrix.
    rotation = getattr(frame, "rotation", 0)
    if rotation:
        image = image.rotate(rotation, expand=True)
    return image.convert("RGB")


def decode_sampled(path, fps=2.0, size=224, deadline=None, max_seconds=600):
    if fps <= 0 or size < 32:
        raise ValueError("fps must be positive and image size at least 32")
    images, timestamps = [], []
    next_sample = 0.0
    last_time = None
    for timestamp, frame in iter_frames(path, deadline):
        last_time = timestamp
        if timestamp > max_seconds:
            raise ValueError(f"Video exceeds the supported {max_seconds}s limit")
        if timestamp + 1e-8 < next_sample:
            continue
        image = ImageOps.pad(frame_image(frame), (size, size),
                             method=Image.Resampling.BILINEAR, color=(114, 114, 114))
        images.append(np.asarray(image, dtype=np.uint8))
        timestamps.append(timestamp)
        next_sample = (int(timestamp * fps + 1e-6) + 1) / fps
    if not images or last_time is None:
        raise ValueError("Video contains no decodable frames")
    return np.stack(images), np.asarray(timestamps, dtype=np.float64), float(last_time)


def clip_indices(times, center, frames=8, fps=2.0):
    """Select closest actual PTS, repeating boundary frames when necessary."""
    wanted = center + (np.arange(frames) - (frames - 1) / 2) / fps
    right = np.searchsorted(times, wanted).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(times[left] - wanted) <= np.abs(times[right] - wanted), left, right)


def evidence_frames(path, targets, deadline=None):
    """Decode again at original resolution, retaining nearest frames for each target."""
    targets = sorted(set(float(x) for x in targets))
    if not targets:
        return {}
    chosen = {target: (float("inf"), None, None) for target in targets}
    for timestamp, frame in iter_frames(path, deadline):
        for target in targets:
            distance = abs(timestamp - target)
            if distance < chosen[target][0]:
                chosen[target] = (distance, timestamp, frame)
        if timestamp > targets[-1] + 1:
            break
    result = {}
    for target, (_, timestamp, frame) in chosen.items():
        if frame is None:
            raise ValueError("Cannot extract evidence frame")
        image = frame_image(frame)
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        result[target] = {"timestamp_sec": round(timestamp, 4),
                          "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
    return result
