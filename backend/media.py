"""Decode uploaded QuickTime files in a bounded, separate process."""
import multiprocessing
import time
from pathlib import Path

import av


class InvalidVideo(ValueError):
    pass


def _validate(path: str, result):
    try:
        # Do not resolve external QuickTime data references or network protocols.
        with av.open(path, options={"protocol_whitelist": "file", "enable_drefs": "0", "err_detect": "explode"}) as container:
            if "mov" not in container.format.name.split(",") or not container.streams.video:
                raise InvalidVideo()
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            width, height = stream.codec_context.width, stream.codec_context.height
            if width <= 0 or height <= 0 or width * height > 16_777_216:
                raise InvalidVideo()
            if container.duration is not None and container.duration / av.time_base > 600:
                raise InvalidVideo()
            count = 0
            for frame in container.decode(stream):
                count += 1
                if count > 72_000 or (frame.time is not None and frame.time > 600):
                    raise InvalidVideo()
            if not count or (stream.frames and count < stream.frames):
                raise InvalidVideo()
        result.send(True)
    except Exception:
        result.send(False)
    finally:
        result.close()


def validate_video(path: Path, timeout: float = 20):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_validate, args=(str(path), child), daemon=True)
    try:
        process.start()
        child.close()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if parent.poll(min(.05, max(0, deadline - time.monotonic()))):
                try:
                    valid = parent.recv()
                except EOFError:
                    valid = False
                if valid:
                    return
                break
            if not process.is_alive():
                break
        raise InvalidVideo("Файл повреждён, не содержит декодируемое MOV-видео или превышает ограничения проверки")
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        parent.close()
        child.close()
