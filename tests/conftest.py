import io

import av
from PIL import Image
import pytest


@pytest.fixture
def mov_bytes():
    output = io.BytesIO()
    with av.open(output, "w", format="mov") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(3):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), (30 + i, 60, 80)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()
