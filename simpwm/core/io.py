"""Video output."""

import numpy as np


def write_mp4(frames, path, fps=24):
    """Write RGB uint8 frames (H, W, 3) as H.264 (CRF 18, yuv420p); odd sizes are cropped to even."""
    import av

    frames = [np.ascontiguousarray(f) for f in frames]
    H, W = frames[0].shape[:2]
    H, W = H - H % 2, W - W % 2
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=int(round(fps)))
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "18"}
        for f in frames:
            for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:H, :W]), format="rgb24")):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)
