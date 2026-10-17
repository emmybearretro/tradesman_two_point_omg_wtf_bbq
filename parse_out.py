"""Parse Nemotron-Parse native output into drawing/BOM-ready detections.

The fine-tuned model emits one line per object in its native token format:

    <x_0.083><y_0.0><x_0.2715><y_0.2297><class_Room> Bedroom 1.49x1.37 m

Coordinates are normalized 0-1 tokens (x: 1024 bins, y: 1280 bins). This module
turns that into pixel/normalized boxes plus a room name and a dimension string,
which is what lumber.estimate_house() and the Gradio UI consume.

Replaces the demo-grade regex fallback and the vendored NVIDIA postprocessing
(which only knows document classes, not our Wall/Door/Window/Room vocabulary).
"""
import re

# optional class token: earlier checkpoints emitted it after the coords
LINE = re.compile(
    r"<x_([\d.]+)><y_([\d.]+)><x_([\d.]+)><y_([\d.]+)>"
    r"(?:<class_([A-Za-z_-]+)>)?\s*([^\n<]*)"
)
DIMS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:'|ft|m|cm|mm)?\s*[x×]\s*(\d+(?:\.\d+)?)", re.I)


def parse(text, width=None, height=None, scale=1000.0):
    """-> [{label, room, text, dims, bbox, openings}].

    bbox is [x0, y0, x1, y1] in 0..scale units (the UI's coordinate space) when
    width/height are omitted, or in pixels when they are given.
    """
    sx = (width / scale) if width else 1.0
    sy = (height / scale) if height else 1.0
    out = []
    for m in LINE.finditer(text or ""):
        x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        body = (m.group(6) or "").strip(" |\\")
        label = m.group(5) or "Room"
        dim = DIMS.search(body)
        room = body[: dim.start()].strip(" -:|") if dim else body.strip(" -:|")
        out.append({
            "label": label,
            "room": room or label,
            "text": body,
            "dims": f"{dim.group(1)}x{dim.group(2)}" if dim else "",
            "bbox": [x0 * scale * sx, y0 * scale * sy,
                     x1 * scale * sx, y1 * scale * sy],
            "openings": 2,  # v1 has no door/window objects; see README limits
        })
    return out


def to_pixels(dets, width, height, scale=1000.0):
    for d in dets:
        b = d["bbox"]
        d["bbox"] = [b[0] / scale * width, b[1] / scale * height,
                     b[2] / scale * width, b[3] / scale * height]
    return dets
