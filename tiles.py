"""Tiled inference for large sheets.

The model sees at most ~1664 px on the long side. A 7200 px scanned page squeezed to
that size loses every dimension string, which is exactly how v4 failed on the real
Minot sheet. So: cut the sheet into readable tiles with overlap, run each, then map
boxes back into page space and de-duplicate the ones found twice at the seams.

  from tiles import run_tiled
  dets = run_tiled(lambda im: parse(model_output(im)), Image.open("plan.png"))
"""
from PIL import Image

MAX_SIDE = 1664          # model's comfortable long side
DEFAULT_TILE = 1536
DEFAULT_OVERLAP = 192


def iter_tiles(img, tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP):
    """Yield (crop, (ox, oy)) in page pixels. Panes never exceed `tile`."""
    W, H = img.size
    if W <= MAX_SIDE and H <= MAX_SIDE:
        yield img, (0, 0)
        return
    step = max(64, tile - overlap)
    xs = list(range(0, max(1, W - overlap), step))
    ys = list(range(0, max(1, H - overlap), step))
    for y in ys:
        for x in xs:
            box = (x, y, min(W, x + tile), min(H, y + tile))
            crop = img.crop(box)
            if crop.size[0] < 64 or crop.size[1] < 64:
                continue
            yield crop, (x, y)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_page(dets, offset, crop_size, page_size):
    """Tile-local 0-1000 boxes -> page 0-1000 boxes."""
    ox, oy = offset
    cw, ch = crop_size
    W, H = page_size
    out = []
    for d in dets:
        b = d["bbox"]
        x0 = ox + b[0] / 1000 * cw
        y0 = oy + b[1] / 1000 * ch
        x1 = ox + b[2] / 1000 * cw
        y1 = oy + b[3] / 1000 * ch
        d = dict(d)
        d["bbox"] = [x0 / W * 1000, y0 / H * 1000, x1 / W * 1000, y1 / H * 1000]
        d["_area"] = (x1 - x0) * (y1 - y0) * 1.0 / (cw * ch)  # tile coverage, for dedupe
        out.append(d)
    return out


def nms(dets, thresh=0.55):
    """Drop duplicates from overlapping tiles (same class + high IoU)."""
    kept = []
    for d in sorted(dets, key=lambda x: -x.get("_area", 0)):
        if any(k.get("label") == d.get("label") and iou(k["bbox"], d["bbox"]) >= thresh
               for k in kept):
            continue
        kept.append(d)
    for k in kept:
        k.pop("_area", None)
    return kept


def run_tiled(fn, img, tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP, verbose=False):
    """fn(crop_image) -> [det] in 0-1000 crop space. Returns page-space dets."""
    all_dets = []
    for i, (crop, off) in enumerate(iter_tiles(img, tile, overlap)):
        dets = fn(crop) or []
        mapped = to_page(dets, off, crop.size, img.size)
        all_dets.extend(mapped)
        if verbose:
            print(f"  tile {i+1} @{off} size={crop.size} -> {len(dets)} dets")
    return nms(all_dets)
