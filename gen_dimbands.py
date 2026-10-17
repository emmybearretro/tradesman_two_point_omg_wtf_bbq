#!/usr/bin/env python3
"""Synthetic dimension-band trainer: Minot-style witness lines + ticks + ft-in labels.

Deliberately DIFFERENT plans from any real sample (random grids) — the real
Minot sheet stays held-out. Trains what v3 failed: reading sparse dim strings
(40'-0", 12'-0", 5'-0" x 4'-0") on line-art.

Labels use explicit "Dimension" (dim strings) + "Room" (room boxes).
NOTE: data_convert.py's CLASS_MERGE drops "Dimension" — v4 ingest must whitelist
it (one line) or map Dimension->Room.

  python3 gen_dimbands.py --n 200 --out ./synth_dim --seed 21
"""
import argparse, json, os, random
from PIL import Image, ImageDraw, ImageFont

WALL = (20, 20, 20)
DIM = (40, 40, 40)

def ft_in(rng, lo=8, hi=40):
    return f"{rng.randint(lo, hi)}'-{rng.randint(0, 11)}\""

def tick(d, x, y):
    d.line([x - 5, y + 5, x + 5, y - 5], fill=DIM, width=2)

def load_font(size=26):
    import glob as _g
    from PIL import ImageFont as _F
    for p in (["assets/DejaVuSans.ttf"] +
              _g.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)):
        try:
            return _F.truetype(p, size)
        except Exception:
            continue
    return _F.load_default()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="./synth_dim")
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(f"{a.out}/png", exist_ok=True)
    font = load_font()
    rows = []
    for i in range(a.n):
        W, H = 1200, 900
        im = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(im)
        boxes = []
        # random inner plan rect (NOT the real plan: fresh random footprint)
        px0, py0 = rng.randint(150, 250), rng.randint(150, 220)
        px1, py1 = px0 + rng.randint(500, 800), py0 + rng.randint(350, 550)
        # room grid inside
        cols, rows_n = rng.randint(2, 3), rng.randint(2, 2)
        xs = [px0] + sorted(rng.sample(range(px0 + 150, px1 - 100), cols - 1)) + [px1]
        ys = [py0] + sorted(rng.sample(range(py0 + 120, py1 - 80), rows_n - 1)) + [py1]
        for r in range(rows_n):
            for c in range(cols):
                x0, y0, x1, y1 = xs[c], ys[r], xs[c + 1], ys[r + 1]
                d.rectangle([x0, y0, x1, y1], outline=WALL, width=3)
                boxes.append({"x": x0 / 12, "y": y0 / 9, "w": (x1 - x0) / 12,
                              "h": (y1 - y0) / 9, "label": "Room",
                              "text": f"Room {ft_in(rng, 8, 20)}"})
        # top overall dimension band (like 40'-0" across the sheet top)
        yb = py0 - 90
        d.line([px0, yb, px1, yb], fill=DIM, width=1)
        tick(d, px0, yb); tick(d, px1, yb)
        total = ft_in(rng, 30, 45)
        d.text(((px0 + px1) // 2 - 40, yb - 30), total, fill=DIM, font=font)
        boxes.append({"x": (px0 + px1) / 2 / 12 - 4, "y": (yb - 30) / 9,
                      "w": 9, "h": 3.5, "label": "Dimension", "text": total})
        # segment dims under it (like 12'-0" 11'-7" 16'-5")
        nseg = rng.randint(2, 4)
        cuts = sorted(rng.sample(range(px0 + 80, px1 - 60), nseg - 1))
        edges = [px0] + cuts + [px1]
        ys2 = yb + 45
        for s in range(nseg):
            x0, x1 = edges[s], edges[s + 1]
            if x1 - x0 < 70:
                continue
            d.line([x0, yb, x0, ys2 + 8], fill=DIM, width=1)
            d.line([x0, ys2, x1, ys2], fill=DIM, width=1)
            tick(d, x0, ys2); tick(d, x1, ys2)
            t = ft_in(rng, 8, 20)
            d.text(((x0 + x1) // 2 - 35, ys2 - 28), t, fill=DIM, font=font)
            boxes.append({"x": (x0 + x1) / 2 / 12 - 3.5, "y": (ys2 - 28) / 9,
                          "w": 8, "h": 3.2, "label": "Dimension", "text": t})
        d.line([px1, yb, px1, ys2 + 8], fill=DIM, width=1)
        # window/door tag (like 5'-0" x 4'-0")
        if rng.random() < 0.7:
            tag = f"{rng.randint(3, 6)}'-{rng.randint(0, 11)}\" x {rng.randint(3, 6)}'-{rng.randint(0, 11)}\""
            tx, ty = px0 + rng.randint(0, 200), py1 + 25
            d.text((tx, ty), tag, fill=DIM, font=font)
            boxes.append({"x": tx / 12, "y": ty / 9, "w": 13, "h": 3.2,
                          "label": "Dimension", "text": tag})
        # left vertical band (one span, like 25'-0")
        xb = px0 - 80
        d.line([xb, py0, xb, py1], fill=DIM, width=1)
        tick(d, xb, py0); tick(d, xb, py1)
        vt = ft_in(rng, 20, 35)
        d.text((xb - 70, (py0 + py1) // 2 - 12), vt, fill=DIM, font=font)
        boxes.append({"x": (xb - 70) / 12, "y": (py0 + py1) / 2 / 9 - 1.5,
                      "w": 6.5, "h": 3, "label": "Dimension", "text": vt})
        im.save(f"{a.out}/png/dim_{i:04d}.png")
        rows.append({"image": f"dim_{i:04d}.png", "boxes": boxes})
    json.dump(rows, open(f"{a.out}/labels_dim.json", "w"))
    nd = sum(1 for r in rows for b in r["boxes"] if b["label"] == "Dimension")
    print(f"wrote {len(rows)} dim sheets -> {a.out}/ ({nd} Dimension + "
          f"{sum(1 for r in rows for b in r['boxes'] if b['label'] == 'Room')} Room)")

if __name__ == "__main__":
    main()
