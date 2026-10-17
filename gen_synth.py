#!/usr/bin/env python3
"""Synthetic blueprint generator: unlimited perfectly-labelled training samples.
Outputs SVG (view) + PNG (train) + labels JSON (our LabelStudio format).

Division of labor: synth teaches GEOMETRY + dim-linking with perfect boxes.
Real plans teach texture/fonts/Arabic glyphs. Mix both in train.

  python3 gen_synth.py --n 200 --out ./synth --seed 7
  python3 data_convert.py --labelstudio ./synth/labels_synth.json --imgdir ./synth/png --out ./data/synth.jsonl

English labels + mixed ft/metric dims (matches our transliterated label scheme).
Styles mimic Egyptian pastels + US beige so the model can't cheat on color.
"""
import argparse, json, os, random
from PIL import Image, ImageDraw

ROOM_TYPES = [("Bedroom", (1, 3)), ("Living", (1, 1)), ("Kitchen", (1, 1)),
              ("Bath", (1, 2)), ("Corridor", (0, 1)), ("Balcony", (0, 1))]
PALETTES = [[(255, 214, 214), (214, 255, 214), (214, 230, 255), (255, 244, 214)],
            [(245, 235, 215), (235, 235, 235), (220, 235, 245), (245, 225, 210)]]
WALL = (20, 20, 20)

def fmt_dim(w, h, metric):
    if metric:
        return f"{w * 0.3048:.2f}x{h * 0.3048:.2f} m"
    if random.random() < 0.5:
        return f"{w}' x {h}'"
    return f"{w}x{h}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="./synth")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(f"{a.out}/svg", exist_ok=True)
    os.makedirs(f"{a.out}/png", exist_ok=True)
    # NOTE: ImageDraw roundtrip — recreate image properly per sample
    from PIL import Image as I
    rows = []
    for i in range(a.n):
        W, H = 1000, 1000
        im = I.new("RGB", (W, H), (250, 250, 250))
        d = ImageDraw.Draw(im)
        pal = PALETTES[i % len(PALETTES)]
        cols, rows_n = rng.randint(2, 3), rng.randint(2, 3)
        xs = sorted(rng.sample(range(200, 850), cols - 1))
        ys = sorted(rng.sample(range(200, 850), rows_n - 1))
        xs = [0] + [x + rng.randint(-30, 30) for x in xs] + [W]
        ys = [0] + [y + rng.randint(-30, 30) for y in ys] + [H]
        metric = rng.random() < 0.5
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
               f'<rect width="{W}" height="{H}" fill="#fafafa"/>']
        boxes, ci = [], 0
        for r in range(rows_n):
            for c_ in range(cols):
                x0, y0, x1, y1 = xs[c_], ys[r], xs[c_ + 1], ys[r + 1]
                if x1 - x0 < 150 or y1 - y0 < 150:
                    continue
                name = ROOM_TYPES[ci % len(ROOM_TYPES)][0]
                color = pal[ci % len(pal)]
                ci += 1
                d.rectangle([x0, y0, x1, y1], fill=color, outline=WALL, width=6)
                side = rng.randrange(4)
                if side in (0, 1):
                    dx, dy = rng.randint(x0 + 40, max(x0 + 41, x1 - 90)), y0 if side == 0 else y1
                else:
                    dx, dy = (x0 if side == 2 else x1), rng.randint(y0 + 40, max(y0 + 41, y1 - 90))
                d.arc([dx - 4, dy - 45, dx + 86, dy + 45], 0, 90, fill=WALL, width=3)
                wft = round((x1 - x0) / 55, 1)
                hft = round((y1 - y0) / 55, 1)
                txt = f"{name} {fmt_dim(wft, hft, metric)}"
                d.text((x0 + 20, (y0 + y1) // 2 - 10), txt, fill=(10, 10, 10))
                svg += [f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
                        f'fill="rgb({color[0]},{color[1]},{color[2]})" stroke="black" stroke-width="6"/>',
                        f'<text x="{x0+20}" y="{(y0+y1)//2}">{txt}</text>']
                boxes.append({"x": x0 / 10, "y": y0 / 10, "w": (x1 - x0) / 10,
                              "h": (y1 - y0) / 10, "label": "Room", "text": txt})
        svg.append("</svg>")
        open(f"{a.out}/svg/plan_{i:04d}.svg", "w").write("\n".join(svg))
        im.save(f"{a.out}/png/plan_{i:04d}.png")
        rows.append({"image": f"plan_{i:04d}.png", "boxes": boxes})
    json.dump(rows, open(f"{a.out}/labels_synth.json", "w"))
    print(f"wrote {len(rows)} synth plans -> {a.out}/ (+svg, +labels_synth.json)")

if __name__ == "__main__":
    main()
