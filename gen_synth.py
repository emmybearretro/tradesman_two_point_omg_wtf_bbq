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
from PIL import Image, ImageDraw, ImageFont

ROOM_TYPES = [("Bedroom", (1, 3)), ("Living", (1, 1)), ("Kitchen", (1, 1)),
              ("Bath", (1, 2)), ("Corridor", (0, 1)), ("Balcony", (0, 1))]
PALETTES = [[(255, 214, 214), (214, 255, 214), (214, 230, 255), (255, 244, 214)],
            [(245, 235, 215), (235, 235, 235), (220, 235, 245), (245, 225, 210)]]
WALL = (20, 20, 20)

EN_NAMES = ["Bedroom", "Living", "Kitchen", "Bath", "Corridor", "Balcony"]
# render name per lang; truth text is ALWAYS EN transliteration (matches train scheme)
LANGS = {
    "en": {"rooms": EN_NAMES, "units": "mix", "dims": True},
    "ar": {"rooms": ["نوم", "معيشة", "مطبخ", "حمام", "طرقة", "بلكونة"], "units": "m",
           "dims": True, "eastern_digits": True},
    "fi": {"rooms": ["MH", "OH", "K", "KPH", "VH", "PARVEKE"], "units": "m", "dims": "sometimes"},
    "es": {"rooms": ["Dormitorio", "Salón", "Cocina", "Baño", "Pasillo", "Balcón"],
           "units": "m", "dims": True},
    "fr": {"rooms": ["Chambre", "Salon", "Cuisine", "Salle de bain", "Couloir", "Balcon"],
           "units": "m", "dims": True},
    "de": {"rooms": ["Schlafzimmer", "Wohnzimmer", "Küche", "Bad", "Flur", "Balkon"],
           "units": "m", "dims": True},
    "hi": {"rooms": ["बेडरूम", "लिविंग रूम", "रसोई", "बाथरूम", "गलियारा", "बालकनी"],
           "units": "m", "dims": True},
}
MIX20 = (["en"] * 5 + ["ar"] * 4 + ["fi"] * 3 + ["es"] * 2 + ["fr"] * 2 +
         ["de"] * 2 + ["hi"] * 2)
AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

def load_font(lang, size=28):
    import glob as _g
    from PIL import ImageFont as _F

    def prefer_full(paths):
        # Display cuts (Condensed/ExtraBold/Thin/Black) are often subset and miss
        # Latin + digits. Plain Medium/Regular first.
        def key(p):
            n = p.lower()
            bad = ("condensed" in n) + ("extrabold" in n) + ("thin" in n) + ("black" in n)
            good = ("medium" in n) + ("regular" in n)
            return (bad, -good, p)
        return sorted(set(paths), key=key)

    if lang == "ar":
        cands = prefer_full(
            ["assets/NotoSansArabic.ttf"] +
            _g.glob("/usr/share/fonts/**/NotoSansArabic*.ttf", recursive=True) +
            _g.glob("/usr/share/fonts/**/Amiri*.ttf", recursive=True))
    elif lang == "hi":
        cands = prefer_full(
            ["assets/NotoSansDevanagari.ttf"] +
            _g.glob("/usr/share/fonts/**/NotoSansDevanagari*.ttf", recursive=True) +
            _g.glob("/usr/share/fonts/**/Lohit*.ttf", recursive=True))
    else:
        cands = prefer_full(
            ["assets/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"] +
            _g.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True))
    for p in cands:
        try:
            return _F.truetype(p, size)
        except Exception:
            continue
    return _F.load_default()

def shape_ar(s):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(s))
    except Exception:
        return s

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
    ap.add_argument("--langs", default="en", help="'mix' (=MIX20 incl. Hindi) or comma list: en,ar,fi,es,fr,de,hi")
    a = ap.parse_args()
    lang_seq = list(MIX20) if a.langs == "mix" else [s for s in (l.strip() for l in a.langs.split(",")) if s in LANGS]
    assert lang_seq, "empty --langs"
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
        lang = lang_seq[i % len(lang_seq)]
        pack = LANGS[lang]
        use_m = True if pack["units"] == "m" else (rng.random() < 0.5)
        policy = pack["dims"]
        font = load_font(lang)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
               f'<rect width="{W}" height="{H}" fill="#fafafa"/>']
        boxes, ci = [], 0
        for r in range(rows_n):
            for c_ in range(cols):
                x0, y0, x1, y1 = xs[c_], ys[r], xs[c_ + 1], ys[r + 1]
                if x1 - x0 < 150 or y1 - y0 < 150:
                    continue
                name = ROOM_TYPES[ci % len(ROOM_TYPES)][0]
                ri = ci % len(EN_NAMES)
                render_name = pack["rooms"][ri]
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
                show = True if policy is True else (rng.random() < 0.6)
                if use_m and show:
                    dim_w, dim_h = f"{wft * 0.3048:.2f}", f"{hft * 0.3048:.2f}"
                    render_dim = f"{dim_w}x{dim_h} م" if lang == "ar" else f"{dim_w}x{dim_h} m"
                    if lang == "ar":
                        render_dim = render_dim.translate(AR_DIGITS)
                    truth_dim = f"{dim_w}x{dim_h} m"
                elif show:
                    render_dim = truth_dim = fmt_dim(wft, hft, False)
                else:
                    render_dim = truth_dim = ""
                render_txt = f"{render_name} {render_dim}".strip()
                truth_txt = f"{EN_NAMES[ri]} {truth_dim}".strip()
                draw_name = shape_ar(render_name) if lang == "ar" else render_name
                cy = (y0 + y1) // 2 - 10
                if lang in ("ar", "hi") and truth_dim:
                    # Script fonts lie about Latin coverage (tofu x/m): room name in
                    # the script font, dims (western digits) in DejaVu. Fully legible.
                    latn = load_font("en")
                    d.text((x0 + 20, cy), truth_dim, fill=(10, 10, 10), font=latn)
                    dw = d.textlength(truth_dim, font=latn)
                    d.text((x0 + 20 + dw + 10, cy), draw_name, fill=(10, 10, 10),
                           font=font)
                else:
                    draw_txt = shape_ar(f"{render_name} {render_dim}".strip()) if lang == "ar" \
                        else f"{render_name} {render_dim}".strip()
                    basic = getattr(getattr(ImageFont, "Layout", None), "BASIC", None)
                    kwargs = {"layout_engine": basic} if (lang == "ar" and basic is not None) else {}
                    d.text((x0 + 20, cy), draw_txt, fill=(10, 10, 10),
                           font=font, **kwargs)
                svg += [f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
                        f'fill="rgb({color[0]},{color[1]},{color[2]})" stroke="black" stroke-width="6"/>',
                        f'<text x="{x0+20}" y="{(y0+y1)//2}">{render_txt}</text>']
                boxes.append({"x": x0 / 10, "y": y0 / 10, "w": (x1 - x0) / 10,
                              "h": (y1 - y0) / 10, "label": "Room", "text": truth_txt})
        svg.append("</svg>")
        open(f"{a.out}/svg/plan_{i:04d}.svg", "w").write("\n".join(svg))
        im.save(f"{a.out}/png/plan_{i:04d}.png")
        rows.append({"image": f"plan_{i:04d}.png", "boxes": boxes})
    json.dump(rows, open(f"{a.out}/labels_synth.json", "w"))
    print(f"wrote {len(rows)} synth plans -> {a.out}/ (+svg, +labels_synth.json)")

if __name__ == "__main__":
    main()
