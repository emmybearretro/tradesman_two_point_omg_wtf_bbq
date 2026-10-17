"""Deterministic lumber takeoff. NO AI here - pure math judges trust.
Stud wall 16" OC + double top plate + single bottom + 10% waste.
"""
import math, re

STUD_SPACING_IN = 16
WASTE = 0.10
PRICES = {"stud_2x4x8": 3.80, "plate_2x4x8": 3.20, "header_2x8": 18.0}
DIM_RE = re.compile(r"""(\d+(?:\.\d+)?)\s*('|ft|feet|\"|in|inch|m|meter|cm|mm)?\s*(?:x|×|,|by)\s*(\d+(?:\.\d+)?)\s*('|ft|feet|\"|in|inch|m|meter|cm|mm)?""", re.I)
M_TO_FT = 3.28084

def _to_ft(v, mark, metric_ctx):
    mark = (mark or "").lower()
    if mark in ('"', "in", "inch"):
        return v / 12
    if mark in ("m", "meter") or (not mark and metric_ctx):
        return v * M_TO_FT
    if mark == "cm":
        return v * M_TO_FT / 100
    if mark == "mm":
        return v * M_TO_FT / 1000
    return v  # ft, ', feet, or bare imperial default

def parse_dim(text):
    """'Bedroom 12 x 14' / '12ft x 14' / '4.50x3.91 m' / '12\'6\" x 14' -> (w_ft, l_ft) or None."""
    t = text.replace("’", "'").replace("″", '"').replace("′", "'")
    # 10'-3" / 10' 3" -> 10.25 (decimal feet) before pair-matching
    t = re.sub(r"""(\d+)\s*'\s*-?\s*(\d+(?:\.\d+)?)\s*" """,
               lambda m: str(float(m.group(1)) + float(m.group(2)) / 12), t)
    metric_ctx = bool(re.search(r"\bm\b|meter|متر", t)) and not re.search(r"ft|feet|'", t)
    m = DIM_RE.search(t)
    if not m:
        return None
    return (_to_ft(float(m.group(1)), m.group(2), metric_ctx),
            _to_ft(float(m.group(3)), m.group(4), metric_ctx))

def studs_for_wall(length_ft, openings=0):
    studs = math.ceil(length_ft * 12 / STUD_SPACING_IN) + 1
    studs += openings * 2  # king + jack per opening
    return studs

def estimate_room(name, w_ft, l_ft, openings=2, height_ft=8):
    perim = 2 * (w_ft + l_ft)
    studs = studs_for_wall(w_ft) * 2 + studs_for_wall(l_ft) * 2
    studs += openings * 2
    plates = math.ceil(perim / 8) * 3  # 2 top + 1 bottom, 8ft sticks
    studs = math.ceil(studs * (1 + WASTE))
    cost = studs * PRICES["stud_2x4x8"] + plates * PRICES["plate_2x4x8"] + openings * PRICES["header_2x8"]
    return {"room": name, "size": f"{w_ft}x{l_ft}", "perim_ft": round(perim, 1),
            "studs": studs, "plates": plates, "headers": openings, "cost": round(cost, 2)}

def estimate_house(detections):
    """detections: [{label, text, openings?}] -> (rows, totals)."""
    rows = []
    for d in detections:
        if d.get("label", "").lower() not in ("room",):
            continue
        dim = parse_dim(d.get("text", ""))
        if not dim:
            continue
        rows.append(estimate_room(d.get("room", d["text"][:24]), dim[0], dim[1],
                                  openings=d.get("openings", 2)))
    total = {"studs": sum(r["studs"] for r in rows),
             "plates": sum(r["plates"] for r in rows),
             "cost": round(sum(r["cost"] for r in rows), 2)}
    return rows, total

if __name__ == "__main__":
    demo = [{"label": "Room", "room": "Bedroom", "text": "Bedroom 12 x 14", "openings": 3},
            {"label": "Room", "room": "Garage", "text": "20 x 22", "openings": 2}]
    rows, tot = estimate_house(demo)
    for r in rows: print(r)
    print("TOTAL", tot)
