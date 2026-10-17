#!/usr/bin/env python3
"""Build real-plan training rows from the hand labels in data/labels_*.json.

Labels are percent coords per sheet: {image, boxes:[{x,y,w,h,label,text}]} where
x/y/w/h are 0-100 of the FULL sheet. We emit the model's 0-1000 bbox space.

  python3 build_real.py --out ./data/real.jsonl            # all 49 plans
  python3 build_real.py --out ./data/real_holdout.jsonl --only-holdout
  python3 build_real.py --out ./data/real_train.jsonl --holdout 8
"""
import argparse, glob, json, os, random

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
SEARCH = ("blueprints/houses", "blueprints/usa", "blueprints")
# NEVER train on the demo sheet: real_quote/ is the pitch sample (505 11th St NW) and
# must stay unseen, same as test/. Excluded by path, not by name, so a re-export with a
# different filename still can't leak in.
EXCLUDE_DIRS = ("real_quote", "test")


def excluded(path):
    p = os.path.abspath(path)
    return any(os.sep + d + os.sep in p or p.endswith(os.sep + d) for d in EXCLUDE_DIRS)


def resolve(name):
    if os.path.isabs(name) and os.path.exists(name):
        return name
    for d in SEARCH:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def load_all():
    seen, rows = set(), []
    for f in sorted(glob.glob("data/labels_*.json")):
        for item in json.load(open(f)):
            name = os.path.basename(item["image"])
            if name in seen:
                continue
            path = resolve(item["image"]) or resolve(name)
            if not path or excluded(path):
                continue
            seen.add(name)
            objs = []
            for b in item.get("boxes", []):
                if b.get("label") not in ("Room", "Wall", "Door", "Window"):
                    continue
                objs.append({
                    "bbox": [round(b["x"] * 10), round(b["y"] * 10),
                             round((b["x"] + b["w"]) * 10), round((b["y"] + b["h"]) * 10)],
                    "class": b["label"],
                    "text": (b.get("text") or "").strip(),
                })
            if objs:
                rows.append({"image": path, "prompt": PROMPT, "objects": objs})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/real.jsonl")
    ap.add_argument("--holdout", type=int, default=0, help="keep N plans OUT of this file")
    ap.add_argument("--only-holdout", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    rows = load_all()
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    hold = rows[: a.holdout] if a.holdout else []
    keep = rows[a.holdout:] if a.holdout else rows
    if a.only_holdout:
        keep = hold
    with open(a.out, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    objs = sum(len(r["objects"]) for r in keep)
    print(f"{len(keep)} plans / {objs} objects -> {a.out}"
          + (f"  (held out {len(hold)})" if hold else ""))


if __name__ == "__main__":
    main()
