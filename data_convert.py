"""FloorPlanCAD + custom blueprints -> Nemotron-Parse-2.0 JSONL.
Usage:
  python data_convert.py --coco labels.json --imgdir ./pngs --out ./data/train.jsonl
  python data_convert.py --labelstudio custom.json --imgdir ./blueprints --out ./data/custom.jsonl
Input images MUST already be rasterized PNG/RGB at ~1664x2048.
FloorPlanCAD vectors -> render first (AutoCAD/dxf2img), thicken lines to 3px.
"""
import argparse, json, os
from PIL import Image

TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

# Merge 30 FloorPlanCAD classes -> 4 we care about for lumber. Win = 4 done well.
CLASS_MERGE = {
    "wall": "Wall", "partition": "Wall",
    "door": "Door", "sliding_door": "Door",
    "window": "Window",
    "bedroom": "Room", "bathroom": "Room", "kitchen": "Room",
    "livingroom": "Room", "room": "Room",
}
KEEP = {"Wall", "Door", "Window", "Room"}

def norm_box(xmin, ymin, xmax, ymax, w, h):
    # Parse expects 0-1000 normalized ints
    return [int(xmin/w*1000), int(ymin/h*1000), int(xmax/w*1000), int(ymax/h*1000)]

def coco_to_rows(coco_path, imgdir):
    coco = json.load(open(coco_path))
    cats = {c["id"]: c["name"].lower() for c in coco["categories"]}
    imgs = {i["id"]: i for i in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    rows = []
    for iid, anns in by_img.items():
        meta = imgs[iid]
        p = os.path.join(imgdir, meta["file_name"])
        if not os.path.exists(p):
            continue
        w, h = meta["width"], meta["height"]
        objs = []
        for a in anns:
            raw = cats.get(a["category_id"], "")
            cls = CLASS_MERGE.get(raw, None)
            if cls not in KEEP:
                continue
            x, y, bw, bh = a["bbox"]
            objs.append({"bbox": norm_box(x, y, x+bw, y+bh, w, h),
                         "class": cls, "text": cls})
        if not objs:
            continue
        rows.append({"image": p, "prompt": TASK_PROMPT, "objects": objs})
    return rows

def labelstudio_to_rows(ls_path, imgdir):
    # Expects LabelStudio YOLO-ish export: [{image, boxes:[{x,y,w,h,label,text}]}] in 0-100 pct
    data = json.load(open(ls_path))
    rows = []
    for item in data:
        p = os.path.join(imgdir, item["image"])
        if not os.path.exists(p):
            continue
        w, h = Image.open(p).size
        objs = []
        for b in item.get("boxes", []):
            if b["label"] not in KEEP:
                continue
            objs.append({"bbox": norm_box(b["x"]/100*w, b["y"]/100*h,
                                          (b["x"]+b["w"])/100*w, (b["y"]+b["h"])/100*h, w, h),
                         "class": b["label"], "text": b.get("text", b["label"])})
        rows.append({"image": p, "prompt": TASK_PROMPT, "objects": objs})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default=None)
    ap.add_argument("--labelstudio", default=None)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=2000)
    a = ap.parse_args()
    rows = []
    if a.coco:
        rows += coco_to_rows(a.coco, a.imgdir)
    if a.labelstudio:
        rows += labelstudio_to_rows(a.labelstudio, a.imgdir)
    rows = rows[:a.max]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}")

if __name__ == "__main__":
    main()
