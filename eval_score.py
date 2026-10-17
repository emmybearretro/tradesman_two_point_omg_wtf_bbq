"""Score a checkpoint against ground truth on the synth jsonl (exact GT boxes).

  python3 eval_score.py --ckpt ./ckpt-v2 --n 8

Reports, per image and averaged: objects found, greedy IoU-matched mean IoU,
and the room-label text match rate (the dimension string is what the BOM uses).
"""
import argparse, json, re, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig

BASE = "nvidia/NVIDIA-Nemotron-Parse-2.0"
from classes import CLASS_TOKENS as NEW
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
PAT = re.compile(r"<x_([\d.]+)><y_([\d.]+)><x_([\d.]+)><y_([\d.]+)>(?:<class_(\w+)>)?\s*([^\n<]*)")


def parse(text, w, h):
    out = []
    for m in PAT.finditer(text):
        x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
        out.append({"bbox": [x0 * w, y0 * h, x1 * w, y1 * h],
                    "class": m.group(5) or "", "text": m.group(6).strip()})
    return out


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./ckpt-v2")
    ap.add_argument("--jsonl", default="data/train_v1.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.add_tokens(NEW)
    gen = GenerationConfig.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.ckpt, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()

    rows = [json.loads(l) for l in open(a.jsonl)]
    rows = [r for r in rows if r["objects"] and r["objects"][0]["class"] in ("Room", "Window")][:a.n]
    ious, txt_hits, ngt, npred = [], 0, 0, 0
    for r in rows:
        img = Image.open(r["image"]).convert("RGB")
        w, h = img.size
        enc = proc(images=[img], text=PROMPT, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, generation_config=gen, max_new_tokens=a.max_new_tokens)
        pred = parse(tok.decode(out[0], skip_special_tokens=False), w, h)
        gts = list(r["objects"])
        npred += len(pred); ngt += len(gts)
        used = set()
        for g in gts:
            best, bi = 0.0, -1
            for j, p in enumerate(pred):
                if j in used:
                    continue
                v = iou(g["bbox"], p["bbox"])
                if v > best:
                    best, bi = v, j
            if bi >= 0 and best > 0.3:
                used.add(bi); ious.append(best)
                if g["text"].strip().lower() in pred[bi]["text"].lower():
                    txt_hits += 1
        print(f"{r['image'].split('/')[-1]}: gt={len(gts)} pred={len(pred)} matched={len(used)}")

    m = sum(ious) / len(ious) if ious else 0.0
    print(f"\nimages={len(rows)} gt_objs={ngt} pred_objs={npred}")
    print(f"matched_objs={len(ious)}  mean_IoU={m:.3f}  "
          f"text_match={txt_hits}/{len(ious) if ious else 0}")


if __name__ == "__main__":
    main()
