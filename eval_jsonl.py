"""Score a checkpoint on OUR jsonl format (0-1000 boxes), e.g. held-out real plans.

  python3 build_real.py --out data/real_holdout.jsonl --holdout 8 --only-holdout
  python3 eval_jsonl.py --ckpt ./ckpt-v5-export --jsonl data/real_holdout.jsonl

Reports room recall/precision at IoU .5/.75, mean IoU, and dimension-text agreement.
"""
import argparse, json, statistics as st, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig
from parse_out import parse, DIMS
from classes import CLASS_TOKENS

BASE_ID = "nvidia/NVIDIA-Nemotron-Parse-2.0"
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dims(s):
    m = DIMS.search(s or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def dims_close(a, b, tol=0.05):
    pa, pb = dims(a), dims(b)
    return bool(pa and pb) and all(abs(x - y) <= tol * max(abs(y), 1e-6) for x, y in zip(pa, pb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--tile", type=int, default=0, help="tile sheets larger than this (px)")
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(BASE_ID, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
    tok.add_tokens(CLASS_TOKENS)
    gen = GenerationConfig.from_pretrained(BASE_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.ckpt, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()

    def run(img):
        if a.tile and max(img.size) > a.tile:
            from tiles import run_tiled
            return run_tiled(lambda c: _infer(c), img, tile=a.tile)
        return _infer(img)

    def _infer(img):
        enc = proc(images=[img.convert("RGB")], text=PROMPT, return_tensors="pt",
                   add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, generation_config=gen,
                                 max_new_tokens=a.max_new_tokens)
        return parse(tok.decode(out[0], skip_special_tokens=False))

    rows = [json.loads(l) for l in open(a.jsonl)]
    ious, h50, h75, dimok, ngt, npred, nroom_pred = [], 0, 0, 0, 0, 0, 0
    for r in rows:
        img = Image.open(r["image"]).convert("RGB")
        pred = run(img)
        npred += len(pred)
        nroom_pred += sum(1 for p in pred if p.get("label") == "Room")
        used = set()
        for g in r["objects"]:
            if g.get("class") != "Room":
                continue
            ngt += 1
            best, bi = 0.0, -1
            for j, p in enumerate(pred):
                if j in used:
                    continue
                v = iou(g["bbox"], p["bbox"])
                if v > best:
                    best, bi = v, j
            if bi >= 0 and best >= 0.5:
                used.add(bi); h50 += 1; ious.append(best)
                h75 += best >= 0.75
                if dims_close(g.get("text", ""), pred[bi].get("text", "")):
                    dimok += 1
        print(f"{r['image'].split('/')[-1]}: gt={sum(1 for g in r['objects'] if g.get('class')=='Room')} "
              f"pred={len(pred)} (rooms {sum(1 for p in pred if p.get('label')=='Room')}) matched={len(used)}")

    print(f"\nplans={len(rows)} room_gt={ngt} pred_total={npred}")
    print(f"recall@.5={h50}/{ngt} ({h50/max(ngt,1):.1%})  recall@.75={h75}/{ngt} ({h75/max(ngt,1):.1%})")
    print(f"precision_rooms={h50}/{nroom_pred} ({h50/max(nroom_pred,1):.1%})")
    print(f"mean_IoU={st.mean(ious):.3f}" if ious else "mean_IoU=n/a")
    print(f"dims_within_5pct={dimok}/{h50} ({dimok/max(h50,1):.1%})")


if __name__ == "__main__":
    main()
