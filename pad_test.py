"""Does letterboxing to the processor's canvas aspect fix real-plan boxes?

The image processor resizes with aspect ratio and CENTER-PADS to (H=2048, W=1664):
canvas aspect = 1664/2048 = 0.8125. Any image whose aspect differs gets white bands,
and we don't know which coordinate space the model reports. This test pre-letterboxes
the plan to 0.8125 (same bands, but we know exactly where they are), maps predictions
back to the original frame, and scores against ground truth.

  python3 pad_test.py --ckpt ./ckpt-v5-export --jsonl data/real_holdout.jsonl
"""
import argparse, json, statistics as st, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig
from parse_out import parse
from classes import CLASS_TOKENS

BASE_ID = "nvidia/NVIDIA-Nemotron-Parse-2.0"
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
CANVAS_ASPECT = 1664 / 2048


def letterbox(img, aspect=CANVAS_ASPECT, fill=(255, 255, 255)):
    w, h = img.size
    if w / h > aspect:
        nw, nh = w, int(round(w / aspect))
    else:
        nh, nw = h, int(round(h * aspect))
    canvas = Image.new("RGB", (nw, nh), fill)
    ox, oy = (nw - w) // 2, (nh - h) // 2
    canvas.paste(img, (ox, oy))
    return canvas, (ox, oy, w, h)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(BASE_ID, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
    tok.add_tokens(CLASS_TOKENS)
    gen = GenerationConfig.from_pretrained(BASE_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.ckpt, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()

    rows = [json.loads(l) for l in open(a.jsonl)]
    tot_gt = tot_hit = tot_pred = 0
    ious = []
    for r in rows:
        img = Image.open(r["image"]).convert("RGB")
        canvas, box = letterbox(img)
        enc = proc(images=[canvas], text=PROMPT, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, generation_config=gen,
                                 max_new_tokens=a.max_new_tokens)
        # model reports in canvas space (0-1000 of the padded canvas)
        pred = parse(tok.decode(out[0], skip_special_tokens=False))
        for d in pred:
            b = d["bbox"]
            W, H = canvas.size
            d["bbox"] = [(b[0] / 1000 * W - box[0]) / box[2] * 1000,
                         (b[1] / 1000 * H - box[1]) / box[3] * 1000,
                         (b[2] / 1000 * W - box[0]) / box[2] * 1000,
                         (b[3] / 1000 * H - box[1]) / box[3] * 1000]
        tot_pred += len(pred)
        gt = [g for g in r["objects"] if g.get("class") == "Room"]
        tot_gt += len(gt)
        used, hit = set(), 0
        for g in gt:
            best, bi = 0.0, -1
            for j, p in enumerate(pred):
                if j in used:
                    continue
                v = iou(g["bbox"], p["bbox"])
                if v > best:
                    best, bi = v, j
            if bi >= 0 and best >= 0.5:
                used.add(bi); hit += 1; ious.append(best)
        tot_hit += hit
        print(f"{r['image'].split('/')[-1]}: gt={len(gt)} pred={len(pred)} matched={hit}")
    print(f"\nplans={len(rows)} recall@.5={tot_hit}/{tot_gt} "
          f"({tot_hit/max(tot_gt,1):.1%}) mean_IoU={st.mean(ious) if ious else 0:.3f}")


if __name__ == "__main__":
    main()
