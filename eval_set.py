"""Score a checkpoint on the held-out test set (test/labels_synth.json).

Labels are percent coords: {"image": "plan_0000.png", "boxes": [{x,y,w,h,label,text}]}
where x/y/w/h are 0-100 percent of the image.

  python3 eval_set.py --ckpt ./ckpt-v2-export --set test
  python3 eval_set.py --ckpt base --set test          # baseline comparison
"""
import argparse, json, os, statistics as st, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig
from parse_out import parse

BASE_ID = "nvidia/NVIDIA-Nemotron-Parse-2.0"
NEW = ["<class_Wall>", "<class_Door>", "<class_Window>", "<class_Room>"]
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def norm_text(s):
    return " ".join((s or "").lower().replace("×", "x").split())


def dims_numeric(s):
    """'Bedroom 3.90x1.98 m' -> (3.90, 1.98); None if no dims."""
    from parse_out import DIMS
    m = DIMS.search(s or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def dims_close(a, b, tol=0.05):
    pa, pb = dims_numeric(a), dims_numeric(b)
    if not pa or not pb:
        return False
    return all(abs(x - y) <= tol * max(abs(y), 1e-6) for x, y in zip(pa, pb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./ckpt-v2-export")
    ap.add_argument("--set", default="test")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(BASE_ID, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
    tok.add_tokens(NEW)
    gen = GenerationConfig.from_pretrained(BASE_ID, trust_remote_code=True)
    src = BASE_ID if a.ckpt == "base" else a.ckpt
    model = AutoModel.from_pretrained(src, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()

    labels = json.load(open(os.path.join(a.set, "labels_synth.json")))
    ious, hits50, hits75, txt_ok, dims_ok, n_gt, n_pred = [], 0, 0, 0, 0, 0, 0
    per_image = []

    for rec in labels:
        path = os.path.join(a.set, "png", rec["image"])
        img = Image.open(path).convert("RGB")
        w, h = img.size
        gt = []
        for b in rec["boxes"]:
            x0 = b["x"] / 100 * w
            y0 = b["y"] / 100 * h
            gt.append({"bbox": [x0, y0, x0 + b["w"] / 100 * w, y0 + b["h"] / 100 * h],
                       "text": b.get("text", ""), "label": b.get("label", "Room")})
        enc = proc(images=[img], text=PROMPT, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, generation_config=gen,
                                 max_new_tokens=a.max_new_tokens)
        pred = parse(tok.decode(out[0], skip_special_tokens=False))

        n_gt += len(gt); n_pred += len(pred)
        used, img_ious = set(), []
        for g in gt:
            best, bi = 0.0, -1
            for j, p in enumerate(pred):
                if j in used:
                    continue
                v = iou(g["bbox"], p["bbox"])
                if v > best:
                    best, bi = v, j
            if bi >= 0 and best >= 0.5:
                used.add(bi)
                hits50 += 1; img_ious.append(best); ious.append(best)
                if best >= 0.75:
                    hits75 += 1
                if norm_text(g["text"]) and norm_text(g["text"]) in norm_text(pred[bi]["text"]):
                    txt_ok += 1
                if dims_close(g["text"], pred[bi]["text"]):
                    dims_ok += 1
        per_image.append((rec["image"], len(gt), len(pred), len(img_ious)))
        print(f"{rec['image']}: gt={len(gt)} pred={len(pred)} matched={len(img_ious)}")

    print(f"\nimages={len(labels)} gt_objects={n_gt} pred_objects={n_pred}")
    print(f"recall@0.5={hits50}/{n_gt} ({hits50/n_gt:.1%})  "
          f"recall@0.75={hits75}/{n_gt} ({hits75/n_gt:.1%})")
    print(f"precision@0.5={hits50}/{n_pred} ({hits50/n_pred:.1%})" if n_pred else "")
    print(f"mean_IoU={st.mean(ious):.3f}" if ious else "mean_IoU=n/a")
    print(f"dim_text_exact={txt_ok}/{hits50} ({txt_ok/hits50:.1%})" if hits50 else "")
    print(f"dims_within_5pct={dims_ok}/{hits50} ({dims_ok/hits50:.1%})" if hits50 else "")


if __name__ == "__main__":
    main()
