"""Probe one sheet (no ground truth): tiled inference, per-class counts, sample dets.

  python3 probe_sheet.py --ckpt ./ckpt-v5-export --image rq_p2.png --tile 1536
"""
import argparse, collections, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig
from parse_out import parse
from classes import CLASS_TOKENS

BASE_ID = "nvidia/NVIDIA-Nemotron-Parse-2.0"
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=192)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--show", type=int, default=14)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(BASE_ID, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
    tok.add_tokens(CLASS_TOKENS)
    gen = GenerationConfig.from_pretrained(BASE_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.ckpt, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()

    def infer(img):
        enc = proc(images=[img.convert("RGB")], text=PROMPT, return_tensors="pt",
                   add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, generation_config=gen,
                                 max_new_tokens=a.max_new_tokens)
        return parse(tok.decode(out[0], skip_special_tokens=False))

    img = Image.open(a.image).convert("RGB")
    print(f"sheet {img.size[0]}x{img.size[1]}")
    if a.tile and max(img.size) > a.tile:
        from tiles import run_tiled, iter_tiles
        n = sum(1 for _ in iter_tiles(img, a.tile, a.overlap))
        print(f"tiling into {n} crops of {a.tile}px")
        dets = run_tiled(infer, img, tile=a.tile, overlap=a.overlap, verbose=True)
    else:
        dets = infer(img)
    c = collections.Counter(d.get("label", "?") for d in dets)
    print(f"\n{len(dets)} dets: {dict(c)}")
    for d in dets[: a.show]:
        b = [round(v) for v in d["bbox"]]
        print(f"  {d.get('label','?'):9s} {b} {d.get('text','')!r}")


if __name__ == "__main__":
    main()
