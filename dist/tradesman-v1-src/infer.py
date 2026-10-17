#!/usr/bin/env python3
"""4GB-VRAM laptop inference: load -> process ONE image -> unload -> BOM.
Swaps the model out after every image so the GPU is free for the next step.

  python infer.py --ckpt ./ckpt-v1 --image plan.png --out bom.csv

Needs: torch (any CUDA), transformers, pillow. No gradio, no peft.
Download ckpt from the Brev box: scp / rsync ./ckpt-v1 down (~2GB fp16).
"""
import argparse, csv, gc
import torch
from PIL import Image
from lumber import estimate_house

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
MAX_SIDE = 1536  # Parse likes <=1664; 1536 keeps 4GB headroom

def load(ckpt, use_cuda):
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig
    dt = torch.float16 if use_cuda else torch.float32
    m = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=dt)
    if use_cuda:
        m = m.cuda()
    m.eval()
    # Processor is never fine-tuned (weights + tokenizer are); ckpt dirs ship none.
    # Always use base: a ckpt-loaded processor silently yields broken [1,1] inputs.
    processor = AutoProcessor.from_pretrained("nvidia/NVIDIA-Nemotron-Parse-2.0",
                                              trust_remote_code=True)
    return (m, processor,
            AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True),
            GenerationConfig.from_pretrained(ckpt, trust_remote_code=True))

def unload(pipe, use_cuda):
    del pipe
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()  # model fully out; VRAM free for next swap

@torch.inference_mode()
def extract(pipe, image, use_cuda):
    m, proc, tok, gen = pipe
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    enc = proc(images=[image.convert("RGB")], text=PROMPT,
               return_tensors="pt", add_special_tokens=False)
    if use_cuda:
        enc = {k: v.cuda() for k, v in enc.items()}
    out = m.generate(**enc, generation_config=gen)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="bom.csv")
    ap.add_argument("--cpu", action="store_true", help="force CPU (slow, any laptop)")
    a = ap.parse_args()
    use_cuda = torch.cuda.is_available() and not a.cpu
    print(f"device: {'cuda/fp16' if use_cuda else 'cpu/fp32'}")

    pipe = load(a.ckpt, use_cuda)          # model IN
    try:
        text = extract(pipe, Image.open(a.image), use_cuda)
    finally:
        unload(pipe, use_cuda)             # model OUT (swap)
    print("raw:", text[:300])

    # v1: whole-image text -> one Room entry per dim pair found (see README: wire real boxes next)
    import re
    dets = [{"label": "Room", "room": t[:24], "text": t, "openings": 2}
            for t in re.findall(r".{0,24}\d+\s*(?:'|ft|m)?\s*(?:x|×)\s*\d+.*", text)]
    rows, tot = estimate_house(dets)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["room", "size", "perim_ft", "studs", "plates", "headers", "cost"])
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rooms -> {a.out} | Studs {tot['studs']} ${tot['cost']}")

if __name__ == "__main__":
    main()
