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

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
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
def extract(pipe, image, use_cuda, max_side=1536, max_new_tokens=512, enc_states=None):
    m, proc, tok, gen = pipe
    if enc_states is not None:
        # precomputed encoder states (see --dump-enc): decoder-only, fits any 4GB GPU
        from transformers.modeling_outputs import BaseModelOutput
        dec_in = {k: v.cuda() for k, v in enc_states.items() if k in ("input_ids", "attention_mask")}
        out = m.generate(**dec_in, encoder_outputs=BaseModelOutput(
            last_hidden_state=enc_states["hidden"].to(dtype=m.dtype).cuda()),
            generation_config=gen, max_new_tokens=max_new_tokens)
        return tok.decode(out[0], skip_special_tokens=False)
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    enc = proc(images=[image.convert("RGB")], text=PROMPT,
               return_tensors="pt", add_special_tokens=False)
    tiny_gpu = use_cuda and torch.cuda.get_device_properties(0).total_memory < 8e9
    if tiny_gpu:
        # 4GB cards (no flash/mem_efficient here): 13k-token encoder attention needs
        # 10GB+ with the math backend. Run the encoder once on CPU, decode on GPU.
        from transformers.modeling_outputs import BaseModelOutput
        m.get_encoder().float().cpu()
        torch.cuda.empty_cache()
        eo = m.get_encoder()(pixel_values=enc["pixel_values"])
        dec_in = {k: (v.cuda() if k != "pixel_values" else v) for k, v in enc.items()}
        out = m.generate(**dec_in,
                         encoder_outputs=BaseModelOutput(
                             last_hidden_state=eo.last_hidden_state.cuda()),
                         generation_config=gen, max_new_tokens=max_new_tokens)
    else:
        if use_cuda:
            enc = {k: v.cuda() for k, v in enc.items()}
        out = m.generate(**enc, generation_config=gen, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--emit-json", default=None, metavar="PATH",
                    help="machine-readable {text, detections, rows, total} for queue workers")
    ap.add_argument("--out", default="bom.csv")
    ap.add_argument("--cpu", action="store_true", help="force CPU (slow, any laptop)")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dump-enc", default=None, metavar="STATES.PT",
                    help="CPU encoder once, save hidden states (slow, offline, run before demo)")
    ap.add_argument("--enc-states", default=None, metavar="STATES.PT",
                    help="skip encoder, GPU-decode from dumped states (fast, demo path)")
    ap.add_argument("--tile", type=int, default=0, metavar="PX",
                    help="tile large sheets into readable crops (e.g. 1536); 0 = off")
    ap.add_argument("--tile-overlap", type=int, default=192)
    a = ap.parse_args()
    use_cuda = torch.cuda.is_available() and not a.cpu
    print(f"device: {'cuda/fp16' if use_cuda else 'cpu/fp32'}")

    pipe = load(a.ckpt, use_cuda)          # model IN
    if a.dump_enc:
        m, proc, tok, gen = pipe
        m.get_encoder().float().cpu()  # encoder lives on CPU for the dump
        torch.cuda.empty_cache()
        im = Image.open(a.image).convert("RGB")
        if max(im.size) > a.max_side:
            im.thumbnail((a.max_side, a.max_side))
        enc = proc(images=[im], text=PROMPT, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            eo = m.get_encoder()(pixel_values=enc["pixel_values"])
        torch.save({"hidden": eo.last_hidden_state,
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc.get("attention_mask")}, a.dump_enc)
        print("dumped", a.dump_enc)
        unload(pipe, use_cuda)
        return
    blob = torch.load(a.enc_states, map_location="cpu", weights_only=True) if a.enc_states else None
    img = Image.open(a.image).convert("RGB")
    dets = None
    try:
        if a.tile or max(img.size) > a.max_side * 1.5:
            # big sheet: read it in tiles at native resolution instead of shrinking the
            # whole page until the dimension strings vanish
            from tiles import run_tiled
            from parse_out import parse as _parse
            side = a.tile or a.max_side
            dets = run_tiled(
                lambda crop: _parse(extract(pipe, crop, use_cuda, side, a.max_new_tokens)),
                img, tile=side, overlap=a.tile_overlap, verbose=True)
            text = ""
            print(f"tiled: {len(dets)} detections on {img.size[0]}x{img.size[1]}")
        else:
            text = extract(pipe, img, use_cuda, a.max_side, a.max_new_tokens, blob)
    finally:
        unload(pipe, use_cuda)             # model OUT (swap)
    print("raw:", text[:300])

    # real per-room boxes + dimension strings from the model's native token output
    from parse_out import parse
    if dets is None:
        dets = parse(text)
    print(f"parsed {len(dets)} rooms")
    rows, tot = estimate_house(dets)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["room", "size", "perim_ft", "studs", "plates", "headers", "cost"])
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rooms -> {a.out} | Studs {tot['studs']} ${tot['cost']}")
    if a.emit_json:
        import json as _json
        _json.dump({"text": text, "detections": dets, "rows": rows, "total": tot},
                   open(a.emit_json, "w"))
        print("wrote", a.emit_json)

if __name__ == "__main__":
    main()
