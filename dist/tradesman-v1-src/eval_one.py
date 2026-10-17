"""One-image eval: base model vs fine-tuned checkpoint, same prompt/processor.

Usage:
  python3 eval_one.py --ckpt ./ckpt-v1 --image blueprints/usa/usa_000.png
  python3 eval_one.py --ckpt base --image blueprints/usa/usa_000.png

Processor/tokenizer/generation config always come from the BASE repo: save_pretrained()
on this model does not persist the image processor, so ./ckpt-v1 has no preprocessor.
Weights come from --ckpt ("base" = untouched base repo).
"""
import argparse, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig

BASE = "nvidia/NVIDIA-Nemotron-Parse-2.0"
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()
    src = BASE if a.ckpt == "base" else a.ckpt

    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    # the fine-tune appended class tokens; the BASE tokenizer doesn't know them, so
    # decode() silently DROPS their ids (ids past the base vocab-end). Add them back.
    tok.add_tokens(["<class_Wall>", "<class_Door>", "<class_Window>", "<class_Room>"])
    gen = GenerationConfig.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModel.from_pretrained(src, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()

    img = Image.open(a.image).convert("RGB")
    enc = proc(images=[img], text=PROMPT, return_tensors="pt", add_special_tokens=False)
    enc = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(**enc, generation_config=gen,
                             max_new_tokens=a.max_new_tokens)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(text)
    print(f"\n--- {a.ckpt} | {a.image} | {len(text.splitlines())} lines, {len(text)} chars")


if __name__ == "__main__":
    main()
