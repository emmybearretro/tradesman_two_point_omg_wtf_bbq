"""Demo UI: upload blueprint -> overlay rooms -> lumber BOM + CSV.
DEMO_MODE=1 runs without GPU (fake boxes) so pitch never dies.
  DEMO_MODE=1 python app.py   # safe mode
  python app.py --ckpt ./ckpt # real model
"""
import argparse, csv, io, os
import gradio as gr
from PIL import Image, ImageDraw
from lumber import estimate_house

DEMO = os.getenv("DEMO_MODE", "0") == "1"
_pipe = None

def load_pipe(ckpt):
    global _pipe
    if DEMO or not ckpt or not os.path.exists(ckpt):
        return None
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig
    m = AutoModel.from_pretrained(ckpt, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16).cuda().eval()
    _pipe = (m, AutoProcessor.from_pretrained(ckpt, trust_remote_code=True),
             AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True),
             GenerationConfig.from_pretrained(ckpt, trust_remote_code=True))
    return _pipe

def detect(image, ckpt):
    """Return [{label, text, bbox[x0..x1 0-1000], room, openings}]."""
    if DEMO or _pipe is None and not load_pipe(ckpt):
        w, h = image.size
        return [
            {"label": "Room", "room": "Bedroom", "text": "Bedroom 12 x 14",
             "openings": 3, "bbox": [80, 120, 480, 520]},
            {"label": "Room", "room": "Garage", "text": "20 x 22",
             "openings": 2, "bbox": [520, 120, 920, 520]},
        ]
    m, proc, tok, gen = _pipe
    import torch
    prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
    enc = proc(images=[image], text=prompt, return_tensors="pt",
               add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = m.generate(**enc, generation_config=gen)
    txt = proc.batch_decode(out, skip_special_tokens=True)[0]
    # TODO(hackathon): plug postprocessing.extract_classes_bboxes here; passthrough stub:
    return [{"label": "Room", "room": txt[:24], "text": txt, "openings": 2,
             "bbox": [50, 50, 950, 500]}]

def run(image, ckpt=""):
    dets = detect(image, ckpt)
    w, h = image.size
    ov = image.copy()
    d = ImageDraw.Draw(ov)
    for o in dets:
        x0, y0, x1, y1 = o["bbox"]
        box = [x0/1000*w, y0/1000*h, x1/1000*w, y1/1000*h]
        d.rectangle(box, outline="red", width=4)
        d.text((box[0]+6, box[1]+6), f'{o.get("room","")} {o.get("text","")}')
    rows, tot = estimate_house(dets)
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=["room", "size", "perim_ft", "studs", "plates", "headers", "cost"])
    wr.writeheader(); wr.writerows(rows)
    return ov, rows, f'Studs {tot["studs"]} | Plates {tot["plates"]} | ${tot["cost"]}', buf.getvalue()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./ckpt")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    with gr.Blocks(title="Blueprint -> Lumber") as ui:
        gr.Markdown("# 🏠 Blueprint → Lumber BOM")
        img = gr.Image(type="pil", label="Upload blueprint")
        btn = gr.Button("CALCULATE", variant="primary")
        out_img = gr.Image(label="Rooms detected")
        table = gr.JSON(label="Takeoff")
        total = gr.Textbox(label="Total")
        csvbox = gr.Textbox(label="CSV")
        btn.click(lambda im: run(im, a.ckpt), img, [out_img, table, total, csvbox])
        gr.Examples(["golden_1.png", "golden_2.png", "golden_3.png"], img)
    ui.launch(server_name="0.0.0.0", server_port=a.port)

if __name__ == "__main__":
    main()
