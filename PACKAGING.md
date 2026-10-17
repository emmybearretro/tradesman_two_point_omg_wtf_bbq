# PACKAGING — what goes where

Two artifacts, shipped separately because GitHub will not take the weights.

| artifact | size | destination |
|---|---|---|
| source bundle (`dist/tradesman-v1-src.zip`) | 17 MB | GitHub repo |
| weights (`ckpt-v2-export/`, 8 files) | 1.7 GB | GitHub **Release asset** (2 GB/file limit) or Hugging Face Hub |
| full trainer output (`ckpt-v2/`) | 12 GB | **do not ship** — 5 GB of it is `optimizer.pt` |

Build the source bundle: `./package.sh` → `dist/tradesman-v1-src.{tar.gz,zip}`.
Build the weights bundle (on the box): see "Export the weights" below.

## Why the weights can't just be committed

GitHub blocks files > 100 MB outright; Git LFS free tier is 1 GB storage + 1 GB/month
bandwidth, and this model is 1.7 GB before anyone downloads it. A **release asset** is the
free path (2 GB per file), Hugging Face Hub is the sane path if you want `from_pretrained`
to work by name.

## Export the weights (on the Brev box)

```bash
cd /home/ubuntu/hackathon
rm -rf ckpt-v2-export && mkdir ckpt-v2-export
cp ckpt-v2/{model.safetensors,config.json,generation_config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja,hf_nemotron_parse_config.py,hf_nemotron_parse_modeling.py} ckpt-v2-export/
chmod 644 ckpt-v2-export/model.safetensors
tar -cf ckpt-v2-export.tar ckpt-v2-export       # 1.81 GB, store don't gzip: the
                                                # weights are incompressible and gzip
                                                # takes minutes for ~0% saving
sha256sum ckpt-v2-export.tar
# 693a5b1b6254b5dd12508e633cb638844084d1c9fd22e547280106654db489ad  (ckpt-v2)
```

Pull it down: `scp` / `rsync` from the box, or upload straight from the box.

## Option A — GitHub release asset (recommended)

```bash
gh release create v1-demo ckpt-v2-export.tar \
   --title "v1 demo weights" \
   --notes "Nemotron-Parse-2.0 full fine-tune, 20 epochs, 220 plans. Room boxes + dims."
```

Then in the README: download, `tar xf ckpt-v2-export.tar`, `python infer.py --ckpt ./ckpt-v2-export ...`.

## Option B — Hugging Face Hub (nicer: loads by name)

```bash
pip install -U huggingface_hub
huggingface-cli login                       # needs a write token
huggingface-cli upload <your-user>/tradesman-parse-v1 ckpt-v2-export . --repo-type model
```

`AutoModel.from_pretrained("<your-user>/tradesman-parse-v1", trust_remote_code=True)` then works.

## Option C — plain git push (code only)

```bash
cd /home/hc/hackathon
git add -A -- . ':!dist' ':!uploads' ':!test'
git commit -m "v1: native-format training, robust inference, packaging"
git push origin master
```

`ckpt*/`, `synth/`, `data/*.jsonl`, `*.log`, `blueprints/all/` are already gitignored, so the
repo stays ~17 MB.

## What it runs in

Runtime (the demo, `app.py`):

    python 3.12
    torch (cu128 build)  +  transformers==5.6.1  +  accelerate==1.12.0  +  gradio>=5
    timm, open_clip_torch, einops, pillow, beautifulsoup4
    pip install -r requirements.txt

    python app.py --ckpt ./ckpt-v2-export          # Gradio UI on :7860, needs CUDA
    DEMO_MODE=1 python app.py                      # no GPU, fake boxes, never crashes on stage
    python infer.py --ckpt ./ckpt-v2-export --image plan.png --out bom.csv   # CLI -> BOM
    python infer.py --ckpt ./ckpt-v2-export --image plan.png --cpu           # no-GPU fallback
    python eval_one.py --ckpt ./ckpt-v2-export --image plan.png              # raw model output
    python eval_score.py --ckpt ./ckpt-v2-export --n 8                       # IoU + text score

Hardware:

* Training was done on 1x H200 141 GB (full fine-tune, batch 1, no grad-checkpoint).
* Inference needs ~3-4 GB VRAM at bf16/fp16 (0.9 B params → 1.7 GB weights + activations
  at 1536 px). `infer.py` loads → runs one image → unloads, so a 4 GB laptop card works.
* CPU works (`--cpu`, fp32, slow: tens of seconds to minutes per plan).

## Before you ship — known limits

1. The model emits **Room** boxes only. Training data was 1157 Room + 10 Window; there are
   **zero Wall and zero Door examples**, so it cannot produce them. Wall/door detection is
   the v2 data problem, not a tuning problem.
2. Trained and scored on the same 220 images (35/35 matched, mean IoU 0.899, dims 35/35).
   No held-out split exists yet — hold out 20 before quoting accuracy on stage.
3. `optimizer.pt` inside `ckpt-v2/checkpoint-560` is only needed to resume training. Never
   ship it; it is 3.6 GB of Adam state.
