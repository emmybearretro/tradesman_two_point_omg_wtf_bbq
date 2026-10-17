# 🏆 Blueprint → Lumber — hackathon kit

## Brev start (5 min)
1. `brev.nvidia.com → Launch Instance → A100 40GB → pytorch:25.03-py3`
2. `pip install -r requirements.txt`
3. `python -c "from lumber import estimate_house; print('ok')"`

## Pipeline
```
data_convert.py  # FloorPlanCAD + 50 hand labels -> data/train.jsonl
train_lora.py    # Parse-2.0 + LoRA r16 -> ./ckpt
lumber.py        # deterministic takeoff (test: python lumber.py)
app.py           # Gradio demo (safe: DEMO_MODE=1 python app.py)
```

## Training mix (synth + real)
- `python gen_synth.py --n 200 --out ./synth` → perfect-box geometry/dim-linking (SVG+PNG+labels)
- Real: `data/custom.jsonl` (10 EG houses, 147 rooms) + `blueprints/usa/` goldens + FloorPlanCAD symbols
- Merge: `cat data/synth.jsonl data/custom.jsonl > data/train.jsonl` (keep ~20% real minimum)
- `bash run-train-fast.sh ./data/train.jsonl ./ckpt-v1 4` (H100) or `bash run-train.sh` (L4)
- Demo: `python app.py --ckpt ./ckpt-v1` → upload golden USA plan → screenshot BOM

## Pitch insurance
- `DEMO_MODE=1` always works with zero GPU. Record backup video.
- Keep 3 golden blueprints that always parse perfectly.
