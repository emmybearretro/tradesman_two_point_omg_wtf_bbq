# 📕 RUNBOOK — from zero to trained demo (10h version)

## A. Your steps (in order, check off)

- [ ] **1. Sign up:** `brev.nvidia.com` (NVIDIA account). $125 credits land in your org.
- [ ] **2. Login locally:** `brev login --skip-browser` ← JWT from console `settings/cli`. (CLI `v0.6.337` already installed here.)
- [ ] **3. Launch 2 boxes** (console → New Instance):
  - `train` — H100, Lambda, $1.99/hr (all training)
  - `side` — L4, $0.17/hr (labels, Gradio, backup)
- [ ] **4. Ship repo:** push `/home/hc/hackathon` to private GitHub → clone to `/home/ubuntu/workspace/` on both boxes.
- [ ] **5. Setup both:** `cd hackathon && bash setup-brev.sh` (~10 min: torch cu128, deps, model pre-download).
- [ ] **6. Arabic zero-shot (5 min, decides data):** run base Parse-2.0 on `blueprints/houses/house_000.png`. Readable Arabic dims → keep EG data. Gibberish → train `synth + usa` only (`cat data/synth.jsonl data/usa.jsonl > data/train.jsonl`).
- [ ] **7. Train v1:** `tmux new -s train && bash run-train-fast.sh ./data/train.jsonl ./ckpt-v1 4` (H100 ~1-2h, unattended).
- [ ] **8. Demo check:** `python app.py --ckpt ./ckpt-v1` → upload `blueprints/usa/usa_000.png`. Working → **record backup video NOW**.
- [ ] **9. Polish:** more labels → v2 if time; else freeze. `DEMO_MODE=1 python app.py` always works with zero GPU.
- [ ] **10. Pitch:** live upload → BOM + $ total → CSV export screenshot. Touch nothing 2h before.

Spot: NEVER for `train` (reclaim kills runs). STOP boxes when you sleep (disk persists, DELETE wipes).

## B. Training-data provenance (everything NOT generated)

| # | Source | URL | License | Taken | Local |
|---|---|---|---|---|---|
| 1 | Voxel51/FloorPlanCAD (test split, 5308) | `huggingface.co/datasets/Voxel51/FloorPlanCAD` | HF tag `cc-by-sa-4.0`; card text says non-commercial ⚠️ | 30 PNGs 1000×1000 (symbols pre-train; boxes available via `--coco`, not yet ingested) | `blueprints/floorplancad/` |
| 2 | Ahmed167/floor-plans-dataset (31) | `huggingface.co/datasets/Ahmed167/floor-plans-dataset` | **NONE listed** ⚠️ — looks like one Alexandria firm's portfolio (same engineer stamp); hackathon-only, replace before commercial use | 29 labeled (019/021 skipped: unreadable thumbnails) | `blueprints/houses/` |
| 3 | jprve/FloorPlansV2 (2831, CubiCasa5k-derived) | `huggingface.co/datasets/jprve/FloorPlansV2` / orig `zenodo.org/records/2613548` (CC) | Upstream CubiCasa5k = Creative Commons; mirror has no separate tag ⚠️ | 20 PNGs (4 dimensioned US/AU + 16 Finnish) | `blueprints/usa/` |
| 4 | Nithins03/us-architectural-floorplan-sft (12k) | `huggingface.co/datasets/Nithins03/us-architectural-floorplan-sft` | Apache-2.0 | **NOT in train** (text-only, no images) — referenced for US room-size validation | — |

Generated in-repo: `synth/` (200 PNG+SVG via `gen_synth.py`, perfect boxes) → `data/synth.jsonl`.
Labels: AI-estimated boxes (±5%) + transliterated dims, `data/labels_*.json` → `data/full.jsonl` (49 houses, 649 boxes).
Final: `data/train.jsonl` = **249 images / 1667 objects** (200 synth + 49 real).

## C. Key commands

```bash
brev ls | brev shell <name>          # status / ssh by name
bash setup-brev.sh                   # per-box setup
bash run-train-fast.sh ./data/train.jsonl ./ckpt-v1 4   # H100 train
bash run-train.sh                    # L4 fallback (A100: same, slower)
python app.py --ckpt ./ckpt-v1       # demo (:7860)
DEMO_MODE=1 python app.py            # bulletproof backup demo
python lumber.py                     # calculator self-test
```
