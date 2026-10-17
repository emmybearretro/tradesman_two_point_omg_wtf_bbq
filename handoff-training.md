# 🔥 TRAINING HANDOFF — everything, no context needed

## 1. Current state (2026-10-17 ~17:10 UTC)

- **Training is NOT running.** Last launch was aborted before execution.
- **Box:** `superior-harlequin-bison` (Brev, H200 141GB, RUNNING). Org `draditch-bf5f76-03bu`.
- **Repo on box:** `/home/ubuntu/hackathon` (branch `master`, pulled through `4f9a948` — re-pull before running).
- **Repo on GitHub:** `emmybearretro/tradesman_two_point_omg_wtf_bbq`, branch `master`.
- **Local:** `/home/hc/hackathon` (authoritative for code/labels; box is generated-data + ckpts).

## 2. What is proven working

- `lumber.py` parser suite 5/5: feet, ft-in (`10'-3"`), metric (`m`), cm/mm guards, HxW-tag rejection.
- `data/train_v1.jsonl` recipe: **220 images / 1167 objects** = 200 synth + 20 USA (see §5 to rebuild — jsonl/synth are gitignored, regenerate on any fresh box).
- Demo path end-to-end in `DEMO_MODE=1` (found + fixed a real CSV bug this way).
- `infer.py`: 4GB-VRAM laptop runner (load → 1 image → unload). Needs `extract_classes_bboxes` wiring in harness (currently regex fallback).
- `setup-brev.sh` hardened: bootstraps pip (ensurepip→apt), `PIP_BREAK_SYSTEM_PACKAGES=1`, `python3 -m pip` everywhere.
- Box deps installed: torch 2.11.0+cu128 (CUDA True), transformers 5.6.1, peft 0.21.2, all requirements.

## 3. Training crash log (all four, in order)

1. **peft 0.17.0 vs transformers 5.6.1** — `ImportError: HybridCache`. Fixed: `requirements.txt` → `peft>=0.21`; box upgraded to 0.21.2.
2. **Unpadded labels** — `RuntimeError: stack expects equal size`. Fixed: `make_collate()` in `train_lora.py` (pads input_ids/attention_mask/labels, masks label pads to -100).
3. **PEFT + custom encoder-decoder** — `AttributeError: ... no attribute 'inputs_embeds'`. Decision: skip PEFT on H200, full fine-tune (`--full 1`). 141GB makes LoRA unnecessary.
4. **Full-ft batch 8 OOM** (138GB). Then **grad-ckpt broken**: NVIDIA remote code hits `NameError: logger` inside `gradient_checkpointing_enable()` with transformers 5.x. **Conclusion: batch 1, no ckpt, full-ft** (~20-30GB, fits).

## 4. Resume command (run this next on the box)

```bash
cd /home/ubuntu/hackathon && git pull -q
tmux new -d -s train 'python3 train_lora.py --train ./data/train_v1.jsonl --out ./ckpt-v1 --epochs 4 --batch 1 --grad-ckpt 0 --full 1 --lr 2e-5 > train_v1.log 2>&1; echo TRAIN_DONE >> train_v1.log'
tail -f train_v1.log   # expect 0/112 steps/epoch climbing, loss falling
```

If box is fresh (no `data/train_v1.jsonl`): rebuild first —
```bash
python3 gen_synth.py --n 200 --out ./synth --seed 7
python3 -c 'import os,glob; b=os.path.abspath("blueprints"); os.makedirs(b+"/all",exist_ok=True); [os.symlink(os.path.join(b,"houses" if "/houses/" in f else "usa",os.path.basename(f)), os.path.join(b,"all",os.path.basename(f))) for f in glob.glob("blueprints/houses/*.png")+glob.glob("blueprints/usa/*.png")]'
python3 -c 'import json,glob; r=[]; [r.extend(json.load(open(f))) for f in sorted(glob.glob("data/labels_usa*.json"))]; json.dump(r,open("data/usa_all.json","w"))'
python3 data_convert.py --labelstudio ./synth/labels_synth.json --imgdir ./synth/png --out ./data/synth.jsonl
python3 data_convert.py --labelstudio data/usa_all.json --imgdir ./blueprints/all --out ./data/usa.jsonl
cat data/synth.jsonl data/usa.jsonl > data/train_v1.jsonl
```
⚠️ `blueprints/all/` symlinks MUST use absolute box paths (relative CWD targets break from inside `all/`). Never commit `all/` (gitignored).

## 5. Data map

- Real labels: `data/labels_*.json` (tracked) — 49 houses / 649 boxes: 29 EG (metric, transliterated, eye-estimated ±5%) + 20 USA (4 dimensioned ft/ft-in/mm + 16 Finnish room-only). Skipped: `houses/019,021` (unreadable thumbnails).
- Synth: `gen_synth.py` → perfect geometry/dim-linking boxes, EN labels, ft+metric mix.
- FloorPlanCAD 30 (`blueprints/floorplancad/`): NOT ingested; boxes available via `data_convert.py --coco`. Bench for v2 symbols.
- Provenance + licenses: `RUNBOOK.md` §B (Ahmed167 = no license listed, hackathon-only).

## 6. Key findings that constrain the plan

- **Arabic zero-shot FAILS**: base Parse-2.0 emits 2 tokens (`متر` + coord) then EOS on `house_000.png`; uncapped run burned 5+ min CPU at 2% GPU. EG data benched to v2; **v1 = synth + USA only**.
- Neither Nemotron OCR variant lists Arabic; multilingual trained on mOSCAR incl. `arb_Arab` (per NVIDIA staff, HF discussion #6) — unvalidated, possible v2 path with 50+ samples.
- USA `017/018` are AU mm plans (window/door tags as `Window` objects) — don't call them American on stage.
- Laptop target: 4GB VRAM, fp16 ckpt (~2GB), swap per image (`infer.py`). `--cpu` fallback exists.

## 7. Open items for the human

- [ ] Run §4, watch loss → record backup demo video the moment v1 works (`DEMO_MODE=1` always works regardless).
- [ ] Harness: wire `extract_classes_bboxes` postprocessing into `infer.py`/`app.py` (regex fallback is demo-grade only).
- [ ] QC 3 USA goldens before pitch (my boxes ±5%).
- [ ] v2 if time: EG data (needs Arabic-glyph→EN mapping volume), FloorPlanCAD symbols.
- [ ] STOP box when sleeping (disk persists); H200 burn so far ≈ $8-12 of $125.
