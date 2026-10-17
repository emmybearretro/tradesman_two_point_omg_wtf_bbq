# TRAINING — HOW TO RUN IT

Last updated: 2026-10-17 17:50 UTC
Box: `superior-harlequin-bison` (H200, RUNNING) — `brev exec superior-harlequin-bison -- ...`
Repo on box: `/home/ubuntu/hackathon` · Repo on GitHub: `emmybearretro/tradesman_two_point_omg_wtf_bbq` (`master`)

---

## TL;DR (the whole thing, 3 commands)

```bash
# 1. get to the box
brev shell superior-harlequin-bison
cd /home/ubuntu/hackathon && git pull -q

# 2. patch the model's remote code (REQUIRED — see §3) and make sure data exists
python3 fix_remote_code.py          # idempotent, prints "patched" or "already patched"
ls -l data/train_v1.jsonl           # must exist; if not, rebuild with §5

# 3. launch detached and walk away
setsid nohup python3 train_lora.py --train ./data/train_v1.jsonl --out ./ckpt-v1 \
    --epochs 4 --batch 1 --grad-ckpt 0 --full 1 --lr 2e-5 > train_v1.log 2>&1 < /dev/null &
```

Then watch it with `tail -f train_v1.log`. Done when the log ends with `TRAIN_DONE` and `./ckpt-v1/` has weights.

---

## 1. Is it running right now?

```bash
pgrep -af train_lora.py                       # one python3 line = running
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
tail -c 400 train_v1.log | tr '\r' '\n' | tail -2
```

Healthy looks like: ~28 GB VRAM, 95–100% GPU util, a progress bar like `24/112 [02:46<08:51, 6.04s/it]`.
Zero VRAM + no process = it died; read the end of `train_v1.log`.

## 2. Expected timeline

112 optimizer steps total (4 epochs, 220 images, effective batch 8 via grad-accum 8).
~6–7 s/step on the H200 → **roughly 12–14 minutes end to end**.

## 3. The crash that was blocking us (fixed)

Crash #4 in the old handoff was diagnosed as a PEFT problem. It was not.
The model's *own* remote code, in its training branch, does this:

```python
decoder_inputs_embeds = decoder_outputs.inputs_embeds     # hf_nemotron_parse_modeling.py:603
```

transformers 5.x removed `inputs_embeds` from `BaseModelOutputWithPastAndCrossAttentions`,
so the first training forward raises `AttributeError` — with PEFT, with full fine-tune,
with any batch size. That is why every launch died at step 0.

**Fix:** `fix_remote_code.py` rewrites that line in the HF cache to recompute the same
tensor from the decoder inputs:

```python
decoder_inputs_embeds = getattr(decoder_outputs, "inputs_embeds", None)
if decoder_inputs_embeds is None:
    decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids)
```

`embed_tokens` is the same scaled embedding the decoder used internally, so this is
numerically equivalent to what transformers 4.x handed back. The script is idempotent and
targets `~/.cache/huggingface/modules/transformers_modules/nvidia/**/hf_nemotron_parse_modeling.py`.

**It must be re-run on any fresh box** (the cache is per-machine). Add it to `setup-brev.sh`
when convenient.

## 4. Do not use tmux for this

The previous run was killed mid-training (`KeyboardInterrupt`, step 59/112) because the tmux
server was born inside an SSH session and died with it. `setsid nohup ... < /dev/null &`
puts the job in its own session; it survives disconnects, closed laptops, and tool calls dying.

To stop a run deliberately: `pkill -f train_lora.py`.

## 5. Training data — rebuild only if `data/train_v1.jsonl` is missing

Expected: **220 images / 1167 objects** = 200 synth + 20 USA. jsonl + synth are gitignored.

```bash
python3 gen_synth.py --n 200 --out ./synth --seed 7
python3 -c 'import os,glob; b=os.path.abspath("blueprints"); os.makedirs(b+"/all",exist_ok=True); [os.symlink(os.path.join(b,"houses" if "/houses/" in f else "usa",os.path.basename(f)), os.path.join(b,"all",os.path.basename(f))) for f in glob.glob("blueprints/houses/*.png")+glob.glob("blueprints/usa/*.png")]'
python3 -c 'import json,glob; r=[]; [r.extend(json.load(open(f))) for f in sorted(glob.glob("data/labels_usa*.json"))]; json.dump(r,open("data/usa_all.json","w"))'
python3 data_convert.py --labelstudio ./synth/labels_synth.json --imgdir ./synth/png --out ./data/synth.jsonl
python3 data_convert.py --labelstudio data/usa_all.json --imgdir ./blueprints/all --out ./data/usa.jsonl
cat data/synth.jsonl data/usa.jsonl > data/train_v1.jsonl
wc -l data/train_v1.jsonl      # expect 220
```

`blueprints/all/` symlinks must be absolute (done via `os.path.abspath` above). Never commit `all/`.

## 6. When it finishes

```bash
tail -c 500 train_v1.log                     # last line: TRAIN_DONE
du -sh ckpt-v1 && ls ckpt-v1                 # full-ft weights, expect several GB
```

Then, in order:
1. Sanity-infer one image with the new checkpoint (compare against `DEMO_MODE=1` baseline).
2. Record the backup demo video while it works — `DEMO_MODE=1` always works regardless.
3. QC the 3 USA goldens before the pitch (the human-drawn boxes are eye-estimated ±5%).
4. Wire `extract_classes_bboxes` postprocessing into `infer.py` / `app.py` — the regex path is demo-grade.

## 7. If it dies again

| Symptom in log | Cause | Do this |
|---|---|---|
| `AttributeError: ... 'inputs_embeds'` | remote-code patch missing | `python3 fix_remote_code.py`, relaunch |
| `ImportError: HybridCache` | peft too old | `python3 -m pip install -U 'peft>=0.21'` |
| `stack expects equal size` | collator regressed | check `make_collate()` padding in `train_lora.py` |
| `CUDA out of memory` | batch too big / ckpt on | keep `--batch 1 --grad-ckpt 0` |
| `NameError: logger` in `gradient_checkpointing_enable` | grad-ckpt + transformers 5 | already avoided (`--grad-ckpt 0`) |
| process gone, no traceback | launched under tmux/SSH | relaunch with `setsid nohup` |

## 8. Cost

H200 burns while the box is up, training or not. Stop the box when sleeping (`brev stop superior-harlequin-bison`) — disk persists. Spend to date ≈ $8–12 of $125.
