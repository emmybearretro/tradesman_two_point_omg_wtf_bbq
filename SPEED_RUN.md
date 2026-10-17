# ⚡ SPEED RUN — spend ~$40 of the $125, finish in half the time

Even going fast you won't crack $50. $125 never binds. GPU was never the bottleneck — labelling is.

## The fleet (run 2 instances in parallel)

| Instance | GPU | $/hr | Job |
|---|---|---|---|
| `train` | H100 (Lambda) | $1.99 | all training, nothing else |
| `side` | L4 | $0.17 | data_convert, labelling host, Gradio build |

Two instances = labelling + training overlap instead of serialize. Combined burn $2.16/hr → 15h = ~$32.

## Playbook

1. **Launch both now.** `train` = H100, `side` = L4. `bash setup-brev.sh` on both (~10 min, cached torch).
2. **Sweep in parallel, don't guess.** On `train`, run 2 configs back-to-back (H100 chews 2k rows × 4 epochs in ~1-2h each):
   `bash run-train-fast.sh ./data/train.jsonl ./ckpt-lr2e4 4` then lr variant. Keep the winner, delete the loser.
3. **Label while it trains.** Side instance never idles: annotate next 50 while H100 burns. This overlap is where the hours come from, not GPU speed.
4. **Scale, don't rebuild.** Need more VRAM? Console → Hardware slider on `train`. Disk follows.
5. **STOP both when you sleep.** Resume = 2 min. A full 24h wall-clock day costs ~$25-40, not $125.

## Rules

- H100 primary because $1.99/hr is absurd value (vs GCP A100-80GB at $6.21 — never touch that).
- Spot: still NO for `train` (reclaim kills runs). Spot L4 side is fine for labelling.
- If H100 waitlisted in your region: fall back to A100 40GB $1.10, same scripts with `run-train.sh`.
