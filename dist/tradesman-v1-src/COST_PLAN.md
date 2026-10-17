# $125 Brev battle plan — Parse-2.0 is <1B, don't pay for big iron until final

## Live Brev prices (nvidia.com/en-us/launchables/pricing, check console to confirm)

| GPU | On-demand | Spot | $125 buys | Use for |
|---|---|---|---|---|
| L4 24GB | $0.17/hr | $0.05/hr | 735h / 2500h | dev, data, test trains, demo |
| A100 40GB (Lambda) | $1.10/hr | — | 113h | final overnight train 4-6h = ~$6 |
| H100 (Lambda) | $1.99/hr | — | 62h | emergency burst only, 2h = $4 |
| CPU 2x4 | $0.04/hr | $0.02 | — | idle labelling, scale down to this |

Disk is $0.00013/GB/hr — negligible. STOP keeps disk+workspace, DELETE wipes it.

## Playbook (total ~$10-12, leaves $113 buffer)

1. **Build on L4 on-demand** ($0.17): setup, `data_convert.py`, smoke train 1 epoch, Gradio. ~8h = $1.36.
2. **Test cheap**: L4 spot $0.05 for throwaway runs. Never demo on spot (can be reclaimed).
3. **Final train on A100 40GB** ($1.10): scale same instance up (console → Hardware slider, disk follows, brief restart), run 4 epochs ~5h = $5.50. Scale back to L4 for demo.
4. **Demo on L4 on-demand**: stable + cheap. Record backup video.
5. **STOP, never DELETE** between sessions. `brev login` → `brev ls` → `brev shell <name>`.

## CLI cheat sheet

```bash
brev login --skip-browser   # paste JWT from brev.nvidia.com/settings/cli
brev ls                     # live prices + instances
brev shell <name>           # ssh by name, no IPs
# inside: cd /home/ubuntu/workspace/hackathon && bash setup-brev.sh
tmux new -s train && bash run-train.sh   # survives SSH drop
```

## Why L4 first

Parse-2.0 + LoRA r16 bf16 fits 24GB with grad-checkpointing (see `--grad-ckpt 1`). A100 only buys speed (~1.6x), not capability. Spend the $125 on *iterations + labelling*, not idle H100.
