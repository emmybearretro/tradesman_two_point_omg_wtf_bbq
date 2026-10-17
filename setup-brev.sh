#!/bin/bash
# Run ONCE per Brev instance, inside /home/ubuntu/workspace/hackathon. Idempotent + cached.
# Usage: bash setup-brev.sh
set -e
cd "$(dirname "$0")"

echo "== pip =="; python3 -m ensurepip --default-pip >/dev/null 2>&1 || sudo apt-get install -y -qq python3-pip 2>&1 | tail -1 || true
echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "== torch (cu128) =="
python3 -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1 || true
echo "== repo deps =="
python3 -m pip install -q -r requirements.txt 2>&1 | tail -2 || true
echo "== model pre-download (persists on STOP, not delete) =="
python3 - <<'PY'
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download("nvidia/NVIDIA-Nemotron-Parse-2.0")
    print("cached:", p)
except Exception as e:
    print("skip pre-download:", e)
PY
python3 -c "from lumber import estimate_house; print('lumber ok')"
echo "DONE. Next: bash run-train.sh  |  DEMO_MODE=1 python app.py"
