#!/usr/bin/env python3
"""Basic image OCR with the tesseract binary (placeholder until the
vision model is trained).

Same shape as transcribe.py: is_ocrable() routes, ocr() extracts,
ocr_file() is the _process_job entry point. No venv subprocess needed
here — tesseract is a system binary, CPU-only, ~100MB RAM.

Usage:
    from ocr import ocr_file
    result_dict = ocr_file("/path/to/plan.jpg", project="default")
"""
import os
import subprocess

MODEL_ID = "tesseract 5.5.3 (eng, psm 6)"
VISION_MODEL_ID = "nemotron-parse-v2/ckpt-v3 (local GPU)"
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}
VISION_CKPTS = ("downloads/ckpt-v3-export", "downloads/ckpt-v2-export")
STATES_DIRNAME = "states"

def _app_dir():
    return os.path.dirname(os.path.abspath(__file__))

def _venv_python():
    venv_py = os.path.join(_app_dir(), ".venv-gpu", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    import sys
    return sys.executable

def _ckpt_dir():
    for c in VISION_CKPTS:
        p = c if os.path.isabs(c) else os.path.join(_app_dir(), c)
        if os.path.isdir(p):
            return p
    return None

def _states_path(path):
    import hashlib
    with open(path, "rb") as fh:
        key = hashlib.md5(fh.read()).hexdigest()
    d = os.path.join(_app_dir(), STATES_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + ".pt")

def _infer_json(args, timeout):
    import json as _json
    cmd = [_venv_python(), os.path.join(_app_dir(), "infer.py")] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"vision infer failed: {(r.stderr or '').strip()[-500:]}")
    return r.stdout

def vision_file(path, project="default", name=None):
    """v3 vision: dump-then-decode (states cached by content hash) -> result dict."""
    import json as _json
    import tempfile
    from results import build_result
    ckpt = _ckpt_dir()
    if ckpt is None:
        raise RuntimeError("no vision checkpoint")
    st = _states_path(path)
    if not os.path.exists(st):
        _infer_json(["--cpu", "--ckpt", ckpt, "--image", path,
                     "--dump-enc", st], timeout=1500)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_json = tf.name
    _infer_json(["--ckpt", ckpt, "--image", path, "--enc-states", st,
                 "--emit-json", out_json, "--max-new-tokens", "512"], timeout=600)
    with open(out_json, encoding="utf-8") as fh:
        data = _json.load(fh)
    os.unlink(out_json)
    try:
        from PIL import Image as _Image
        w, h = _Image.open(path).size
    except Exception:
        w = h = None
    lines = []
    for d in data.get("detections", []):
        b = d.get("bbox", [0, 0, 0, 0])
        if w and h:
            b = [b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h]
        lines.append({"text": (d.get("room", "") + " " + d.get("dims", "")).strip(),
                      "conf": 100.0, "box": [round(v, 1) for v in b]})
    result = build_result(name or os.path.basename(path), project,
                          "image", VISION_MODEL_ID, data.get("text", ""), lines)
    result["bom_rows"] = data.get("rows", [])
    result["bom_total"] = data.get("total", {})
    return result


def is_ocrable(name):
    return os.path.splitext(name or "")[1].lower() in IMAGE_EXTS


def ocr(path, model_id=MODEL_ID):
    """OCR one image -> (text, lines). Lines carry conf + pixel boxes."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    text = _run(path, None).strip()
    lines = _parse_tsv(_run(path, "tsv"))
    return text, lines


def _run(path, ext):
    cmd = ["tesseract", path, "stdout", "-l", "eng",
           "--oem", "1", "--psm", "6"]
    if ext:
        cmd.append(ext)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"tesseract failed: {(r.stderr or '').strip()[-300:]}")
    return r.stdout


def _parse_tsv(tsv):
    # TSV holds text at word level (5); group words into lines.
    lines = []
    rows = tsv.splitlines()
    if not rows:
        return lines
    header = rows[0].split("\t")
    try:
        idx = {k: header.index(k) for k in (
            "level", "line_num", "text", "conf",
            "left", "top", "width", "height")}
    except ValueError:
        return lines
    cur, cur_box, cur_conf = None, None, []
    def flush():
        if cur and cur[1]:
            x0, y0, x1, y1 = cur_box
            lines.append({"text": " ".join(cur[1]).strip(),
                          "conf": round(sum(cur_conf) / len(cur_conf), 1),
                          "box": [x0, y0, x1, y1]})
    for row in rows[1:]:
        cols = row.split("\t")
        if len(cols) < len(header) or cols[idx["level"]] != "5":
            continue
        t = cols[idx["text"]].strip()
        if not t:
            continue
        key = cols[idx["line_num"]]
        try:
            conf = float(cols[idx["conf"]])
            x, y = int(cols[idx["left"]]), int(cols[idx["top"]])
            w, h = int(cols[idx["width"]]), int(cols[idx["height"]])
        except ValueError:
            continue
        if cur is None or cur[0] != key:
            flush()
            cur, cur_box, cur_conf = [key, []], [x, y, x + w, y + h], []
        cur[1].append(t)
        cur_conf.append(conf)
        cur_box[0] = min(cur_box[0], x)
        cur_box[1] = min(cur_box[1], y)
        cur_box[2] = max(cur_box[2], x + w)
        cur_box[3] = max(cur_box[3], y + h)
    flush()
    return lines


def ocr_file(path, project="default", name=None, model_id=MODEL_ID):
    """Image -> result dict. v3 vision when a ckpt is present, else tesseract."""
    if _ckpt_dir() is not None:
        try:
            return vision_file(path, project, name)
        except Exception as e:
            print(f"vision failed, tesseract fallback: {e}")
    from results import build_result
    text, lines = ocr(path, model_id)
    return build_result(name or os.path.basename(path), project,
                        "image", model_id, text, lines)


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="OCR one image file")
    ap.add_argument("path")
    ap.add_argument("--project", default="default")
    a = ap.parse_args()
    print(json.dumps(ocr_file(a.path, a.project), indent=2))


if __name__ == "__main__":
    main()
