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

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}


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
    """OCR one image -> result dict (results.py format, kind=image)."""
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
