#!/usr/bin/env python3
"""Uniform result sidecar for queue processing.

Every processed file (audio transcript or image OCR) gets one JSON
sidecar next to the source:

    uploads/<project>/<basename>.result.json

    {
      "source": "<original file name>",
      "project": "<project>",
      "kind": "audio" | "video" | "image",
      "model": "<model id that produced text>",
      "text": "<transcript or OCR text>",
      "lines": [{"text": ..., "conf": ..., "box": [x0,y0,x1,y1]}],
      "created_at": "YYYY-MM-DD HH:MM:SS"
    }

`lines` is empty for audio. Consumers read one format regardless of
source; `kind`/`model` say how `text` was produced.
"""
import json
import os
from datetime import datetime

RESULT_EXT = ".result.json"


def result_path(source_path):
    return os.path.splitext(source_path)[0] + RESULT_EXT


def build_result(source_name, project, kind, model, text, lines=None):
    return {
        "source": source_name,
        "project": project,
        "kind": kind,
        "model": model,
        "text": text or "",
        "lines": lines or [],
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_result(source_path, result):
    """Write sidecar next to source; returns sidecar path."""
    out = result_path(source_path)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    return out


def result_url(job_url):
    """Sidecar URL derived from the job's file URL."""
    if not job_url:
        return ""
    base, _ = os.path.splitext(job_url)
    return base + RESULT_EXT
