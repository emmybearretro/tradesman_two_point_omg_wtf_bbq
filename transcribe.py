#!/usr/bin/env python3
"""Transcribe voice/video uploads with NVIDIA Parakeet-TDT 0.6B v2.

Why this model: 600M params, ~1.2GB in fp16, peak inference ~2GB,
fits the 4GB GTX 1650 Ti with headroom. English SOTA for its size,
TDT decoder is much faster than RNNT. Load -> transcribe ONE file ->
unload (same swap pattern as infer.py) so the GPU stays free for the
vision model.

Usage:
    from transcribe import transcribe, is_transcribable
    text = transcribe("/path/to/voice.m4a")  # model in, out per call
"""
import gc
import os
import subprocess
import tempfile

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"

# Voice memos from iPhone + videos with an audio track.
AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".oga", ".ogg",
    ".caf", ".flac", ".opus", ".wma", ".aiff", ".aif",
}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".3gp", ".mkv", ".webm"}
TRANSCRIBABLE_EXTS = AUDIO_EXTS | VIDEO_EXTS


def is_transcribable(name):
    return os.path.splitext(name or "")[1].lower() in TRANSCRIBABLE_EXTS


def _to_wav_16k(src, dst):
    """ffmpeg: any audio/video -> 16kHz mono wav (what Parakeet wants)."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src,
           "-ac", "1", "-ar", "16000", "-vn", dst]
    subprocess.run(cmd, check=True)


def load(model_id=MODEL_ID):
    """Load Parakeet onto CUDA if available. Caller MUST unload().

    4GB fit: checkpoint lands on CPU, cast to fp16 there (~1.2GB),
    then moved to CUDA. Peak GPU ~1.5GB, leaving headroom next to a
    swapped-out vision model. Falls back to CPU on CUDA OOM.
    """
    import torch
    from nemo.collections.asr.models import ASRModel
    use_cuda = torch.cuda.is_available()
    model = ASRModel.from_pretrained(
        model_name=model_id, map_location=torch.device("cpu"))
    model = model.half().eval()
    if use_cuda:
        try:
            model = model.cuda().eval()
        except RuntimeError:
            torch.cuda.empty_cache()
            use_cuda = False  # CPU fallback; slow but fits anything
    return model, use_cuda


def unload(model, use_cuda):
    del model
    gc.collect()
    if use_cuda:
        import torch
        torch.cuda.empty_cache()  # model fully out; VRAM free for next swap


def transcribe(path, model_id=MODEL_ID):
    """Transcribe one file -> text. Model in, transcribed, model out."""
    import logging
    import torch
    logging.disable(logging.CRITICAL)  # keep NeMo chatter out of stdout
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory(prefix="stt_") as tmp:
            wav = os.path.join(tmp, "audio.wav")
            _to_wav_16k(path, wav)
            model, use_cuda = load(model_id)
            try:
                with torch.inference_mode():
                    texts = model.transcribe([wav], verbose=False)
                if isinstance(texts, tuple):
                    texts = texts[0]
                out = texts[0]
                if hasattr(out, "text"):  # Hypothesis object
                    out = out.text
                return (out or "").strip()
            finally:
                unload(model, use_cuda)
    finally:
        logging.disable(logging.NOTSET)


def _venv_python():
    """GPU venv python (has nemo+cu128 torch); falls back to current."""
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(here, ".venv-gpu", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    import sys
    return sys.executable


def transcribe_file(path, model_id=MODEL_ID, python=None):
    """Transcribe via subprocess under the GPU venv; returns text.

    The server runs on whatever python has Flask (CPU torch, no nemo);
    the STT stack lives in .venv-gpu. Subprocess keeps the interpreters
    separate and frees VRAM when the child exits.
    """
    import subprocess as _sp
    cmd = [_venv_python() if python is None else python,
           os.path.abspath(__file__), path]
    if model_id != MODEL_ID:
        cmd += ["--model", model_id]
    r = _sp.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"transcribe failed: {(r.stderr or '').strip()[-500:]}")
    out = r.stdout
    if "<<<TRANSCRIPT>>>" in out:  # strip any stray loader chatter
        out = out.split("<<<TRANSCRIPT>>>", 1)[1]
        out = out.split("<<<END>>>", 1)[0]
    return out.strip()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Transcribe one audio/video file")
    ap.add_argument("path")
    ap.add_argument("--model", default=MODEL_ID)
    a = ap.parse_args()
    print("<<<TRANSCRIPT>>>")
    print(transcribe(a.path, a.model))
    print("<<<END>>>")




if __name__ == "__main__":
    main()
