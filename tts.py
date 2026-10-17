#!/usr/bin/env python3
"""Local NVIDIA TTS for 4GB RAM: FastPitch + HiFiGAN (NeMo).

Why this pair: single-speaker English, 46M + 85M params (~520MB fp32),
~0.4s first-sentence on CPU, RTF ~0.1 warm. No GPU needed, so Parakeet
STT keeps the 4GB GTX 1650 Ti to itself during phone calls.

  .venv-gpu/bin/python tts.py "I need eighty 2x4 studs" -o quote.wav
  .venv-gpu/bin/python tts.py --mulaw -o prompt.ulaw "Hi, calling for a quote"

API: from tts import speak, speak_mulaw
"""
import gc
import logging
import os
import re

FP_MODEL = os.environ.get("TTS_FP", "tts_en_fastpitch")
HIFI_MODEL = os.environ.get("TTS_HIFI", "tts_en_hifigan")
SAMPLE_RATE = 22050

_pipe = None  # (fastpitch, hifigan) resident on CPU


def load(fp_model=FP_MODEL, hifi_model=HIFI_MODEL):
    """Load both models onto CPU. Idempotent; caller MAY keep resident."""
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel
    logging.disable(logging.CRITICAL)
    try:
        fp = FastPitchModel.from_pretrained(fp_model, map_location="cpu").eval()
        hg = HifiGanModel.from_pretrained(hifi_model, map_location="cpu").eval()
        for p in list(fp.parameters()) + list(hg.parameters()):
            p.requires_grad_(False)
    finally:
        logging.disable(logging.NOTSET)
    _pipe = (fp, hg)
    return _pipe


def unload():
    global _pipe
    if _pipe is not None:
        del _pipe
        _pipe = None
    gc.collect()


def _split_sentences(text, limit=220):
    """Short chunks => constant RAM, lower first-audio latency."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        while len(p) > limit:  # hard-split very long sentences
            cut = p.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            out.append(p[:cut])
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out or ["Hi."]


def synth(text, unload_after=False):
    """Text -> (numpy float32 mono, sample_rate). Models stay loaded."""
    import torch
    import numpy as np
    fp, hg = load()
    chunks = []
    with torch.inference_mode():
        for sent in _split_sentences(text):
            spec = fp.generate_spectrogram(tokens=fp.parse(sent), speaker=None)
            audio = hg.convert_spectrogram_to_audio(spec=spec)
            chunks.append(audio.cpu().numpy().flatten())
    wav = np.concatenate(chunks).astype("float32")
    if unload_after:
        unload()
    return wav, SAMPLE_RATE


def speak(text, out_path, unload_after=False):
    """Write 22kHz WAV. Returns out_path."""
    import soundfile as sf
    wav, sr = synth(text, unload_after=unload_after)
    sf.write(out_path, wav, sr)
    return out_path


def speak_mulaw(text, out_path=None, unload_after=False):
    """Write raw 8kHz mu-law (Twilio Media Streams format). Returns bytes."""
    import audioop
    import numpy as np
    import librosa
    wav, sr = synth(text, unload_after=unload_after)
    wav8k = librosa.resample(wav, orig_sr=sr, target_sr=8000)
    pcm = (np.clip(wav8k, -1, 1) * 32767).astype("<i2").tobytes()
    ulaw = audioop.lin2ulaw(pcm, 2)
    if out_path:
        with open(out_path, "wb") as f:
            f.write(ulaw)
    return ulaw


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="text to speak (or @file)")
    ap.add_argument("-o", default="out.wav")
    ap.add_argument("--mulaw", action="store_true", help="8kHz mu-law output")
    ap.add_argument("--unload", action="store_true", help="free RAM after synth")
    a = ap.parse_args()
    text = a.text
    if text.startswith("@"):
        with open(text[1:]) as f:
            text = f.read()
    if a.mulaw:
        speak_mulaw(text, a.o, unload_after=a.unload)
    else:
        speak(text, a.o, unload_after=a.unload)
    print(f"wrote {a.o}")


if __name__ == "__main__":
    main()
