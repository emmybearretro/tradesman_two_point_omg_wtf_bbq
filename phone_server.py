#!/usr/bin/env python3
"""Phone loop for lumber quotes: Twilio Media Streams <-> Parakeet STT <-> FastPitch TTS.

Layout (fits 4GB VRAM): Parakeet fp16 resident on GPU (~1.5GB), TTS resident
on CPU (~0.5GB RAM, 0 VRAM). Unload the vision ckpt before a call.

  # 1. expose publicly, e.g. ngrok http 8080, then:
  .venv-gpu/bin/python phone_server.py --port 8080 --bom "80x 2x4x8 studs, 30x plates"
  # 2. from another shell:
  .venv-gpu/bin/python place_call.py --to +15551234567 --from $TWILIO_FROM \
      --voice-url https://<ngrok>/voice
  # 3. Twilio <Connect><Stream> hits this box; dialog + transcript saved to
  #    quotes/<call_sid>.json

No Twilio account? --dry-run exercises the dialog + TTS offline:
  .venv-gpu/bin/python phone_server.py --dry-run --bom "80x 2x4x8 studs"
"""
import argparse
import asyncio
import audioop
import base64
import gc
import json
import logging
import os
import re
import struct
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PUBLIC_WS_URL = os.environ.get("PUBLIC_WS_URL", "")  # wss://<host>/media
MONEY_RE = re.compile(r"\$\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)")
PRICE_CTX_RE = re.compile(r"(price|cost|total|quote|charge|each|dollar|bucks)[^.]{0,30}?(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)", re.I)
LEAD_RE = re.compile(r"(today|tomorrow|monday|tuesday|wednesday|thursday|friday|(\d+)\s*(day|week))", re.I)


class QuoteDialog:
    """Pure dialog logic: no audio, no network. Unit-testable."""

    def __init__(self, bom):
        self.bom = bom
        self.turn = 0
        self.prices = []      # raw money mentions from the yard
        self.lead_time = None
        self.transcript = []  # [(who, text)]

    def greeting(self):
        msg = (f"Hi, I'm calling from a job site about a lumber quote. "
               f"I need {self.bom}. Could you give me your best price?")
        self.transcript.append(("us", msg))
        return msg

    def reply(self, yard_text):
        """Yard said yard_text -> (our reply, done?)."""
        self.transcript.append(("yard", yard_text))
        self.turn += 1
        for m in MONEY_RE.findall(yard_text or ""):
            try:
                self.prices.append(float(m.replace(",", "")))
            except ValueError:
                pass
        for m in PRICE_CTX_RE.findall(yard_text or ""):
            try:
                self.prices.append(float(m[1].replace(",", "")))
            except ValueError:
                pass
        lm = LEAD_RE.search(yard_text or "")
        if lm and not self.lead_time:
            self.lead_time = lm.group(0)
        if self.turn == 1:
            msg = "Thanks. Is that in stock now, and when could we pick up or get delivery?"
        elif self.turn == 2:
            msg = ("Got it. So I have the total noted — is that including tax, "
                   "and what's the best number to confirm the order?")
        else:
            best = f" around ${min(self.prices):.2f}" if self.prices else ""
            msg = (f"Perfect, I'll take that back to the crew{best}. "
                   f"Thank you, goodbye.")
            self.transcript.append(("us", msg))
            return msg, True
        self.transcript.append(("us", msg))
        return msg, False

    def quote(self):
        return {"bom": self.bom, "prices_heard": self.prices,
                "lead_time": self.lead_time, "transcript": self.transcript}


def ulaw8k_to_wav16k(frames):
    """Concatenated mu-law 8k bytes -> temp 16k wav path for Parakeet."""
    pcm8k = audioop.ulaw2lin(b"".join(frames), 2)
    pcm8k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.close()
    import wave
    with wave.open(f.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm8k)
    return f.name


class CallSession:
    """One live call: GPU STT handle + CPU TTS + dialog + VAD."""

    SILENCE_MS = 350  # end-of-turn threshold (snappy but not clipped)
    MIN_TURN_MS = 600

    def __init__(self, bom, stt_model, use_cuda):
        self.dialog = QuoteDialog(bom)
        self.stt = stt_model
        self.use_cuda = use_cuda
        self.frames = []       # pending mu-law 8k chunks
        self.quiet_ms = 0
        self.speaking = False

    def vad_frame(self, ulaw):
        """Feed one ~20ms mu-law chunk. Returns True when a turn is ready."""
        pcm, _ = audioop.ulaw2lin(ulaw, 2), None
        n = len(pcm) // 2
        peak = audioop.max(pcm, 2) if n else 0
        self.frames.append(ulaw)
        if peak > 1200:  # voice energy at 8k mu-law
            self.quiet_ms = 0
            self.speaking = True
        elif self.speaking:
            self.quiet_ms += 20
        total_ms = len(b"".join(self.frames)) // 160 * 20 // 20 * 20
        if self.speaking and self.quiet_ms >= self.SILENCE_MS and total_ms >= self.MIN_TURN_MS:
            return True
        if len(b"".join(self.frames)) > 160 * 400:  # 8s cap: force-turn
            return True
        return False

    def transcribe_turn(self):
        import torch
        raw = b"".join(self.frames)
        self.frames, self.quiet_ms, self.speaking = [], 0, False
        if len(raw) < 160 * 30:  # <0.6s: noise, skip STT
            return ""
        path = ulaw8k_to_wav16k([raw])
        try:
            with torch.inference_mode():
                out = self.stt.transcribe([path], verbose=False)
            if isinstance(out, tuple):
                out = out[0]
            text = out[0]
            if hasattr(text, "text"):
                text = text.text
            return (text or "").strip()
        finally:
            os.unlink(path)


def load_stt():
    """Parakeet fp16 on GPU if present (same recipe as transcribe.py)."""
    import torch
    from nemo.collections.asr.models import ASRModel
    use_cuda = torch.cuda.is_available()
    logging.disable(logging.CRITICAL)
    try:
        m = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2",
                                     map_location=torch.device("cpu"))
        m = m.half().eval()
        if use_cuda:
            try:
                m = m.cuda().eval()
            except RuntimeError:
                torch.cuda.empty_cache()
                use_cuda = False
    finally:
        logging.disable(logging.NOTSET)
    return m, use_cuda


def tts_to_ulaw(text):
    from tts import speak_mulaw
    return speak_mulaw(text)  # resident CPU models, ~0.2-0.4s first audio


# ---- Twilio wiring: TwiML webhook (stdlib) + media websocket (websockets pkg) ----

class VoiceHandler(BaseHTTPRequestHandler):
    ws_url = ""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        twiml = (f'<?xml version="1.0" encoding="UTF-8"?><Response>'
                 f'<Connect><Stream url="{self.ws_url}"/></Connect></Response>')
        body = twiml.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


async def handle_media(ws, bom, save_dir):
    """Twilio Media Streams handler: JSON text frames with base64 mu-law."""
    import torch
    stt, use_cuda = load_stt()
    import tts as tts_mod
    tts_mod.load()
    call_sid, frames, turn_i = "local", [], 0
    session = None
    try:
        # greet immediately so the yard hears us without a gap
        dialog = QuoteDialog(bom)
        await ws.send(json.dumps({"event": "media",
                                  "media": {"payload": base64.b64encode(
                                      tts_to_ulaw(dialog.greeting())).decode()}}))
        async for raw in ws:
            msg = json.loads(raw)
            ev = msg.get("event")
            if ev == "start":
                call_sid = msg.get("start", {}).get("callSid", "local")
                session = CallSession(bom, stt, use_cuda)
                session.dialog = dialog
            elif ev == "media" and session is not None:
                ulaw = base64.b64decode(msg["media"]["payload"])
                if session.vad_frame(ulaw):
                    yard = session.transcribe_turn()
                    if not yard:
                        continue
                    reply, done = session.dialog.reply(yard)
                    await ws.send(json.dumps({"event": "media", "media": {
                        "payload": base64.b64encode(tts_to_ulaw(reply)).decode()}}))
                    if done:
                        break
            elif ev == "stop":
                break
        q = (session.dialog if session else dialog).quote()
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{call_sid}.json"), "w") as f:
            json.dump({"call": call_sid, "quote": q, "ts": time.time()}, f, indent=1)
        print(f"saved quote {call_sid}: prices={q['prices_heard']} lead={q['lead_time']}")
    finally:
        try:
            del stt
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
        except Exception:
            pass


def serve(args):
    ws_url = PUBLIC_WS_URL or f"wss://<public-host>/media"
    VoiceHandler.ws_url = ws_url if args.voice_url is None else args.voice_url.replace("https://", "wss://").replace("http://", "ws://") + "/media"
    http = HTTPServer(("0.0.0.0", args.port), VoiceHandler)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    print(f"TwiML webhook : http://0.0.0.0:{args.port}/voice  -> {VoiceHandler.ws_url}")
    print("point Twilio voice URL at the public /voice address; audio streams on /media")
    try:
        import websockets
    except ImportError:
        sys.exit("pip install websockets  (.venv-gpu/bin/pip install websockets twilio)")
    from websockets.asyncio.server import serve as ws_serve

    async def main():
        async with ws_serve(lambda ws: handle_media(ws, args.bom, args.save_dir),
                            "0.0.0.0", args.port + 1):
            print(f"media websocket: ws://0.0.0.0:{args.port + 1}/media")
            await asyncio.get_running_loop().create_future()
    asyncio.run(main())


def dry_run(args):
    """No Twilio: simulate yard replies, synth every line, print quote JSON."""
    import tts as tts_mod
    d = QuoteDialog(args.bom)
    print("US:", d.greeting())
    tts_mod.speak(d.greeting(), "/tmp/quote_0.wav")
    simulated = ["That comes to $612.40 total, in stock.",
                 "Pickup tomorrow morning works, plus tax.",
                 "555-0142, ask for the yard desk."]
    for i, yard in enumerate(simulated, 1):
        reply, done = d.reply(yard)
        print(f"YARD: {yard}\nUS: {reply}")
        tts_mod.speak(reply, f"/tmp/quote_{i}.wav")
        if done:
            break
    print(json.dumps(d.quote(), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bom", default="80x 2x4x8 studs, 30x plates")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--voice-url", default=None, help="public wss base, e.g. wss://abc.ngrok.io")
    ap.add_argument("--save-dir", default="quotes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if struct.calcsize("P") == 8:
        pass
    (dry_run if args.dry_run else serve)(args)


if __name__ == "__main__":
    main()
