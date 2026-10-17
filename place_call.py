#!/usr/bin/env python3
"""Place an outbound lumber-quote call via Twilio (stdlib only, no twilio pkg).

  export TWILIO_SID=AC... TWILIO_TOKEN=... TWILIO_FROM=+15557654321
  .venv-gpu/bin/python place_call.py --to +15551234567 \
      --voice-url https://<public-host>/voice
"""
import argparse
import base64
import json
import os
import urllib.parse
import urllib.request


def place_call(to, voice_url, sid, token, from_):
    cred = base64.b64encode(f"{sid}:{token}".encode()).decode()
    data = urllib.parse.urlencode({"To": to, "From": from_, "Url": voice_url}).encode()
    req = urllib.request.Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json",
        data=data, headers={"Authorization": f"Basic {cred}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True)
    ap.add_argument("--voice-url", required=True, help="public https://<host>/voice")
    ap.add_argument("--from-", dest="from_", default=os.environ.get("TWILIO_FROM", ""))
    ap.add_argument("--sid", default=os.environ.get("TWILIO_SID", ""))
    ap.add_argument("--token", default=os.environ.get("TWILIO_TOKEN", ""))
    a = ap.parse_args()
    assert a.sid and a.token and a.from_, "set TWILIO_SID/TWILIO_TOKEN/TWILIO_FROM"
    res = place_call(a.to, a.voice_url, a.sid, a.token, a.from_)
    print("call sid:", res.get("sid"), res.get("status"))


if __name__ == "__main__":
    main()
