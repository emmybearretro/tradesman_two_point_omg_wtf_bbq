#!/usr/bin/env python3
"""Queue worker: processes pending upload jobs from the local queue.

Run this on the laptop (same machine as camera_upload_server.py):

    /home/hc/.local/share/mise/installs/python/3.15.0/bin/python3 queue_worker.py

It polls POST /queue/process once per job until the queue is drained, then
sleeps and repeats. Processing logic lives in _process_job() inside
camera_upload_server.py (default: log the job and mark it done).
"""
import json
import os
import sys
import time

import urllib.request
import urllib.error

BASE = os.environ.get("QUEUE_BASE", "http://127.0.0.1:5000")
POLL_INTERVAL = float(os.environ.get("QUEUE_POLL_INTERVAL", "3"))

def post(path, data=None):
    url = BASE + path
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}")
    except Exception as e:
        raise RuntimeError(f"request failed: {e}")

def main():
    print(f"queue worker: base={BASE} interval={POLL_INTERVAL}s", flush=True)
    while True:
        try:
            res = post("/queue/process")
            if res.get("processed"):
                print(f"processed job: {res}", flush=True)
            else:
                # no pending jobs; sleep
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"worker error: {e}", flush=True)
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
