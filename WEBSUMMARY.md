# Session Summary — local iPhone capture hub

**Session:** `local-96060080a1ca4a8122e2079ddbb552c02d7e4a52d37f42bcd8c8d752375e8778`
**When:** 2026-10-17, 13:20 → 16:32 EDT (~3h12m)
**Where:** `/home/hc/hackathon` · model `main` · 179 messages / 79 tool calls
**Topic:** Build a locally hosted web page that an iPhone can reach over ad-hoc Wi-Fi to
capture photos and voice, upload them to the laptop, then queue them for processing.

---

## Goal

A single self-hosted Flask page, opened from an iPhone on the same Wi-Fi/hotspot, with:

- live camera capture + photo-library upload
- voice recording
- files landing on the laptop in an organized, unique naming scheme
- a "we're done uploading" step that pushes the batch into a processing queue

The user repeated **"DO NOT USE SUBAGENTS"** several times early in the session.

## Deliverables (both on disk in `/home/hc/hackathon`)

| File | Size | Role |
|---|---|---|
| `camera_upload_server.py` | 22.8 KB / 631 lines | Single-file Flask app — inline HTML/CSS/JS page, `POST /upload`, `/files`, `/uploads/<project>/<file>`, `/queue/enqueue`, `/queue/status`; ffmpeg audio extraction |
| `queue_worker.py` | 1.7 KB | File-backed JSONL job queue worker (Redis was not installed; built dependency-free so it can be swapped for Redis later) |

Storage layout:

```
uploads/<project>/<UTC-timestamp>_<uuid8>_<original-name>
uploads/processed/
uploads/.queue/          # JSONL job log + worker.log
```

Server binds `0.0.0.0:5000` and prints the LAN URL for the phone.

## What was built, in order

1. **Base page** — photo capture, voice capture, upload to the local harness, gallery of
   previously uploaded files. Served from the laptop, reachable by IP from the iPhone.
2. **Interaction fixes** — the first cut did nothing on tap. Fixed by making the launch
   controls real `label`/`for` targets, moving work onto explicit handlers, and wrapping
   every handler in try/catch so failures surface as visible status text instead of silence.
3. **Upload reliability** — "says uploading, logs show nothing, then fails" was traced
   through the JS → fetch → Flask path until uploads landed and were logged.
4. **Full-image display** — gallery was center-cropping photos to squares
   (`aspect-ratio: 1` + `object-fit: cover`). Changed to `object-fit: contain` and made
   thumbnails click through to the full-size image.
5. **Project names + UUIDs** — a project field on the page; files stored under
   `uploads/<project>/` with a UTC timestamp and short UUID prefix; `/files` and the
   static route made project-aware; gallery shows the project in the metadata line.
6. **Processing queue** — a "done uploading" action enqueues every file in the project;
   endpoints `POST /queue/enqueue` and `GET /queue/status`; `queue_worker.py` moves jobs
   `pending → processing → done` and appends to a worker log. Verified end-to-end
   (enqueue → pending → process → done), then test artifacts removed.
7. **Audio extraction** — iOS records "voice" as video. On upload, ffmpeg now extracts the
   audio track to a sibling `.wav`, the JSON response carries `audio_url`, and the gallery
   renders it as `<audio controls>`. Verified: 257 KB WAV returned as `audio/x-wav`,
   listed by `/files`, playable in the page.

## Friction worth remembering

- **Stale server processes** were the main time sink. Twice the source contained a fix but
  the running process was old code (`audio_url` missing despite correct file). A clean
  kill-all + restart resolved it; always confirm the PID/start time, not just the file.
- Repeated tool-call mistakes: `notify` passed to foreground `terminal` calls, and `&`
  backgrounding rejected in favour of `background=true`. Each self-corrected but cost turns.
- A few user turns received *"Your request was not processed"* — empty assistant replies —
  including the final request. One compaction event mid-session (PRIOR CONTEXT note at
  message 5540; some patch outputs demoted).

## Unfinished

The last user request — **"an option to upload via photos off the phone"**, i.e. a clear
photo-library picker separate from live camera capture — was never implemented. The session
ended on a dropped turn immediately after it.

## State at handoff

- `camera_upload_server.py` and `queue_worker.py` exist and are tracked (last touched by
  commit `8f317d4`).
- Working tree: 2 deleted test images under `uploads/`, 2 untracked `states/*.pt` files —
  unrelated to this work.
- The server was last verified running on port 5000 with audio extraction active; it was
  not left running at session end.
