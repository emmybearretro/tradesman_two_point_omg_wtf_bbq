#!/usr/bin/env python3
"""Local camera + voice upload hub.

Host this on your laptop, connect your iPhone to the same Wi-Fi (or an
ad-hoc hotspot the laptop creates), and open http://<laptop-ip>:<port>.

The page uses <input type="file" capture="camera/microphone>, which on iOS
launches the native Camera / Voice Memos apps. That works over plain HTTP
(no HTTPS / secure-context required), unlike getUserMedia / MediaRecorder.

Uploads land in ./uploads/ (the local harness). The server also lists and
serves back anything already there.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
QUEUE_DIR = os.path.join(UPLOAD_DIR, ".queue")
QUEUE_FILE = os.path.join(QUEUE_DIR, "jobs.jsonl")
PROCESSED_DIR = os.path.join(UPLOAD_DIR, "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def _queue_read():
    jobs = []
    if not os.path.exists(QUEUE_FILE):
        return jobs
    with open(QUEUE_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except Exception:
                continue
    return jobs


def _queue_write(jobs):
    tmp = QUEUE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    os.replace(tmp, QUEUE_FILE)


def _job_id(project, name):
    return hashlib.md5(f"{project}/{name}".encode()).hexdigest()


def _already_queued(jobs, project, name):
    return any(j.get("project") == project and j.get("name") == name for j in jobs)


def _enqueue(project):
    """Enqueue every file in uploads/<project> that isn't already queued."""
    jobs = _queue_read()
    project_dir = os.path.join(UPLOAD_DIR, project)
    added = 0
    if os.path.isdir(project_dir):
        for root, dirs, files in os.walk(project_dir):
            for fn in sorted(files):
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, UPLOAD_DIR)
                proj = rel.split(os.sep)[0] if os.sep in rel else ""
                if proj != project:
                    continue
                if _already_queued(jobs, project, fn):
                    continue
                job = {
                    "job_id": _job_id(project, fn),
                    "project": project,
                    "name": fn,
                    "path": path,
                    "url": f"/uploads/{rel}",
                    "status": "pending",
                    "queued_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                }
                jobs.append(job)
                added += 1
    _queue_write(jobs)
    return added


def _process_job(job):
    """Process one job: STT for audio/video, OCR for images (results.py format).

    Audio/video (incl. .mov/.mp4 from iPhone) go through transcribe.py
    (nvidia/parakeet-tdt-0.6b-v2, GPU-venv subprocess, VRAM freed on exit).
    Images go through ocr.py (tesseract placeholder until the vision
    model is trained). Both write one `.result.json` sidecar next to the
    source and attach it to the job dict. Derived .wav files from
    _extract_audio are skipped (the source video job carries the result).
    """
    from results import build_result, result_url, save_result
    log_path = os.path.join(QUEUE_DIR, "worker.log")
    path = job.get("path", "")
    name = job.get("name", "")
    project = job.get("project", "")
    text = ""
    try:
        from transcribe import (
            MODEL_ID as STT_MODEL, is_transcribable, transcribe_file)
        transcribable = is_transcribable(name or path)
    except ImportError:
        transcribable = False
        transcribe_file = None
        STT_MODEL = ""
    try:
        from ocr import MODEL_ID as OCR_MODEL, is_ocrable, ocr_file
        ocrable = is_ocrable(name or path)
    except ImportError:
        ocrable = False
        ocr_file = None
        OCR_MODEL = ""
    if transcribable and path and os.path.exists(path):
        base, ext = os.path.splitext(path)
        if ext.lower() == ".wav" and any(
            os.path.exists(base + v) or os.path.exists(base + v.upper())
            for v in VIDEO_EXTS
        ):
            line = f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} " \
                f"[{job['job_id']}] {job['project']}/{job['name']} skipped (derived wav)\n"
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
            job["skipped"] = "derived wav"
            return True
        text = transcribe_file(path)
        kind = "video" if ext.lower() in VIDEO_EXTS else "audio"
        result = build_result(name, project, kind, STT_MODEL, text)
    elif ocrable and path and os.path.exists(path):
        result = ocr_file(path, project, name)
        text = result["text"]
    else:
        result = None
    if result is not None:
        job["result"] = result
        job["text"] = text
        job["result_path"] = save_result(path, result)
        job["result_url"] = result_url(job.get("url", ""))
    line = f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} " \
        f"[{job['job_id']}] {job['project']}/{job['name']} -> {job['url']}"
    if text:
        line += f" text_chars={len(text)}"
    line += "\n"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line)
    # Uncomment to relocate processed files:
    # dest = os.path.join(PROCESSED_DIR, job["project"], job["name"])
    # os.makedirs(os.path.dirname(dest), exist_ok=True)
    # os.replace(job["path"], dest)
    return True


VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv", ".webm", ".3gp", ".m4v")


def _extract_audio(input_path):
    """Extract the audio track from a video file to a WAV file alongside.

    Returns the output path, or None if extraction failed (e.g. no audio
    stream). The original file is left untouched.
    """
    base, _ext = os.path.splitext(input_path)
    out_path = base + ".wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                out_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception:
        pass
    return None

app = Flask(__name__, static_folder=os.path.join(APP_DIR, "static"))


def _safe_name(filename):
    base = secure_filename(filename) or "upload"
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".bin"
    return f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}_{stem}{ext}"


def _safe_project(project):
    project = (project or "default").strip()
    project = secure_filename(project) or "default"
    if project in (".", "..") or "/" in project or "\\" in project:
        project = "default"
    return project


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    project = _safe_project(request.form.get("project"))
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "no files"}), 400
    subdir = os.path.join(UPLOAD_DIR, project)
    os.makedirs(subdir, exist_ok=True)
    out = []
    for f in files:
        if f.filename == "":
            continue
        name = _safe_name(f.filename)
        path = os.path.join(subdir, name)
        f.save(path)
        audio_url = None
        is_video = (f.content_type and f.content_type.startswith("video/")) or \
                   name.lower().endswith(VIDEO_EXTS)
        if is_video:
            audio_path = _extract_audio(path)
            if audio_path:
                audio_name = os.path.basename(audio_path)
                audio_url = f"/uploads/{project}/{audio_name}"
        out.append({
            "name": name,
            "project": project,
            "original": f.filename,
            "url": f"/uploads/{project}/{name}",
            "audio_url": audio_url,
            "size": os.path.getsize(path),
            "type": f.content_type or "application/octet-stream",
        })
    return jsonify({"uploads": out})


@app.route("/uploads/<path:name>")
def uploaded(name):
    return send_from_directory(UPLOAD_DIR, name, conditional=True)


@app.route("/files")
def files():
    items = []
    for root, dirs, filenames in os.walk(UPLOAD_DIR):
        dirs.sort()
        for fn in sorted(filenames):
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, UPLOAD_DIR)
            project = rel.split(os.sep)[0] if os.sep in rel else ""
            items.append({
                "name": fn,
                "project": project,
                "url": f"/uploads/{rel}",
                "size": os.path.getsize(path),
                "modified": datetime.utcfromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S"),
            })
    return jsonify({"files": items})


@app.route("/queue", methods=["POST"])
def queue():
    project = _safe_project(request.form.get("project"))
    added = _enqueue(project)
    return jsonify({"project": project, "added": added})


@app.route("/queue/status")
def queue_status():
    jobs = _queue_read()
    stats = {"pending": 0, "processing": 0, "done": 0, "total": len(jobs)}
    for j in jobs:
        s = j.get("status", "pending")
        stats[s] = stats.get(s, 0) + 1
    return jsonify({"stats": stats, "jobs": jobs})


@app.route("/queue/process", methods=["POST"])
def queue_process():
    """Process one pending job (used by the worker)."""
    jobs = _queue_read()
    progressed = False
    for j in jobs:
        if j.get("status") == "pending":
            j["status"] = "processing"
            _queue_write(jobs)
            try:
                _process_job(j)
                j["status"] = "done"
            except Exception as e:
                j["status"] = "error"
                j["error"] = str(e)
            _queue_write(jobs)
            progressed = True
            break
    return jsonify({"processed": progressed})


def local_ips():
    import socket
    ips = set()
    try:
        hostname = socket.gethostname()
        for addr in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC,
                                        socket.SOCK_STREAM, 0):
            ips.add(addr[4][0])
    except Exception:
        pass
    # also the default route interface
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Local Uploads</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 1rem; background: #f5f7fa; color: #1a1a1a; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
  h1 { font-size: 1.25rem; margin: 0; }
  .url { font-family: monospace; font-size: 0.85rem; color: #333; background: #eef2f7; padding: 0.4rem 0.6rem; border-radius: 6px; }
  .card { background: #fff; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .row { display: flex; gap: 0.75rem; margin-bottom: 0.75rem; flex-wrap: wrap; }
  .btn { flex: 1; min-width: 120px; padding: 0.75rem 1rem; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.95rem; transition: transform 0.05s; }
  .btn.primary { background: #0066cc; color: #fff; }
  .btn.primary:active { transform: translateY(1px); background: #0052a8; }
  .btn.secondary { background: #e2e8f0; color: #1a1a1a; }
  .btn.secondary:active { background: #c9d4e0; }
  .btn.disabled { opacity: 0.6; cursor: not-allowed; }
  .input-wrap { position: relative; display: inline-block; flex: 1; min-width: 140px; }
  input[type="file"] { position: absolute; overflow: hidden; width: 0.1px; height: 0.1px; opacity: 0; z-index: -1; }
  .choose { display: inline-block; padding: 0.75rem 1rem; border: 2px dashed #9aa5b1; border-radius: 8px; text-align: center; color: #5a6b7a; cursor: pointer; font-size: 0.95rem; }
  .choose:hover { border-color: #0066cc; color: #0066cc; background: #f0f6ff; }
  .choose:active { background: #e6f0ff; }
  .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.75rem; }
  .thumb { border-radius: 8px; overflow: hidden; background: #eef2f7; min-height: 180px; display: flex; align-items: center; justify-content: center; color: #6b7a8a; font-size: 0.8rem; cursor: pointer; }
  .thumb img { width: 100%; height: auto; max-height: 100%; object-fit: contain; display: block; }
  .thumb video { width: 100%; height: auto; max-height: 100%; object-fit: contain; display: block; }
  .audio { width: 100%; border-radius: 8px; background: #eef2f7; padding: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
  .meta { font-size: 0.75rem; color: #6b7a8a; margin-top: 0.25rem; }
  .status { margin-top: 0.5rem; font-size: 0.9rem; color: #222; }
  .ok { color: #166534; }
  .err { color: #b91c1c; }
  .hint { font-size: 0.85rem; color: #4b5563; margin-top: 0.5rem; line-height: 1.5; }
  .empty { text-align: center; color: #6b7a8a; padding: 1rem; font-size: 0.9rem; }
  @media (max-width: 480px) { body { padding: 0.5rem; } .btn { font-size: 0.9rem; padding: 0.6rem 0.8rem; } }
</style>
</head>
<body>
<header>
  <h1>&#128444; Local Uploads</h1>
  <div class="url">http://<span id="ip">0.0.0.0</span>:<span id="port">5000</span></div>
</header>

<div class="card">
  <div class="row">
    <label style="margin-right:0.4rem;align-self:center;">Project</label>
    <input type="text" id="projectInput" placeholder="default" value="default" style="flex:1;min-width:140px;padding:0.5rem 0.75rem;border:1px solid #ccc;border-radius:6px;font-size:0.9rem;">
  </div>
  <div class="row">
  <div class="input-wrap">
    <label class="choose" id="photoChoose" for="photoInput">Take Photo</label>
    <input type="file" id="photoInput" accept="image/*" capture="camera">
  </div>
  <div class="input-wrap">
    <label class="choose" id="voiceChoose" for="voiceInput">Record Voice</label>
    <input type="file" id="voiceInput" accept="audio/*" capture="microphone">
  </div>
  <button class="btn primary" id="uploadBtn">Upload Selected</button>
  </div>
  <div class="status" id="status">Ready. Pick a file (camera opens on iPhone) then tap Upload Selected.</div>
  <div class="row" style="margin-top:0.6rem;">
    <button class="btn secondary" id="queueBtn">Queue for Processing</button>
    <span id="queueStatus" style="margin-left:0.6rem;font-size:0.85rem;color:#4b5563;">Queue empty.</span>
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0">Gallery</h2>
  <div id="gallery" class="gallery"></div>
</div>

<div class="hint">On iPhone: join the laptop's Wi-Fi/ad-hoc network, then open this page. Camera &amp; voice launch the native apps. Files are stored locally on the laptop under <code>uploads/</code>.</hint>

<script>
const ipEl = document.getElementById("ip");
const portEl = document.getElementById("port");
const status = document.getElementById("status");
const gallery = document.getElementById("gallery");
const uploadBtn = document.getElementById("uploadBtn");
const photoInput = document.getElementById("photoInput");
const voiceInput = document.getElementById("voiceInput");
// label 'for' attribute alone activates the input on iOS; no programmatic click() needed

// show the laptop's real ip (server echoes it; fallback to 0.0.0.0)
fetch("/files").then(r => r.json()).then(d => {
  // nothing to do; just a health check
}).catch(() => {});

let selected = []; // {file, preview}

function readAsDataURL(file, cb) {
  const r = new FileReader();
  r.onload = () => cb(r.result);
  r.onerror = () => cb(null);
  r.readAsDataURL(file);
}

function addPreview(file) {
  return new Promise((resolve) => {
    readAsDataURL(file, (url) => {
      selected.push({ file, preview: url });
      resolve();
    });
  });
}

function updateStatus(msg, ok) {
  status.textContent = msg;
  status.className = "status " + (ok ? "ok" : "err");
}

function renderGallery() {
  gallery.innerHTML = "";
  fetch("/files").then(r => r.json()).then(d => {
    const files = d.files || [];
    if (files.length === 0) {
      gallery.innerHTML = '<div class="empty">No uploads yet.</div>';
      return;
    }
    for (const f of files) {
      const el = document.createElement("div");
      el.className = "thumb";
      const isImage = /\.(jpg|jpeg|png|gif|webp)$/i.test(f.name);
      const isAudio = /\.(mp3|m4a|wav|oga|caf|aac)$/i.test(f.name);
      if (isImage) {
        el.innerHTML = `<a href="/uploads/${encodeURIComponent(f.name)}" target="_blank" rel="noopener"><img src="/uploads/${encodeURIComponent(f.name)}" alt="${f.name}"></a>`;
      } else if (isAudio) {
        el.innerHTML = `<audio controls src="/uploads/${encodeURIComponent(f.name)}"></audio>`;
      } else {
        el.innerHTML = `<div>&#128449; ${f.name}</div>`;
      }
      const meta = document.createElement("div");
      meta.className = "meta";
      const proj = f.project ? ` [${f.project}]` : "";
      meta.textContent = `${f.name}${proj}  •  ${f.size} bytes`;
      el.appendChild(meta);
      gallery.appendChild(el);
    }
  }).catch(() => {
    gallery.innerHTML = '<div class="empty">Could not load gallery.</div>';
  });
}

async function uploadFiles(files) {
  if (!files || files.length === 0) {
    updateStatus("Nothing to upload.", false);
    return false;
  }
  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";
  updateStatus("Uploading...", true);
  const fd = new FormData();
  const project = document.getElementById("projectInput").value.trim() || "default";
  fd.append("project", project);
  for (let i = 0; i < files.length; i++) fd.append("file", files[i]);
  try {
    const r = await fetch("/upload", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "upload failed");
    updateStatus(`Uploaded ${d.uploads.length} file(s).`, true);
    selected = [];
    uploadBtn.disabled = true;
    uploadBtn.textContent = "Upload Selected";
    photoInput.value = "";
    voiceInput.value = "";
    renderGallery();
    return true;
  } catch (err) {
    updateStatus("Upload failed: " + err, false);
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload Selected";
    return false;
  }
}

async function queueForProcessing() {
  const project = document.getElementById("projectInput").value.trim() || "default";
  const btn = document.getElementById("queueBtn");
  btn.disabled = true;
  btn.textContent = "Queuing...";
  try {
    const fd = new FormData();
    fd.append("project", project);
    const r = await fetch("/queue", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "queue failed");
    updateStatus(`Queued ${d.added} file(s) for processing (project: ${project}).`, true);
    await refreshQueueStatus();
  } catch (err) {
    updateStatus("Queue failed: " + err, false);
  } finally {
    btn.disabled = false;
    btn.textContent = "Queue for Processing";
  }
}

async function refreshQueueStatus() {
  try {
    const r = await fetch("/queue/status");
    const d = await r.json();
    const s = d.stats || {};
    const total = s.total || 0;
    const pending = s.pending || 0;
    const done = s.done || 0;
    const el = document.getElementById("queueStatus");
    if (total === 0) {
      el.textContent = "Queue empty.";
    } else {
      el.textContent = `Queued: ${total} (pending ${pending}, done ${done})`;
    }
  } catch (err) {
    document.getElementById("queueStatus").textContent = "Queue status unavailable.";
  }
}

photoInput.addEventListener("change", async (e) => {
  try {
    selected = [];
    for (const f of e.target.files) {
      try { await addPreview(f); } catch {} // preview is optional; never blocks upload
    }
    updateStatus(`Selected ${selected.length} file(s). Uploading...`, true);
    await uploadFiles(e.target.files);
  } catch (err) {
    updateStatus("Error: " + err, false);
  }
});

voiceInput.addEventListener("change", async (e) => {
  try {
    selected = [];
    for (const f of e.target.files) {
      try { await addPreview(f); } catch {} // preview is optional; never blocks upload
    }
    updateStatus(`Selected ${selected.length} voice file(s). Uploading...`, true);
    await uploadFiles(e.target.files);
  } catch (err) {
    updateStatus("Error: " + err, false);
  }
});

uploadBtn.addEventListener("click", () => {
  // read current files directly from the inputs (robust even if change event
  // didn't fire / didn't populate `selected`)
  const files = [];
  for (const f of photoInput.files) files.push(f);
  for (const f of voiceInput.files) files.push(f);
  if (files.length === 0) {
    updateStatus("Nothing selected yet. Tap Take Photo / Record Voice first.", false);
    return;
  }
  uploadFiles(files);
});

queueBtn.addEventListener("click", queueForProcessing);

renderGallery();
refreshQueueStatus();
</script>
</body>
</html>
"""

@app.route("/static/<path>")
def static_file(path):
    return app.send_static_file(path)


def main():
    ap = argparse.ArgumentParser(description="Local camera + voice upload hub")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()

    # write the inline html to static/index.html
    static_dir = os.path.join(APP_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    with open(os.path.join(static_dir, "index.html"), "w") as f:
        f.write(HTML)

    ips = local_ips()
    print(f"Server running on http://{a.host}:{a.port}")
    if ips:
        print("Access from iPhone over Wi-Fi at:")
        for i in ips:
            print(f"  http://{i}:{a.port}")
    print("Uploads stored in:", UPLOAD_DIR)
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
