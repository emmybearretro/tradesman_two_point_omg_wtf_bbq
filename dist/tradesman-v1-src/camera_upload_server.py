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
import os
import sys
from datetime import datetime
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(APP_DIR, "static"))


def _safe_name(filename):
    base = secure_filename(filename) or "upload"
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".bin"
    return f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}_{stem}{ext}"


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "no files"}), 400
    out = []
    for f in files:
        if f.filename == "":
            continue
        name = _safe_name(f.filename)
        path = os.path.join(UPLOAD_DIR, name)
        f.save(path)
        out.append({
            "name": name,
            "original": f.filename,
            "url": f"/uploads/{name}",
            "size": os.path.getsize(path),
            "type": f.content_type or "application/octet-stream",
        })
    return jsonify({"uploads": out})


@app.route("/uploads/<name>")
def uploaded(name):
    return send_from_directory(UPLOAD_DIR, name, conditional=True)


@app.route("/files")
def files():
    items = []
    for name in sorted(os.listdir(UPLOAD_DIR)):
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(path):
            continue
        items.append({
            "name": name,
            "url": f"/uploads/{name}",
            "size": os.path.getsize(path),
            "modified": datetime.utcfromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return jsonify({"files": items})


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
      meta.textContent = `${f.name}  •  ${f.size} bytes`;
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

renderGallery();
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
