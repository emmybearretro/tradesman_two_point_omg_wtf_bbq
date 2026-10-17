#!/usr/bin/env python3
"""PDF text extraction for the queue (same shape as ocr.py/transcribe.py).

Native text pages come out via PyMuPDF directly; pages with no text
layer (scans/photos) fall back to rendering at 300 DPI + tesseract OCR.
CPU-only, ~50MB RAM per page.

Usage:
    from pdf import pdf_file
    result_dict = pdf_file("/path/to/doc.pdf", project="default")
"""
import os
import tempfile

MODEL_ID = "pymupdf 1.28 + tesseract fallback"

PDF_EXTS = {".pdf"}

# Pages with fewer chars than this are treated as scanned -> OCR.
TEXT_THRESHOLD = 50


def is_pdf(name):
    return os.path.splitext(name or "")[1].lower() in PDF_EXTS


def extract(path):
    """Extract one PDF -> (text, lines). Lines carry page numbers."""
    import pymupdf
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    doc = pymupdf.open(path)
    try:
        texts, lines = [], []
        for i, page in enumerate(doc, start=1):
            t = (page.get_text() or "").strip()
            if len(t) >= TEXT_THRESHOLD:
                texts.append(t)
                lines.append({"text": t, "conf": 100.0,
                              "box": [], "page": i, "via": "text"})
            else:
                t, ls = _ocr_page(page)
                texts.append(t)
                for l in ls:
                    l["page"] = i
                lines.extend(ls)
                if not t:
                    lines.append({"text": "", "conf": 0.0, "box": [],
                                  "page": i, "via": "empty"})
        return "\n\n".join(t for t in texts if t).strip(), lines
    finally:
        doc.close()


def _ocr_page(page, dpi=300):
    pix = page.get_pixmap(dpi=dpi, colorspace="gray")
    with tempfile.NamedTemporaryFile(suffix=".png", prefix="pdfocr_") as tmp:
        pix.save(tmp.name)
        from ocr import MODEL_ID as OCR_MODEL, ocr
        text, lines = ocr(tmp.name)
    for l in lines:
        l["via"] = "ocr"
    _ = OCR_MODEL
    return text, lines


def pdf_file(path, project="default", name=None):
    """Extract one PDF -> result dict (results.py format, kind=pdf)."""
    from results import build_result
    text, lines = extract(path)
    return build_result(name or os.path.basename(path), project,
                        "pdf", MODEL_ID, text, lines)


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Extract text from one PDF")
    ap.add_argument("path")
    ap.add_argument("--project", default="default")
    a = ap.parse_args()
    print(json.dumps(pdf_file(a.path, a.project), indent=2))


if __name__ == "__main__":
    main()
