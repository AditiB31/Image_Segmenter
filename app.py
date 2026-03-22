#!/usr/bin/env python3
"""
Image Segmenter Web App
Upload an image → SAM 2.1 segments all objects → download as transparent PNGs.
"""

import gc
import glob
import json
import os
import shutil
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np

from flask import (
    Flask,
    after_this_request,
    jsonify,
    request,
    send_file,
    send_from_directory,
    render_template,
)
from PIL import Image

from config import cfg
from pdf_pipeline import pdf_to_images, get_pdf_page_count
from segmenter import ImageSegmenter

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = cfg["max_upload_mb"] * 1024 * 1024

# Track whether MPS is available for memory management
_HAS_MPS = False
try:
    import torch
    _HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    pass


def _cleanup_memory():
    """Free GPU and Python memory. Call after every major operation."""
    if _HAS_MPS:
        import torch
        torch.mps.empty_cache()
    gc.collect()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
ALLOWED_PDF_EXTENSIONS = {"pdf"}
DATA_DIR = os.path.join(BASE_DIR, cfg["data_dir"])
SEGMENTS_DIR = os.path.join(DATA_DIR, cfg["segments_subdir"])
IMAGES_DIR = os.path.join(DATA_DIR, cfg["images_subdir"])
SESSION_TTL_SECONDS = cfg["session_ttl"]

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model once at startup
print("Initializing SAM 2.1 model...")
segmenter = ImageSegmenter()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _safe_component(name):
    """Validate a path component to prevent directory traversal."""
    if not name or "\0" in name or "/" in name or "\\" in name or ".." in name:
        return False
    return True


def _get_image_path(session_id):
    """Return path to the original uploaded image, or None if missing."""
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    files = os.listdir(session_upload_dir) if os.path.isdir(session_upload_dir) else []
    return os.path.join(session_upload_dir, files[0]) if files else None


def _resolve_slide_context(session_id, slide):
    """Resolve image path and segment dir for a slide-based or image-based session.

    Returns (image_path, seg_output_dir) or (None, None) if invalid.
    """
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    if slide:
        if not _safe_component(slide):
            return None, None
        pdf_session = _load_pdf_session(session_id)
        if pdf_session is None:
            return None, None
        image_path = os.path.join(pdf_session["images_dir"], f"{slide}.png")
        seg_output_dir = os.path.join(session_output_dir, slide)
    else:
        image_path = _get_image_path(session_id)
        seg_output_dir = session_output_dir
    return image_path, seg_output_dir


def _load_pdf_session(session_id):
    """Load PDF session metadata, or None if not a PDF session."""
    path = os.path.join(OUTPUT_DIR, session_id, "pdf_session.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


_last_cleanup = 0
_CLEANUP_INTERVAL = 60  # seconds between cleanup runs


def cleanup_old_sessions():
    """Remove session directories older than TTL. Throttled to run at most once per minute."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    for base in (UPLOAD_DIR, OUTPUT_DIR):
        if not os.path.exists(base):
            continue
        try:
            entries = os.listdir(base)
        except OSError:
            continue
        for name in entries:
            path = os.path.join(base, name)
            try:
                if (
                    os.path.isdir(path)
                    and now - os.path.getmtime(path) > SESSION_TTL_SECONDS
                ):
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass  # directory may have been removed by concurrent request


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    cleanup_old_sessions()

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify(
            {"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}
        ), 400

    session_id = uuid.uuid4().hex[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()

    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)

    filename = f"original.{ext}"
    filepath = os.path.join(session_upload_dir, filename)
    file.save(filepath)

    # Read dimensions
    with Image.open(filepath) as img:
        width, height = img.size

    return jsonify(
        {
            "session_id": session_id,
            "filename": file.filename,
            "width": width,
            "height": height,
        }
    )


@app.route("/segment/<session_id>", methods=["POST"])
def segment(session_id):
    if not _safe_component(session_id):
        return jsonify({"error": "Invalid session"}), 400
    cleanup_old_sessions()

    image_path = _get_image_path(session_id)
    if image_path is None:
        return jsonify({"error": "Session not found"}), 404

    # Parse optional parameters (fall back to config defaults)
    data = request.get_json(silent=True) or {}
    min_area = data.get("min_area", cfg["min_area"])
    max_dim = data.get("max_dim", cfg["max_dim"])

    session_output_dir = os.path.join(OUTPUT_DIR, session_id)

    try:
        segments = segmenter.segment(
            image_path, session_output_dir, min_area=min_area, max_dim=max_dim
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            return jsonify(
                {
                    "error": "Out of memory. Try a smaller image or reduce max_dim.",
                }
            ), 500
        raise
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    _cleanup_memory()

    return jsonify(
        {
            "session_id": session_id,
            "segment_count": len(segments),
            "segments": segments,
        }
    )


@app.route("/segment-image/<session_id>/<filename>")
def segment_image(session_id, filename):
    """Serve thumbnail for gallery preview. Supports ?slide=slide_001 for PDF sessions."""
    if not _safe_component(session_id) or not _safe_component(filename):
        return jsonify({"error": "Invalid path"}), 400
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    slide = request.args.get("slide")
    if slide:
        if not _safe_component(slide):
            return jsonify({"error": "Invalid slide name"}), 400
        thumbs_dir = os.path.join(session_output_dir, slide, "thumbs")
    else:
        thumbs_dir = os.path.join(session_output_dir, "thumbs")
    if not os.path.isdir(thumbs_dir):
        return jsonify({"error": "Session not found"}), 404

    return send_from_directory(thumbs_dir, filename, mimetype="image/png")


@app.route("/download/<session_id>/<filename>")
def download(session_id, filename):
    """Render full-res segment on demand and serve it. Supports ?slide= for PDF sessions."""
    if not _safe_component(session_id) or not _safe_component(filename):
        return jsonify({"error": "Invalid path"}), 400
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    if not os.path.isdir(session_output_dir):
        return jsonify({"error": "Session not found"}), 404

    try:
        idx = int(filename.split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return jsonify({"error": "Invalid filename"}), 400

    slide = request.args.get("slide")
    image_path, seg_output_dir = _resolve_slide_context(session_id, slide)
    if image_path is None or not os.path.exists(image_path):
        return jsonify({"error": "Original image not found"}), 404

    max_upscale = cfg.get("max_upscale", 4)
    upscale = max(1, min(int(request.args.get("upscale", cfg["upscale"])), max_upscale))
    render_dir = os.path.join(seg_output_dir, f"render_{upscale}x")
    out_path = os.path.join(render_dir, filename)

    if not os.path.exists(out_path):
        os.makedirs(render_dir, exist_ok=True)
        try:
            result = segmenter.render_segment(
                image_path, seg_output_dir, idx, upscale=upscale, out_path=out_path
            )
        finally:
            _cleanup_memory()
        if result is None:
            return jsonify({"error": "Segment not found"}), 404

    return send_file(out_path, as_attachment=True)


@app.route("/download-all/<session_id>", methods=["GET", "POST"])
def download_all(session_id):
    """Render selected (or all) segments at requested upscale and ZIP them."""
    if not _safe_component(session_id):
        return jsonify({"error": "Invalid session"}), 400
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    if not os.path.isdir(session_output_dir):
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json(silent=True) or {}
    slide = data.get("slide")

    image_path, seg_output_dir = _resolve_slide_context(session_id, slide)
    if image_path is None or not os.path.exists(image_path):
        return jsonify({"error": "Original image not found"}), 404

    meta_path = os.path.join(seg_output_dir, "masks", "meta.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "No segments found"}), 404

    with open(meta_path) as f:
        meta = json.load(f)

    all_indices = [s["index"] for s in meta["segments"]]
    indices = data.get("indices", all_indices)
    max_upscale = cfg.get("max_upscale", 4)
    upscale = max(1, min(int(data.get("upscale", cfg["upscale"])), max_upscale))

    if not indices:
        return jsonify({"error": "No segments selected"}), 400

    # Render upscaled segments in parallel using ThreadPoolExecutor
    render_dir = os.path.join(seg_output_dir, f"render_{upscale}x")
    os.makedirs(render_dir, exist_ok=True)

    # Load original image once; share across threads (numpy reads are thread-safe)
    with Image.open(image_path) as _img:
        image_array = np.array(_img.convert("RGB"))

    def _render_one(idx):
        filename = f"segment_{idx:03d}.png"
        out_path = os.path.join(render_dir, filename)
        if not os.path.exists(out_path):
            result = segmenter.render_segment(
                image_path,
                seg_output_dir,
                idx,
                meta=meta,
                upscale=upscale,
                out_path=out_path,
                image_array=image_array,
            )
            if result is None:
                return None
        return (out_path, filename)

    rendered = []
    workers = min(os.cpu_count() or 4, cfg["render_workers"])
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_render_one, idx): idx for idx in indices}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    rendered.append(result)
    finally:
        del image_array
        _cleanup_memory()

    if not rendered:
        shutil.rmtree(render_dir, ignore_errors=True)
        return jsonify({"error": "No segments could be rendered"}), 404

    zip_path = os.path.join(seg_output_dir, f"segments_{upscale}x.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, fname in rendered:
            zf.write(path, fname)

    # Free disk space and memory
    shutil.rmtree(render_dir, ignore_errors=True)
    _cleanup_memory()

    @after_this_request
    def _cleanup_zip(response):
        try:
            os.remove(zip_path)
        except OSError:
            pass
        return response

    # Build descriptive download name
    if slide:
        download_name = f"{slide}_segments_{upscale}x.zip"
    else:
        image_path_base = os.path.splitext(os.path.basename(image_path))[0]
        download_name = f"{image_path_base}_segments_{upscale}x.zip"

    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


# ── Manual Annotation Routes ───────────────────────────────────────────


@app.route("/original-image/<session_id>")
def original_image(session_id):
    """Serve the original uploaded image for the annotation canvas."""
    if not _safe_component(session_id):
        return jsonify({"error": "Invalid session"}), 400
    image_path = _get_image_path(session_id)
    if image_path is None:
        return jsonify({"error": "Session not found"}), 404
    return send_file(image_path)


@app.route("/annotate/<session_id>", methods=["POST"])
def annotate(session_id):
    """Run SAM prediction with user-provided point/box/contour prompts."""
    if not _safe_component(session_id):
        return jsonify({"error": "Invalid session"}), 400
    cleanup_old_sessions()

    image_path = _get_image_path(session_id)
    if image_path is None:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json(silent=True) or {}
    prompts = data.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        return jsonify({"error": "No prompts provided"}), 400
    if len(prompts) > 50:
        return jsonify({"error": "Too many prompts (max 50)"}), 400

    max_dim = data.get("max_dim", cfg["max_dim"])
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)

    try:
        segments = segmenter.predict_with_prompts(
            image_path, session_output_dir, prompts, max_dim=max_dim
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            return jsonify(
                {"error": "Out of memory. Try a smaller image."}
            ), 500
        return jsonify({"error": "Segmentation failed"}), 500
    except Exception:
        return jsonify({"error": "Segmentation failed"}), 500

    _cleanup_memory()

    return jsonify(
        {
            "session_id": session_id,
            "segment_count": len(segments),
            "segments": segments,
        }
    )


@app.route("/annotate-slide/<session_id>/<int:slide_index>", methods=["POST"])
def annotate_slide(session_id, slide_index):
    """Run SAM prediction with user prompts on a PDF slide image."""
    if not _safe_component(session_id):
        return jsonify({"error": "Invalid session"}), 400
    cleanup_old_sessions()

    pdf_session = _load_pdf_session(session_id)
    if pdf_session is None:
        return jsonify({"error": "PDF session not found"}), 404

    if slide_index < 0 or slide_index >= len(pdf_session["slides"]):
        return jsonify({"error": "Slide index out of range"}), 400

    slide_info = pdf_session["slides"][slide_index]
    slide_name = slide_info["name"]
    slide_path = os.path.join(pdf_session["images_dir"], slide_info["filename"])

    if not os.path.exists(slide_path):
        return jsonify({"error": "Slide image not found"}), 404

    data = request.get_json(silent=True) or {}
    prompts = data.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        return jsonify({"error": "No prompts provided"}), 400
    if len(prompts) > 50:
        return jsonify({"error": "Too many prompts (max 50)"}), 400

    max_dim = data.get("max_dim", cfg["max_dim"])
    seg_output_dir = os.path.join(OUTPUT_DIR, session_id, slide_name)

    try:
        segments = segmenter.predict_with_prompts(
            slide_path, seg_output_dir, prompts, max_dim=max_dim
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            return jsonify(
                {"error": "Out of memory. Try a smaller image."}
            ), 500
        return jsonify({"error": "Segmentation failed"}), 500
    except Exception:
        return jsonify({"error": "Segmentation failed"}), 500

    _cleanup_memory()

    return jsonify(
        {
            "session_id": session_id,
            "slide_name": slide_name,
            "slide_index": slide_index,
            "segment_count": len(segments),
            "segments": segments,
        }
    )


# ── PDF Upload Routes ──────────────────────────────────────────────────


@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    """Upload PDF, convert to cached slide images, return slide list."""
    cleanup_old_sessions()

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if (
        not file.filename
        or "." not in file.filename
        or file.filename.rsplit(".", 1)[1].lower() not in ALLOWED_PDF_EXTENSIONS
    ):
        return jsonify({"error": "Please upload a PDF file"}), 400

    session_id = uuid.uuid4().hex[:12]
    pdf_name = os.path.splitext(file.filename)[0]

    # Save PDF to uploads
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)
    pdf_path = os.path.join(session_upload_dir, "original.pdf")
    file.save(pdf_path)

    # Convert to cached slide images
    dpi = cfg["pdf_dpi"]
    img_fmt = cfg["pdf_image_format"]
    images_dir = os.path.join(IMAGES_DIR, pdf_name)

    try:
        total_pages = get_pdf_page_count(pdf_path)
    except Exception:
        shutil.rmtree(session_upload_dir, ignore_errors=True)
        return jsonify({"error": "Could not read PDF. The file may be corrupted or password-protected."}), 400

    if total_pages == 0:
        shutil.rmtree(session_upload_dir, ignore_errors=True)
        return jsonify({"error": "PDF has no pages."}), 400

    # Check cache
    cached = False
    info_path = os.path.join(images_dir, "conversion_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if info.get("dpi") == dpi:
            existing = sorted(glob.glob(
                os.path.join(images_dir, f"slide_*.{img_fmt}")
            ))
            if existing:
                cached = True

    if not cached:
        os.makedirs(images_dir, exist_ok=True)
        try:
            pdf_to_images(pdf_path, images_dir, dpi=dpi, fmt=img_fmt)
        except Exception as e:
            shutil.rmtree(session_upload_dir, ignore_errors=True)
            return jsonify({"error": f"Failed to convert PDF: {e}"}), 500
        conversion_info = {
            "dpi": dpi,
            "page_count": total_pages,
            "format": img_fmt,
            "converted_at": datetime.now().isoformat(),
            "pdf_name": pdf_name,
        }
        with open(os.path.join(images_dir, "conversion_info.json"), "w") as f:
            json.dump(conversion_info, f, indent=2)

    # Build slide list with dimensions
    slide_files = sorted(glob.glob(os.path.join(images_dir, f"slide_*.{img_fmt}")))
    slides = []
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(session_output_dir, exist_ok=True)
    thumbs_dir = os.path.join(session_output_dir, "slide_thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    for slide_path in slide_files:
        fname = os.path.basename(slide_path)
        name = os.path.splitext(fname)[0]
        with Image.open(slide_path) as img:
            w, h = img.size
            # Generate slide thumbnail (max 400px)
            thumb_scale = min(400 / w, 400 / h, 1.0)
            if thumb_scale < 1.0:
                thumb = img.resize(
                    (max(1, int(w * thumb_scale)), max(1, int(h * thumb_scale))),
                    Image.BILINEAR,
                )
            else:
                thumb = img.copy()
            thumb.save(os.path.join(thumbs_dir, fname))
            thumb.close()
        slides.append({"name": name, "filename": fname, "width": w, "height": h})

    # Save PDF session metadata
    pdf_session = {
        "pdf_name": pdf_name,
        "images_dir": images_dir,
        "total_pages": total_pages,
        "slides": slides,
        "cached": cached,
    }
    with open(os.path.join(session_output_dir, "pdf_session.json"), "w") as f:
        json.dump(pdf_session, f, indent=2)

    _cleanup_memory()

    return jsonify({
        "session_id": session_id,
        "pdf_name": pdf_name,
        "total_pages": total_pages,
        "slides": slides,
        "cached": cached,
    })


@app.route("/slide-image/<session_id>/<filename>")
def slide_image(session_id, filename):
    """Serve slide thumbnail for the PDF slide browser."""
    if not _safe_component(session_id) or not _safe_component(filename):
        return jsonify({"error": "Invalid path"}), 400
    thumbs_dir = os.path.join(OUTPUT_DIR, session_id, "slide_thumbs")
    if not os.path.isdir(thumbs_dir):
        return jsonify({"error": "Session not found"}), 404
    return send_from_directory(thumbs_dir, filename, mimetype="image/png")


@app.route("/original-slide-image/<session_id>/<int:slide_index>")
def original_slide_image(session_id, slide_index):
    """Serve the full-resolution slide image for the annotation canvas."""
    if not _safe_component(session_id):
        return jsonify({"error": "Invalid session"}), 400
    pdf_session = _load_pdf_session(session_id)
    if pdf_session is None:
        return jsonify({"error": "PDF session not found"}), 404
    if slide_index < 0 or slide_index >= len(pdf_session["slides"]):
        return jsonify({"error": "Slide index out of range"}), 400
    slide_info = pdf_session["slides"][slide_index]
    slide_path = os.path.join(pdf_session["images_dir"], slide_info["filename"])
    if not os.path.exists(slide_path):
        return jsonify({"error": "Slide image not found"}), 404
    return send_file(slide_path)


@app.route("/segment-slide/<session_id>/<int:slide_index>", methods=["POST"])
def segment_slide(session_id, slide_index):
    """Segment a single slide from a PDF upload session."""
    if not _safe_component(session_id):
        return jsonify({"error": "Invalid session"}), 400
    pdf_session = _load_pdf_session(session_id)
    if pdf_session is None:
        return jsonify({"error": "PDF session not found"}), 404

    if slide_index < 0 or slide_index >= len(pdf_session["slides"]):
        return jsonify({"error": "Slide index out of range"}), 400

    slide_info = pdf_session["slides"][slide_index]
    slide_name = slide_info["name"]
    slide_path = os.path.join(pdf_session["images_dir"], slide_info["filename"])

    if not os.path.exists(slide_path):
        return jsonify({"error": "Slide image not found"}), 404

    data = request.get_json(silent=True) or {}
    min_area = data.get("min_area", cfg["min_area"])
    max_dim = data.get("max_dim", cfg["max_dim"])

    seg_output_dir = os.path.join(OUTPUT_DIR, session_id, slide_name)

    # Check for cached segmentation results with matching parameters
    meta_path = os.path.join(seg_output_dir, "masks", "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            cached_settings = meta.get("settings", {})
            if (cached_settings.get("min_area") == min_area
                    and cached_settings.get("max_dim") == max_dim):
                return jsonify({
                    "session_id": session_id,
                    "slide_name": slide_name,
                    "slide_index": slide_index,
                    "segment_count": len(meta["segments"]),
                    "segments": meta["segments"],
                    "cached": True,
                })
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted cache, re-segment

    try:
        segments = segmenter.segment(
            slide_path, seg_output_dir, min_area=min_area, max_dim=max_dim
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            return jsonify({
                "error": "Out of memory. Try a smaller max_dim.",
            }), 500
        raise
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    _cleanup_memory()

    return jsonify({
        "session_id": session_id,
        "slide_name": slide_name,
        "slide_index": slide_index,
        "segment_count": len(segments),
        "segments": segments,
    })


# ── Browse Existing Runs ───────────────────────────────────────────────


@app.route("/browse/runs")
def browse_runs():
    """List available pipeline runs from data/segments/."""
    cleanup_old_sessions()

    if not os.path.isdir(SEGMENTS_DIR):
        return jsonify({"runs": []})

    runs = []
    for name in sorted(os.listdir(SEGMENTS_DIR), reverse=True):
        run_path = os.path.join(SEGMENTS_DIR, name)
        if not os.path.isdir(run_path):
            continue

        # New format: has run_info.json
        info_path = os.path.join(run_path, "run_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            runs.append({
                "name": name,
                "pdf_name": info.get("pdf", name),
                "timestamp": info.get("timestamp", ""),
                "total_segments": info.get("total_segments", 0),
                "slide_count": len(info.get("slides", [])),
                "settings": info.get("settings", {}),
            })
        else:
            # Legacy format: single-slide segment directory with masks/ inside
            if os.path.isdir(os.path.join(run_path, "masks")):
                meta_path = os.path.join(run_path, "masks", "meta.json")
                seg_count = 0
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    seg_count = len(meta.get("segments", []))
                runs.append({
                    "name": name,
                    "pdf_name": name.rsplit("_slide_", 1)[0] if "_slide_" in name else name,
                    "timestamp": "",
                    "total_segments": seg_count,
                    "slide_count": 1,
                    "settings": {},
                    "legacy": True,
                })

    return jsonify({"runs": runs})


@app.route("/browse/run/<run_name>")
def browse_run(run_name):
    """List slides in a pipeline run with segment counts."""
    if not _safe_component(run_name):
        return jsonify({"error": "Invalid run name"}), 400
    run_path = os.path.join(SEGMENTS_DIR, run_name)
    if not os.path.isdir(run_path):
        return jsonify({"error": "Run not found"}), 404

    info_path = os.path.join(run_path, "run_info.json")
    slides = []

    if os.path.exists(info_path):
        # New format: subdirectories per slide
        with open(info_path) as f:
            info = json.load(f)
        for entry in sorted(os.listdir(run_path)):
            slide_dir = os.path.join(run_path, entry)
            if not os.path.isdir(slide_dir) or not entry.startswith("slide_"):
                continue
            meta_path = os.path.join(slide_dir, "masks", "meta.json")
            seg_count = 0
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                seg_count = len(meta.get("segments", []))
            has_rendered = os.path.isdir(os.path.join(slide_dir, "rendered"))
            slides.append({
                "name": entry,
                "segment_count": seg_count,
                "has_rendered": has_rendered,
            })
        # Try to find slide image dimensions from cached images
        images_dir = info.get("images_dir", "")
        return jsonify({
            "run_name": run_name,
            "pdf_name": info.get("pdf", run_name),
            "timestamp": info.get("timestamp", ""),
            "settings": info.get("settings", {}),
            "slides": slides,
            "images_dir": images_dir,
        })
    else:
        # Legacy format: single slide
        meta_path = os.path.join(run_path, "masks", "meta.json")
        seg_count = 0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            seg_count = len(meta.get("segments", []))
        slides.append({
            "name": run_name,
            "segment_count": seg_count,
            "has_rendered": os.path.isdir(os.path.join(run_path, "rendered")),
            "legacy": True,
        })
        return jsonify({
            "run_name": run_name,
            "pdf_name": run_name,
            "slides": slides,
        })


@app.route("/browse/slide-image/<run_name>/<slide_name>")
def browse_slide_image(run_name, slide_name):
    """Serve a slide thumbnail from cached images for browse mode."""
    if not _safe_component(run_name) or not _safe_component(slide_name):
        return jsonify({"error": "Invalid path"}), 400
    run_path = os.path.join(SEGMENTS_DIR, run_name)
    info_path = os.path.join(run_path, "run_info.json")
    if not os.path.exists(info_path):
        return jsonify({"error": "Run not found"}), 404
    with open(info_path) as f:
        info = json.load(f)
    images_dir = info.get("images_dir", "")
    if not images_dir or not os.path.isdir(images_dir):
        return jsonify({"error": "Images not found"}), 404
    # Try common extensions
    for ext in ("png", "jpg", "jpeg"):
        img_path = os.path.join(images_dir, f"{slide_name}.{ext}")
        if os.path.exists(img_path):
            return send_file(img_path, mimetype=f"image/{ext}")
    return jsonify({"error": "Slide image not found"}), 404


@app.route("/browse/run/<run_name>/<slide_name>")
def browse_slide(run_name, slide_name):
    """Return segment metadata for a slide in a pipeline run."""
    if not _safe_component(run_name) or not _safe_component(slide_name):
        return jsonify({"error": "Invalid path"}), 400
    run_path = os.path.join(SEGMENTS_DIR, run_name)

    # New format: slide is a subdirectory
    slide_dir = os.path.join(run_path, slide_name)
    if os.path.isdir(slide_dir):
        meta_path = os.path.join(slide_dir, "masks", "meta.json")
    else:
        # Legacy format: run_name IS the slide directory
        slide_dir = run_path
        meta_path = os.path.join(run_path, "masks", "meta.json")

    if not os.path.exists(meta_path):
        return jsonify({"error": "Segment data not found"}), 404

    with open(meta_path) as f:
        meta = json.load(f)

    return jsonify({
        "run_name": run_name,
        "slide_name": slide_name,
        "segments": meta.get("segments", []),
        "has_rendered": os.path.isdir(os.path.join(slide_dir, "rendered")),
    })


@app.route("/browse/thumb/<run_name>/<slide_name>/<filename>")
def browse_thumb(run_name, slide_name, filename):
    """Serve segment thumbnail from a pipeline run."""
    if not all(_safe_component(c) for c in (run_name, slide_name, filename)):
        return jsonify({"error": "Invalid path"}), 400
    run_path = os.path.join(SEGMENTS_DIR, run_name)

    # New format
    thumbs_dir = os.path.join(run_path, slide_name, "thumbs")
    if not os.path.isdir(thumbs_dir):
        # Legacy: thumbs directly in run directory
        thumbs_dir = os.path.join(run_path, "thumbs")
    if not os.path.isdir(thumbs_dir):
        return jsonify({"error": "Not found"}), 404

    return send_from_directory(thumbs_dir, filename, mimetype="image/png")


@app.route("/browse/download/<run_name>/<slide_name>/<filename>")
def browse_download(run_name, slide_name, filename):
    """Serve rendered segment PNG from a pipeline run."""
    if not all(_safe_component(c) for c in (run_name, slide_name, filename)):
        return jsonify({"error": "Invalid path"}), 400
    run_path = os.path.join(SEGMENTS_DIR, run_name)

    # New format
    rendered_dir = os.path.join(run_path, slide_name, "rendered")
    if not os.path.isdir(rendered_dir):
        # Legacy
        rendered_dir = os.path.join(run_path, "rendered")
    if not os.path.isdir(rendered_dir):
        return jsonify({"error": "Rendered segments not found"}), 404

    filepath = os.path.join(rendered_dir, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    return send_file(filepath, as_attachment=True)


@app.route("/browse/download-all/<run_name>/<slide_name>", methods=["POST"])
def browse_download_all(run_name, slide_name):
    """ZIP selected rendered segments from a pipeline run."""
    if not _safe_component(run_name) or not _safe_component(slide_name):
        return jsonify({"error": "Invalid path"}), 400

    run_path = os.path.join(SEGMENTS_DIR, run_name)

    # Find rendered directory (new or legacy format)
    rendered_dir = os.path.join(run_path, slide_name, "rendered")
    if not os.path.isdir(rendered_dir):
        rendered_dir = os.path.join(run_path, "rendered")
    if not os.path.isdir(rendered_dir):
        return jsonify({"error": "Rendered segments not found"}), 404

    data = request.get_json(silent=True) or {}
    requested_indices = data.get("indices")

    # Collect files to zip
    files = []
    for fname in sorted(os.listdir(rendered_dir)):
        if not fname.endswith(".png"):
            continue
        if requested_indices is not None:
            try:
                idx = int(fname.split("_")[1].split(".")[0])
                if idx not in requested_indices:
                    continue
            except (IndexError, ValueError):
                continue
        files.append((os.path.join(rendered_dir, fname), fname))

    if not files:
        return jsonify({"error": "No segments found"}), 404

    zip_path = os.path.join(run_path, slide_name, f"browse_download.zip")
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, fname in files:
            zf.write(path, fname)

    @after_this_request
    def _cleanup_zip(response):
        try:
            os.remove(zip_path)
        except OSError:
            pass
        return response

    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{slide_name}_segments.zip",
    )


if __name__ == "__main__":
    app.run(debug=False, host=cfg["host"], port=cfg["port"])
