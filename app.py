#!/usr/bin/env python3
"""
Image Segmenter Web App
Upload an image → SAM 2.1 segments all objects → download as transparent PNGs.
"""

import io
import os
import shutil
import time
import uuid
import zipfile

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    send_from_directory,
    render_template,
)
from PIL import Image

from segmenter import ImageSegmenter

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
SESSION_TTL_SECONDS = 3600  # 1 hour

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model once at startup
print("Initializing SAM 2.1 model...")
segmenter = ImageSegmenter()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_sessions():
    """Remove session directories older than TTL."""
    now = time.time()
    for base in (UPLOAD_DIR, OUTPUT_DIR):
        if not os.path.exists(base):
            continue
        for name in os.listdir(base):
            path = os.path.join(base, name)
            if (
                os.path.isdir(path)
                and now - os.path.getmtime(path) > SESSION_TTL_SECONDS
            ):
                shutil.rmtree(path, ignore_errors=True)


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
    cleanup_old_sessions()

    # Find uploaded image
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    if not os.path.isdir(session_upload_dir):
        return jsonify({"error": "Session not found"}), 404

    files = os.listdir(session_upload_dir)
    if not files:
        return jsonify({"error": "No uploaded image found"}), 404

    image_path = os.path.join(session_upload_dir, files[0])

    # Parse optional parameters
    data = request.get_json(silent=True) or {}
    min_area = data.get("min_area", 500)
    max_dim = data.get("max_dim", 4096)

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

    return jsonify(
        {
            "session_id": session_id,
            "segment_count": len(segments),
            "segments": segments,
        }
    )


@app.route("/segment-image/<session_id>/<filename>")
def segment_image(session_id, filename):
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    if not os.path.isdir(session_output_dir):
        return jsonify({"error": "Session not found"}), 404

    return send_from_directory(session_output_dir, filename, mimetype="image/png")


@app.route("/download/<session_id>/<filename>")
def download(session_id, filename):
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    if not os.path.isdir(session_output_dir):
        return jsonify({"error": "Session not found"}), 404

    return send_from_directory(session_output_dir, filename, as_attachment=True)


@app.route("/download-all/<session_id>")
def download_all(session_id):
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    if not os.path.isdir(session_output_dir):
        return jsonify({"error": "Session not found"}), 404

    png_files = sorted(f for f in os.listdir(session_output_dir) if f.endswith(".png"))
    if not png_files:
        return jsonify({"error": "No segments found"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in png_files:
            zf.write(os.path.join(session_output_dir, fname), fname)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="segments.zip",
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
