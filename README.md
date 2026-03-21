# Image Segmenter

A Flask web app that uses Meta's [SAM 2.1](https://github.com/facebookresearch/sam2) (Segment Anything Model) to automatically segment all objects in an uploaded image and export them as individual transparent PNGs. Designed for extracting presentation-ready assets from photos.

## Features

- Drag-and-drop image upload
- Automatic segmentation of all objects using SAM 2.1
- Preview all extracted segments in the browser
- Download individual segments or all as a ZIP
- Apple Silicon (MPS) acceleration with CPU fallback
- Session-based — multiple users can run concurrently, sessions auto-cleanup after 1 hour

## Requirements

- Python 3.10+
- ~400MB disk space for the model checkpoint
- Apple Silicon Mac (MPS) or any machine with a CPU

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd Image_Segmenter

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install SAM 2 and download the model checkpoint (~400MB)
python setup_model.py
```

The setup script installs `sam2` from GitHub (CPU-only, no CUDA build) and downloads `sam2.1_hiera_base_plus.pt` to `checkpoints/`.

## Running

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The model loads into memory at startup (takes a few seconds).

## Usage

1. Upload an image (JPG, PNG, WebP, BMP, or TIFF — max 50MB)
2. Click **Segment** — SAM 2.1 will detect and extract all objects
3. Browse the extracted segments in the gallery
4. Download individual PNGs or all segments as a ZIP

## Configuration

Segmentation parameters can be tuned via the API:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_area` | `500` | Minimum mask area in pixels to keep |
| `max_dim` | `4096` | Max image dimension for inference (larger images are resized) |

Up to 200 segments are returned per image, sorted by area (largest first).

## Architecture

```
app.py          — Flask routes and session management
segmenter.py    — ImageSegmenter class wrapping SAM2AutomaticMaskGenerator
setup_model.py  — One-time setup: installs sam2 and downloads checkpoint
templates/      — Jinja2 HTML template
static/js/      — Frontend JavaScript
static/css/     — Styles
checkpoints/    — Model weights (gitignored)
uploads/        — Temporary session uploads (gitignored)
outputs/        — Temporary session outputs (gitignored)
```

## Notes

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically to handle unsupported MPS ops.
- Output PNGs are full-resolution RGBA crops with light edge feathering for cleaner compositing.
- The model weights and working directories (`uploads/`, `outputs/`, `checkpoints/`) are gitignored.
