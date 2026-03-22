# Image Segmenter

A Flask web app and CLI pipeline that uses Meta's [SAM 2.1](https://github.com/facebookresearch/sam2) (Segment Anything Model) to automatically segment all objects in uploaded images or presentation PDFs and export them as individual transparent PNGs. Designed for extracting presentation-ready assets from photos and slide decks.

## Features

- Drag-and-drop image upload via web UI
- **PDF pipeline** — convert a presentation PDF to slide images, then segment every slide automatically
- Automatic segmentation of all objects using SAM 2.1
- Tight cropping of segments with configurable padding to minimise file size
- Preview all extracted segments in the browser
- Download individual segments or all as a ZIP
- Configurable upscaling (1x–4x) for sticker/print quality
- Apple Silicon (MPS) acceleration with CPU fallback
- Session-based web UI — multiple users can run concurrently, sessions auto-cleanup after 1 hour
- Central `config.yaml` for all tunable parameters

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

## Running the Web App

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The model loads into memory at startup (takes a few seconds).

### Web UI Usage

1. Upload an image (JPG, PNG, WebP, BMP, or TIFF — max 50MB)
2. Click **Segment** — SAM 2.1 will detect and extract all objects
3. Browse the extracted segments in the gallery
4. Download individual PNGs or all segments as a ZIP

## PDF Pipeline (CLI)

Convert a presentation PDF into slide images, then segment each slide to extract all visual assets.

```bash
# Basic — segment all slides
python pdf_pipeline.py data/pdfs/MyPresentation.pdf

# Custom DPI and segment filter
python pdf_pipeline.py data/pdfs/MyPresentation.pdf --dpi 300 --min-area 200

# Process only slides 1–5 at 2x upscale
python pdf_pipeline.py data/pdfs/MyPresentation.pdf --slides 1-5 --upscale 2

# External PDF (outputs created next to the PDF)
python pdf_pipeline.py /path/to/external/deck.pdf
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dpi` | `200` | PDF rendering DPI (72–600) |
| `--min-area` | `500` | Minimum segment area in pixels |
| `--max-dim` | `3072` | Max inference dimension |
| `--upscale` | `1` | Output upscale factor (1–4) |
| `--slides` | all | Slide range, e.g. `1-5` or `3,7,10` |

### Output Folder Structure

**PDF inside `data/pdfs/`** (default):
```
data/images/<pdfname>_<datetime>/
  slide_001.png, slide_002.png, ...
data/segments/<pdfname>_slide_001_<datetime>/
  rendered/segment_000.png, segment_001.png, ...
```

**External PDF** (outputs next to the PDF):
```
<pdf_parent>/slide_images/<pdfname>_<datetime>/
  slide_001.png, slide_002.png, ...
<pdf_parent>/slide_image_segments/<pdfname>_slide_001_<datetime>/
  rendered/segment_000.png, segment_001.png, ...
```

## Configuration

All parameters are configured in `config.yaml`. The file is self-documented with comments explaining each parameter and its recommended range. Key sections:

| Section | Parameters |
|---------|------------|
| Segmentation | `min_area`, `max_dim`, `max_segments`, `upscale` |
| SAM 2.1 Model | `points_per_side`, `pred_iou_thresh`, `stability_score_thresh` |
| Edge Feathering | `feather_min_px`, `feather_max_px`, `feather_factor` |
| Tight Cropping | `tight_crop`, `tight_crop_padding` |
| PDF Pipeline | `pdf_dpi`, `pdf_image_format` |
| Paths | `data_dir`, `pdf_subdir`, `images_subdir`, `segments_subdir` |
| Web Server | `max_upload_mb`, `session_ttl`, `host`, `port` |

Parameters can also be overridden via CLI flags (for the PDF pipeline) or API parameters (for the web app).

## Architecture

```
app.py           — Flask routes and session management
segmenter.py     — ImageSegmenter class wrapping SAM2AutomaticMaskGenerator
pdf_pipeline.py  — CLI: PDF → slide images → segmented assets
config.yaml      — Central configuration (all tunable parameters)
config.py        — Configuration loader with built-in defaults
setup_model.py   — One-time setup: installs sam2 and downloads checkpoint
templates/       — Jinja2 HTML template
static/js/       — Frontend JavaScript
static/css/      — Styles
checkpoints/     — Model weights (gitignored)
uploads/         — Temporary session uploads (gitignored)
outputs/         — Temporary session outputs (gitignored)
data/            — PDF pipeline working directory (gitignored)
```

## Notes

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically to handle unsupported MPS ops.
- Output PNGs are full-resolution RGBA crops with edge feathering and tight cropping for cleaner compositing.
- Segments are tightly cropped to their visible content with configurable padding to save disk space.
- The model weights and working directories (`uploads/`, `outputs/`, `checkpoints/`, `data/`) are gitignored.
- The PDF pipeline saves a `pipeline_summary.json` alongside the slide images for reproducibility.
