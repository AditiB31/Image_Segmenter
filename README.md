# Image Segmenter

A Flask web app and CLI pipeline that uses Meta's [SAM 2.1](https://github.com/facebookresearch/sam2) (Segment Anything Model) to automatically segment all objects in uploaded images or presentation PDFs and export them as individual transparent PNGs. Designed for extracting presentation-ready assets from photos and slide decks.

## Features

- Drag-and-drop image upload via web UI
- **Folder upload** — select a folder of images to browse and annotate them in a gallery view, similar to PDF slides
- **Manual annotation** — click points, draw polygons, or drag boxes to extract specific objects with SAM 2.1 prompts
- **PDF upload** — upload a presentation PDF in the browser, browse slides, and segment individually
- **PDF pipeline** (CLI) — batch process a PDF to extract all visual assets from every slide
- **Browse runs** — view and download segments from previous pipeline runs in the web UI
- **Auto-open file picker** — clicking the Upload Image or Upload PDF tab immediately opens the file picker
- Automatic segmentation of all objects using SAM 2.1
- Tight cropping of segments with configurable padding to minimise file size
- Preview all extracted segments in the browser
- Download individual segments or all as a ZIP
- Configurable upscaling (1x-4x) for sticker/print quality
- **Optimised edge handling** — tuned feathering, contour smoothing, and anti-aliasing for clean segment edges
- Apple Silicon (MPS) acceleration with CPU fallback
- **Smart caching** — slide images are converted once and reused across pipeline runs
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

## Running Tests

```bash
# Install dev dependencies (pytest)
pip install -r requirements-dev.txt

# Run the test suite
python -m pytest tests/
```

## Running the Web App

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The model loads into memory at startup (takes a few seconds).

### Web UI Modes

The web UI has three modes accessible via tabs:

1. **Upload Image** — Upload a single image (JPG, PNG, WebP, BMP, TIFF) or select a folder of images. Clicking the tab auto-opens the file picker. Single images open directly in the annotation canvas. Multiple images or a folder display a gallery grid (like PDF slides) where you click any image to annotate it. Tools include points, polygons, and bounding boxes. Each segment appears immediately in a gallery strip with per-segment removal, quality selector (1x-4x), and ZIP download. An "Auto Segment All" button is available in the toolbar for automatic segmentation.

2. **Upload PDF** — Upload a presentation PDF. Clicking the tab auto-opens the file picker. Slides are converted to images (cached for future use). Click any slide to open it in the annotation canvas for manual segment extraction. Navigate between slides with prev/next controls. The typical workflow is: click slide → annotate segments → download → back to slides → next slide.

3. **Browse Runs** — View results from previous CLI pipeline runs. Select a run, then a slide, to see its extracted segments. Download rendered PNGs directly.

## PDF Pipeline (CLI)

Convert a presentation PDF into slide images, then segment each slide to extract all visual assets.

```bash
# Basic — segment all slides
python pdf_pipeline.py data/pdfs/MyPresentation.pdf

# Custom DPI and segment filter
python pdf_pipeline.py data/pdfs/MyPresentation.pdf --dpi 300 --min-area 200

# Process only slides 1-5 at 2x upscale
python pdf_pipeline.py data/pdfs/MyPresentation.pdf --slides 1-5 --upscale 2

# External PDF (outputs created next to the PDF)
python pdf_pipeline.py /path/to/external/deck.pdf
```

**Smart caching:** Slide images are cached in `data/images/<pdf_name>/`. Running the pipeline again on the same PDF skips the conversion step. If you change the `--dpi`, cached images are automatically regenerated.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dpi` | `200` | PDF rendering DPI (72-600) |
| `--min-area` | `6000` | Minimum segment area in pixels |
| `--max-dim` | `2048` | Max inference dimension |
| `--upscale` | `1` | Output upscale factor (1-4) |
| `--slides` | all | Slide range, e.g. `1-5` or `3,7,10` |

### Output Folder Structure

**Slide images** (cached, no timestamp — converted once per PDF):
```
data/images/<pdf_name>/
    conversion_info.json
    slide_001.png, slide_002.png, ...
```

**Segments** (per run, timestamped):
```
data/segments/<pdf_name>_<timestamp>/
    run_info.json
    slide_001/
        masks/          (compact binary masks + metadata)
        thumbs/         (small preview thumbnails)
        rendered/       (full-resolution transparent PNGs)
    slide_002/
        ...
```

**External PDF** (outputs next to the PDF):
```
<pdf_parent>/slide_images/<pdf_name>/
    slide_001.png, slide_002.png, ...
<pdf_parent>/slide_image_segments/<pdf_name>_<timestamp>/
    slide_001/rendered/segment_000.png, ...
```

### Using Segments in Keynote / Presentations

The **`rendered/`** folder inside each slide's segment directory contains the final transparent PNGs ready for drag-and-drop into Keynote, Google Slides, or PowerPoint. These are full-resolution RGBA images with smooth edge feathering and tight cropping.

- **`rendered/`** — Use these. Full-quality transparent PNGs at the configured upscale level.
- `thumbs/` — Preview only. Small thumbnails used by the web UI gallery.
- `masks/` — Internal data. Compact binary masks used for on-demand rendering.

## Configuration

All parameters are configured in `config.yaml`. The file is self-documented with comments explaining each parameter and its recommended range. Key sections:

| Section | Parameters |
|---------|------------|
| Segmentation | `min_area`, `max_dim`, `max_segments`, `upscale` |
| SAM 2.1 Model | `points_per_side`, `pred_iou_thresh`, `stability_score_thresh` |
| Edge Feathering | `feather_min_px`, `feather_max_px`, `feather_factor` |
| Tight Cropping | `tight_crop`, `tight_crop_padding` |
| PDF Pipeline | `pdf_dpi`, `pdf_image_format`, `render_workers` |
| Paths | `data_dir`, `pdf_subdir`, `images_subdir`, `segments_subdir` |
| Web Server | `max_upload_mb`, `session_ttl`, `host`, `port` |

Parameters can also be overridden via CLI flags (for the PDF pipeline) or API parameters (for the web app).

## Architecture

```
app.py           — Flask routes: image upload, PDF upload, annotation, segmentation, browsing
segmenter.py     — ImageSegmenter class wrapping SAM2AutomaticMaskGenerator + SAM2ImagePredictor
pdf_pipeline.py  — CLI: PDF -> cached slide images -> segmented assets
config.yaml      — Central configuration (all tunable parameters)
config.py        — Configuration loader with built-in defaults
setup_model.py   — One-time setup: installs sam2 and downloads checkpoint
templates/       — Jinja2 HTML template
static/js/       — Frontend JavaScript (three-mode UI)
static/css/      — Styles (light/dark theme)
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
- Memory-optimised for Apple Silicon: processes one slide at a time, limits concurrent render workers, clears GPU memory between slides.
