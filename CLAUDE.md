# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Flask web app and CLI pipeline that uses Meta's SAM 2.1 (Segment Anything Model) to automatically segment all objects in uploaded images or presentation PDFs and export them as individual transparent PNGs. Designed for extracting presentation-ready assets from photos and slide decks.

## Setup

```bash
# Create and activate virtualenv
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install SAM 2 and download the checkpoint (~400MB)
python setup_model.py
```

The setup script installs SAM 2 from GitHub (with `SAM2_BUILD_CUDA=0`) and downloads the `sam2.1_hiera_base_plus.pt` checkpoint to `checkpoints/`.

## Running

```bash
# Web app
python app.py
# Serves on http://127.0.0.1:5000

# PDF pipeline (CLI)
python pdf_pipeline.py data/pdfs/MyPresentation.pdf
python pdf_pipeline.py /external/path/deck.pdf --dpi 300 --slides 1-5 --upscale 2
```

The SAM 2.1 model loads into memory at startup (takes a few seconds). There is no test suite.

## Architecture

- **app.py** — Flask routes and session management. Each upload gets a UUID-based session. Sessions auto-cleanup after `session_ttl` seconds. Image routes: `/upload` (POST), `/segment/<session_id>` (POST), `/segment-image/`, `/download/`, `/download-all/`. PDF routes: `/upload-pdf` (POST), `/slide-image/`, `/segment-slide/`. Browse routes: `/browse/runs`, `/browse/run/<name>`, `/browse/run/<name>/<slide>`, `/browse/thumb/`, `/browse/download/`.
- **segmenter.py** — `ImageSegmenter` class wrapping SAM 2.1's `SAM2AutomaticMaskGenerator`. Handles device selection (MPS -> CPU fallback), image resizing for inference, mask filtering/sorting, and saving cropped RGBA segments with edge feathering and tight cropping.
- **pdf_pipeline.py** — CLI tool for batch processing: converts PDF pages to images via PyMuPDF, then runs segmentation on each slide. Smart caching of slide images (reuses if DPI matches). Memory-optimised for Apple Silicon — processes one slide at a time, limits render workers, clears GPU memory between slides. Also exports `pdf_to_images()` and `get_pdf_page_count()` used by the web app.
- **config.yaml** — Central configuration file with all tunable parameters and inline documentation. Covers segmentation, SAM model params, feathering, tight cropping, PDF pipeline, paths, and web server settings.
- **config.py** — Loads `config.yaml` and exposes a module-level `cfg` dict with built-in defaults. Gracefully falls back to defaults if YAML file or pyyaml is missing.
- **setup_model.py** — One-time setup: installs `sam2` package and downloads the checkpoint.
- **templates/index.html** + **static/js/app.js** + **static/css/style.css** — Single-page frontend with three-mode UI: Upload Image, Upload PDF, Browse Runs. Supports slide navigation, segment gallery with selection/filtering/sorting, and modal preview.

## Key Details

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in app.py and pdf_pipeline.py to handle unsupported MPS ops.
- The model checkpoint lives at `checkpoints/sam2.1_hiera_base_plus.pt` (gitignored).
- `uploads/`, `outputs/`, and `data/` are gitignored working directories.
- Max upload size, session TTL, host, and port are configured via `config.yaml`.
- Images larger than `max_dim` (default 3072px) are resized for inference, then masks are upscaled back.
- Segments are capped at `max_segments` (default 200) per image, sorted by area descending, filtered by `min_area` (default 500px).
- SAM runs with `torch.autocast` float16 on MPS for better throughput on Apple Silicon.
- Segments are tightly cropped to their visible content with configurable padding (`tight_crop_padding`, default 4px) to save disk space.
- Slide images are cached in `data/images/<pdf_name>/` (no timestamp). Running the pipeline again reuses cached images if the DPI matches.
- Segment runs are stored in `data/segments/<pdf_name>_<timestamp>/slide_NNN/` with `masks/`, `thumbs/`, and `rendered/` subdirectories.
- The `rendered/` folder contains the final transparent PNGs for use in Keynote/presentations. `thumbs/` is for preview only.
- External PDFs create `slide_images/` and `slide_image_segments/` next to the PDF.
- All hardcoded parameters have been moved to `config.yaml` — edit that file to tune behaviour without changing code.
- Memory management: `render_workers` (default 4) caps concurrent rendering threads. `gc.collect()` and `torch.mps.empty_cache()` are called between slides and after web requests.
