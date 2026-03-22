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

- **app.py** — Flask routes and session management. Each upload gets a UUID-based session. Sessions auto-cleanup after `session_ttl` seconds. Routes: `/upload` (POST), `/segment/<session_id>` (POST), `/segment-image/`, `/download/`, `/download-all/` (returns zip).
- **segmenter.py** — `ImageSegmenter` class wrapping SAM 2.1's `SAM2AutomaticMaskGenerator`. Handles device selection (MPS -> CPU fallback), image resizing for inference, mask filtering/sorting, and saving cropped RGBA segments with edge feathering and tight cropping.
- **pdf_pipeline.py** — CLI tool for batch processing: converts PDF pages to images via PyMuPDF, then runs segmentation on each slide. Smart folder structure based on whether the PDF is internal (`data/pdfs/`) or external. Memory-optimised for Apple Silicon — processes one slide at a time, clears GPU memory between slides.
- **config.yaml** — Central configuration file with all tunable parameters and inline documentation. Covers segmentation, SAM model params, feathering, tight cropping, PDF pipeline, paths, and web server settings.
- **config.py** — Loads `config.yaml` and exposes a module-level `cfg` dict with built-in defaults. Gracefully falls back to defaults if YAML file or pyyaml is missing.
- **setup_model.py** — One-time setup: installs `sam2` package and downloads the checkpoint.
- **templates/index.html** + **static/js/app.js** + **static/css/style.css** — Single-page frontend with drag-and-drop upload.

## Key Details

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in app.py and pdf_pipeline.py to handle unsupported MPS ops.
- The model checkpoint lives at `checkpoints/sam2.1_hiera_base_plus.pt` (gitignored).
- `uploads/`, `outputs/`, and `data/` are gitignored working directories.
- Max upload size, session TTL, host, and port are configured via `config.yaml`.
- Images larger than `max_dim` (default 3072px) are resized for inference, then masks are upscaled back.
- Segments are capped at `max_segments` (default 200) per image, sorted by area descending, filtered by `min_area` (default 500px).
- SAM runs with `torch.autocast` float16 on MPS for better throughput on Apple Silicon.
- Segments are tightly cropped to their visible content with configurable padding (`tight_crop_padding`, default 4px) to save disk space.
- PDF pipeline output structure: `data/images/<name>_<datetime>/` for slide images, `data/segments/<name>_slide_NNN_<datetime>/rendered/` for segment PNGs. External PDFs create `slide_images/` and `slide_image_segments/` next to the PDF.
- All hardcoded parameters have been moved to `config.yaml` — edit that file to tune behaviour without changing code.
