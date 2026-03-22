# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Flask web app that uses Meta's SAM 2.1 (Segment Anything Model) to automatically segment all objects in an uploaded image and export them as individual transparent PNGs. Designed for extracting presentation-ready assets from photos.

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
python app.py
# Serves on http://127.0.0.1:5000
```

The SAM 2.1 model loads into memory at startup (takes a few seconds). There is no test suite.

## Architecture

- **app.py** — Flask routes and session management. Each upload gets a UUID-based session. Sessions auto-cleanup after 1 hour. Routes: `/upload` (POST), `/segment/<session_id>` (POST), `/segment-image/`, `/download/`, `/download-all/` (returns zip).
- **segmenter.py** — `ImageSegmenter` class wrapping SAM 2.1's `SAM2AutomaticMaskGenerator`. Handles device selection (MPS → CPU fallback), image resizing for inference, mask filtering/sorting, and saving cropped RGBA segments with edge feathering.
- **setup_model.py** — One-time setup: installs `sam2` package and downloads the checkpoint.
- **templates/index.html** + **static/js/app.js** + **static/css/style.css** — Single-page frontend with drag-and-drop upload.

## Key Details

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in app.py to handle unsupported MPS ops.
- The model checkpoint lives at `checkpoints/sam2.1_hiera_base_plus.pt` (gitignored).
- `uploads/` and `outputs/` are gitignored working directories for session data.
- Max upload size is 50MB. Images larger than `max_dim` (default 3072px) are resized for inference, then masks are upscaled back.
- Segments are capped at 200 per image, sorted by area descending, filtered by `min_area` (default 500px).
- SAM runs at 3072px inference resolution with `torch.autocast` float16 on MPS for better quality and throughput on Apple Silicon.
