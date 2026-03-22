#!/usr/bin/env python3
"""
PDF Pipeline: Convert a presentation PDF into slide images, then run
SAM 2.1 segmentation on each slide to extract individual assets.

Usage:
    python pdf_pipeline.py data/pdfs/MyPresentation.pdf
    python pdf_pipeline.py /path/to/external.pdf --dpi 300 --min-area 200
    python pdf_pipeline.py deck.pdf --slides 1-5 --upscale 2

Folder structure (default, PDF inside data/pdfs/):
    data/images/<pdfname>/slide_001.png, ...            (cached, no timestamp)
    data/segments/<pdfname>_<datetime>/slide_001/...    (per-run, timestamped)

Folder structure (external PDF):
    <pdf_parent>/slide_images/<pdfname>/slide_001.png, ...
    <pdf_parent>/slide_image_segments/<pdfname>_<datetime>/slide_001/...
"""

import argparse
import gc
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from config import cfg


def pdf_to_images(pdf_path, output_dir, dpi=200, fmt="png", page_indices=None):
    """Convert PDF pages to image files.

    Args:
        page_indices: 0-based page indices to render, or None for all pages.

    Returns list of paths for rendered pages.
    """
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        if page_indices is None:
            page_indices = range(total)

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        results = []

        for i in page_indices:
            page = doc[i]
            pix = page.get_pixmap(matrix=matrix)
            img_path = os.path.join(output_dir, f"slide_{i + 1:03d}.{fmt}")
            pix.save(img_path)
            pix = None  # free pixmap memory immediately
            results.append(img_path)
            print(f"  Page {i + 1}/{total} -> {os.path.basename(img_path)}")

    return results


def get_pdf_page_count(pdf_path):
    """Return the number of pages in a PDF without rasterizing."""
    import fitz

    with fitz.open(pdf_path) as doc:
        return len(doc)


def parse_slide_range(spec, total):
    """Parse slide specification like '1-5' or '3,7,10' into 0-based indices."""
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start))
            end = min(total, int(end))
            indices.update(range(start - 1, end))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                indices.add(idx)
    return sorted(indices)


def _get_cached_images(images_dir, dpi, img_fmt, page_indices=None):
    """Check if cached slide images exist at the requested DPI.

    Returns (slide_paths, cached) where cached=True if images were reused.
    """
    info_path = os.path.join(images_dir, "conversion_info.json")

    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if info.get("dpi") == dpi:
            existing = sorted(glob.glob(
                os.path.join(images_dir, f"slide_*.{img_fmt}")
            ))
            if existing:
                if page_indices is not None:
                    # Filter to only requested slides
                    slide_paths = []
                    for p in existing:
                        # Extract slide number from filename (slide_001.png → 0)
                        base = os.path.splitext(os.path.basename(p))[0]
                        num = int(base.split("_")[1]) - 1
                        if num in page_indices:
                            slide_paths.append(p)
                    return slide_paths, True
                return existing, True
        else:
            # DPI changed — remove stale cache
            print(f"  DPI changed ({info['dpi']} -> {dpi}), re-converting...")
            shutil.rmtree(images_dir)

    return None, False


def main():
    parser = argparse.ArgumentParser(
        description="PDF -> Slide Images -> Segments pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_pipeline.py data/pdfs/deck.pdf
  python pdf_pipeline.py /external/path/deck.pdf --dpi 300
  python pdf_pipeline.py deck.pdf --slides 1-5 --upscale 2 --min-area 200
        """,
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=f"PDF render DPI (default: {cfg['pdf_dpi']})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=None,
        help=f"Min segment area in pixels (default: {cfg['min_area']})",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help=f"Max inference dimension (default: {cfg['max_dim']})",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=None,
        help=f"Output upscale factor (default: {cfg['upscale']})",
    )
    parser.add_argument(
        "--slides",
        type=str,
        default=None,
        help="Slide range, e.g. '1-5' or '3,7,10' (default: all)",
    )
    args = parser.parse_args()

    # Resolve parameters — CLI args override config
    dpi = args.dpi if args.dpi is not None else cfg["pdf_dpi"]
    min_area = args.min_area if args.min_area is not None else cfg["min_area"]
    max_dim = args.max_dim if args.max_dim is not None else cfg["max_dim"]
    upscale = args.upscale if args.upscale is not None else cfg["upscale"]
    img_fmt = cfg["pdf_image_format"]

    # Validate PDF path
    pdf_path = os.path.abspath(args.pdf_path)
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Determine output paths ─────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_pdf_dir = os.path.abspath(
        os.path.join(base_dir, cfg["data_dir"], cfg["pdf_subdir"])
    )

    # Check if the PDF lives inside the project's data/pdfs/ directory
    try:
        is_internal = os.path.commonpath([pdf_path, default_pdf_dir]) == default_pdf_dir
    except ValueError:
        is_internal = False  # different drives / mount points

    if is_internal:
        images_base = os.path.join(base_dir, cfg["data_dir"], cfg["images_subdir"])
        segments_base = os.path.join(base_dir, cfg["data_dir"], cfg["segments_subdir"])
    else:
        pdf_parent = os.path.dirname(pdf_path)
        images_base = os.path.join(pdf_parent, "slide_images")
        segments_base = os.path.join(pdf_parent, "slide_image_segments")

    # Slide images dir: cached by pdf_name (no timestamp)
    images_dir = os.path.join(images_base, pdf_name)

    # ── Step 1: Convert PDF to images (with caching) ──────────────────
    print(f"\n{'=' * 60}")
    print(f"PDF Pipeline: {pdf_name}")
    print(f"{'=' * 60}")

    total_pages = get_pdf_page_count(pdf_path)
    page_indices = None
    if args.slides:
        page_indices = parse_slide_range(args.slides, total_pages)

    # Check cache
    slide_paths, cached = _get_cached_images(images_dir, dpi, img_fmt, page_indices)

    if cached:
        print(f"\nStep 1: Reusing {len(slide_paths)} cached slide images (DPI={dpi})")
        print(f"  {images_dir}")
    else:
        os.makedirs(images_dir, exist_ok=True)
        if page_indices is not None:
            print(
                f"\nStep 1: Converting {len(page_indices)} selected slides (DPI={dpi})..."
            )
        else:
            print(f"\nStep 1: Converting PDF to images (DPI={dpi})...")

        t0 = time.time()
        slide_paths = pdf_to_images(
            pdf_path, images_dir, dpi=dpi, fmt=img_fmt, page_indices=page_indices,
        )
        print(f"  {len(slide_paths)} slides saved to {images_dir}")
        print(f"  Conversion time: {time.time() - t0:.1f}s")

        # Write conversion info for future cache hits
        conversion_info = {
            "dpi": dpi,
            "page_count": total_pages,
            "format": img_fmt,
            "converted_at": datetime.now().isoformat(),
            "pdf_name": pdf_name,
        }
        with open(os.path.join(images_dir, "conversion_info.json"), "w") as f:
            json.dump(conversion_info, f, indent=2)

    gc.collect()

    # ── Step 2: Segment each slide ─────────────────────────────────────
    print(f"\nStep 2: Segmenting slides (min_area={min_area}, max_dim={max_dim})...")
    print("  Loading SAM 2.1 model...")

    import torch
    from segmenter import ImageSegmenter
    from PIL import Image
    import numpy as np

    segmenter = ImageSegmenter()

    # Segment run directory: <pdf_name>_<timestamp>/
    run_dir = os.path.join(segments_base, f"{pdf_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    total_segments = 0
    slide_results = []
    workers = min(os.cpu_count() or 4, cfg["render_workers"])

    for slide_idx, slide_path in enumerate(slide_paths):
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        seg_dir = os.path.join(run_dir, slide_name)
        os.makedirs(seg_dir, exist_ok=True)

        print(f"\n  [{slide_idx + 1}/{len(slide_paths)}] {slide_name}")
        t1 = time.time()

        # Run segmentation
        segments = segmenter.segment(
            slide_path, seg_dir, min_area=min_area, max_dim=max_dim
        )

        seg_time = time.time() - t1
        print(f"    {len(segments)} segments detected ({seg_time:.1f}s)")

        # Render full-resolution segments with shared image array
        if segments:
            render_dir = os.path.join(seg_dir, "rendered")
            os.makedirs(render_dir, exist_ok=True)

            # Load slide image once for all segment renders
            with Image.open(slide_path) as img:
                image_array = np.array(img.convert("RGB"))

            # Load meta once
            with open(os.path.join(seg_dir, "masks", "meta.json")) as f:
                meta = json.load(f)

            def _render_one(seg):
                out = os.path.join(render_dir, seg["filename"])
                ImageSegmenter.render_segment(
                    slide_path,
                    seg_dir,
                    seg["index"],
                    meta=meta,
                    upscale=upscale,
                    out_path=out,
                    image_array=image_array,
                )

            t2 = time.time()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_render_one, segments))

            del image_array, meta
            gc.collect()
            render_time = time.time() - t2
            print(
                f"    Rendered {len(segments)} segments at {upscale}x ({render_time:.1f}s)"
            )
            print(f"    -> {render_dir}")

        total_segments += len(segments)
        slide_results.append(
            {
                "slide": slide_name,
                "segments": len(segments),
                "output_dir": seg_dir,
            }
        )

        # Free memory between slides
        if segmenter.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Done! {total_segments} total segments from {len(slide_paths)} slides.")
    print(f"  Slide images: {images_dir}")
    print(f"  Segments:     {run_dir}")
    print(f"{'=' * 60}\n")

    # Save run summary inside the segment run directory
    summary = {
        "pdf": pdf_name,
        "timestamp": timestamp,
        "images_dir": images_dir,
        "settings": {
            "dpi": dpi,
            "min_area": min_area,
            "max_dim": max_dim,
            "upscale": upscale,
        },
        "slides": slide_results,
        "total_segments": total_segments,
    }
    summary_path = os.path.join(run_dir, "run_info.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
