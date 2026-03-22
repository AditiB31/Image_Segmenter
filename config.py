"""
Configuration loader for Image Segmenter.

Reads config.yaml from the project root and exposes a module-level `cfg` dict.
Every key has a built-in default so the app works even without the YAML file.
"""

import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE_DIR, "config.yaml")

_DEFAULTS = {
    # Segmentation
    "min_area": 500,
    "max_dim": 3072,
    "max_segments": 200,
    "upscale": 1,
    # SAM 2.1 model
    "points_per_side": 32,
    "pred_iou_thresh": 0.84,
    "stability_score_thresh": 0.90,
    "min_mask_region_area": 100,
    "iou_dedup_thresh": 0.9,
    # Thumbnails
    "thumb_max": 512,
    # Edge feathering
    "feather_min_px": 2.0,
    "feather_max_px": 5.0,
    "feather_factor": 0.012,
    # Edge processing (advanced)
    "morph_kernel_size": 3,
    "contour_sigma_min": 2.5,
    "contour_sigma_max": 18.0,
    "contour_sigma_divisor": 180.0,
    "feather_blur_kernel": 5,
    "feather_blur_sigma": 0.8,
    "dist_transform_mask_size": 5,
    # Tight cropping
    "tight_crop": True,
    "tight_crop_padding": 4,
    # PDF pipeline
    "pdf_dpi": 200,
    "pdf_image_format": "png",
    "render_workers": 4,
    # Paths
    "data_dir": "data",
    "pdf_subdir": "pdfs",
    "images_subdir": "images",
    "segments_subdir": "segments",
    # Safety
    "max_upscale": 4,
    # Web server
    "max_upload_mb": 50,
    "session_ttl": 3600,
    "host": "127.0.0.1",
    "port": 5000,
}


def load_config(path=_CONFIG_PATH):
    """Load configuration from YAML, falling back to built-in defaults."""
    config = dict(_DEFAULTS)
    if os.path.exists(path):
        try:
            import yaml

            with open(path) as f:
                user = yaml.safe_load(f) or {}
            config.update(user)
        except ImportError:
            pass  # pyyaml not installed — use defaults silently
    return config


cfg = load_config()
