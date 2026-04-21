"""
Configuration loader for Image Segmenter.

Reads config.yaml from the project root and exposes a module-level `cfg` dict.
Every key has a built-in default so the app works even without the YAML file.
"""

import logging
import os

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE_DIR, "config.yaml")

_DEFAULTS = {
    # Segmentation
    "min_area": 6000,
    "min_color_std": 15.0,
    "min_saturation": 25,
    "max_dim": 2048,
    "max_segments": 50,
    "upscale": 1,
    # SAM 2.1 model
    "points_per_side": 32,
    "pred_iou_thresh": 0.84,
    "stability_score_thresh": 0.90,
    "min_mask_region_area": 500,
    "min_fill_ratio": 0.45,
    "max_aspect_ratio": 3.5,
    "iou_dedup_thresh": 0.8,
    "containment_thresh": 0.7,
    # Thumbnails
    "thumb_max": 512,
    # Edge feathering (balanced defaults for smooth edges)
    "feather_min_px": 1.5,
    "feather_max_px": 4.0,
    "feather_factor": 0.012,
    # Edge processing (advanced)
    "morph_kernel_size": 3,
    "contour_sigma_min": 2.0,
    "contour_sigma_max": 18.0,
    "contour_sigma_divisor": 150.0,
    "feather_blur_kernel": 5,
    "feather_blur_sigma": 0.6,
    "mask_upscale_sharpness": 0.06,
    "dist_transform_mask_size": 5,
    # Tight cropping
    "tight_crop": True,
    "tight_crop_padding": 4,
    # PDF pipeline
    "pdf_dpi": 200,
    "pdf_image_format": "png",
    "render_workers": 2,
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


def load_config(path: str = _CONFIG_PATH) -> dict:
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
        except Exception as exc:
            logger.warning("Failed to parse %s: %s — using defaults", path, exc)
    return config


cfg = load_config()
