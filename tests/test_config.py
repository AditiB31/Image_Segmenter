"""Tests for configuration loading."""

import os
import tempfile

import pytest
import yaml

# Import the config module's internals
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, _DEFAULTS


class TestConfigDefaults:
    """Test that built-in defaults are correct."""

    def test_defaults_exist(self):
        """All expected keys are present in defaults."""
        required_keys = [
            "min_area", "max_dim", "max_segments", "upscale",
            "points_per_side", "pred_iou_thresh", "stability_score_thresh",
            "feather_min_px", "feather_max_px", "feather_factor",
            "morph_kernel_size", "contour_sigma_min", "contour_sigma_max",
            "feather_blur_kernel", "feather_blur_sigma",
            "mask_upscale_sharpness", "tight_crop", "tight_crop_padding",
            "pdf_dpi", "render_workers", "max_upload_mb", "session_ttl",
        ]
        for key in required_keys:
            assert key in _DEFAULTS, f"Missing default key: {key}"

    def test_edge_feathering_defaults(self):
        """Edge feathering defaults use optimised values."""
        assert _DEFAULTS["feather_min_px"] == 1.5
        assert _DEFAULTS["feather_max_px"] == 4.0
        assert _DEFAULTS["feather_factor"] == 0.012

    def test_edge_processing_defaults(self):
        """Edge processing defaults use optimised values."""
        assert _DEFAULTS["contour_sigma_min"] == 2.0
        assert _DEFAULTS["contour_sigma_divisor"] == 150.0
        assert _DEFAULTS["feather_blur_kernel"] == 5
        assert _DEFAULTS["feather_blur_sigma"] == 0.6
        assert _DEFAULTS["mask_upscale_sharpness"] == 0.06

    def test_feather_min_less_than_max(self):
        """feather_min_px should be less than feather_max_px."""
        assert _DEFAULTS["feather_min_px"] < _DEFAULTS["feather_max_px"]

    def test_blur_kernel_is_odd(self):
        """Blur kernel must be odd for OpenCV."""
        assert _DEFAULTS["feather_blur_kernel"] % 2 == 1
        assert _DEFAULTS["morph_kernel_size"] % 2 == 1

    def test_numeric_ranges(self):
        """Ensure config values are within sensible ranges."""
        assert 0 < _DEFAULTS["min_area"] <= 50000
        assert 1024 <= _DEFAULTS["max_dim"] <= 4096
        assert 1 <= _DEFAULTS["max_segments"] <= 500
        assert 0.5 <= _DEFAULTS["pred_iou_thresh"] <= 1.0
        assert 0.5 <= _DEFAULTS["stability_score_thresh"] <= 1.0
        assert 1 <= _DEFAULTS["max_upload_mb"] <= 200


class TestConfigLoading:
    """Test YAML config loading and merging."""

    def test_load_with_missing_file(self):
        """Loading from nonexistent path should return defaults."""
        cfg = load_config("/nonexistent/path/config.yaml")
        assert cfg == _DEFAULTS

    def test_load_overrides_defaults(self):
        """YAML values should override defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"min_area": 999, "max_dim": 1024}, f)
            f.flush()
            try:
                cfg = load_config(f.name)
                assert cfg["min_area"] == 999
                assert cfg["max_dim"] == 1024
                # Non-overridden values should remain defaults
                assert cfg["max_segments"] == _DEFAULTS["max_segments"]
            finally:
                os.unlink(f.name)

    def test_load_empty_yaml(self):
        """Empty YAML file should return defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            try:
                cfg = load_config(f.name)
                assert cfg == _DEFAULTS
            finally:
                os.unlink(f.name)

    def test_load_actual_config(self):
        """Loading the project's config.yaml should work."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )
        if os.path.exists(config_path):
            cfg = load_config(config_path)
            assert isinstance(cfg, dict)
            # Verify the edge optimisations are applied
            assert cfg["feather_min_px"] == 1.5
            assert cfg["feather_max_px"] == 4.0
            assert cfg["mask_upscale_sharpness"] == 0.06


class TestConfigConsistency:
    """Test that config.yaml and config.py defaults are consistent."""

    def test_yaml_matches_python_defaults(self):
        """Values in config.yaml should match updated Python defaults."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")

        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f)

        # Check edge-related values match between YAML and Python defaults
        edge_keys = [
            "feather_min_px", "feather_max_px", "feather_factor",
            "contour_sigma_min", "contour_sigma_divisor",
            "feather_blur_kernel", "feather_blur_sigma",
            "mask_upscale_sharpness",
        ]
        for key in edge_keys:
            assert yaml_cfg[key] == _DEFAULTS[key], (
                f"Mismatch for {key}: YAML={yaml_cfg[key]}, "
                f"Python={_DEFAULTS[key]}"
            )
