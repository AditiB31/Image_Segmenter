"""
Core segmentation module using SAM 2.1.
Loads the model once and provides a segment() method to extract all objects
from an image as transparent PNGs.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints", "sam2.1_hiera_base_plus.pt"
)
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


class ImageSegmenter:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH, model_cfg=MODEL_CFG):
        self.device = self._get_device()
        print(f"Loading SAM 2.1 on device: {self.device}")

        sam2_model = build_sam2(
            model_cfg,
            checkpoint_path,
            device=self.device,
        )

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
        print("SAM 2.1 model loaded successfully.")

    def _get_device(self):
        """Try MPS (Apple Silicon GPU), fall back to CPU."""
        if torch.backends.mps.is_available():
            try:
                # Quick test to verify MPS actually works
                t = torch.zeros(1, device="mps")
                del t
                return torch.device("mps")
            except Exception:
                print("MPS available but not functional, falling back to CPU.")
        return torch.device("cpu")

    def segment(self, image_path, output_dir, min_area=500, max_dim=2048):
        """
        Segment all objects in an image and save each as a transparent PNG.

        Args:
            image_path: Path to the input image.
            output_dir: Directory to save segment PNGs.
            min_area: Minimum mask area in pixels to keep.
            max_dim: Max dimension for inference (larger images are resized).

        Returns:
            List of dicts with segment metadata.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load original image at full resolution
        original = Image.open(image_path).convert("RGB")
        orig_w, orig_h = original.size
        original_np = np.array(original)

        # Resize for inference if needed
        scale = 1.0
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            inference_image = np.array(original.resize((new_w, new_h), Image.LANCZOS))
        else:
            inference_image = original_np

        # Generate masks
        masks = self.mask_generator.generate(inference_image)

        # Filter by area (scaled back to original resolution)
        area_scale = 1.0 / (scale * scale) if scale != 1.0 else 1.0
        masks = [m for m in masks if m["area"] * area_scale >= min_area]

        # Sort by area descending (largest segments first)
        masks.sort(key=lambda m: m["area"], reverse=True)

        # Cap at 200 segments
        masks = masks[:200]

        results = []
        for idx, mask_data in enumerate(masks):
            seg_mask = mask_data["segmentation"]  # bool array at inference resolution
            bbox = mask_data["bbox"]  # [x, y, w, h] at inference resolution

            if scale != 1.0:
                # Scale mask back to original resolution
                seg_mask = cv2.resize(
                    seg_mask.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                # Scale bbox
                x, y, w, h = bbox
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
            else:
                x, y, w, h = [int(v) for v in bbox]

            # Clamp bbox to image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, orig_w - x)
            h = min(h, orig_h - y)

            if w <= 0 or h <= 0:
                continue

            # Crop image and mask to bounding box
            cropped_rgb = original_np[y:y + h, x:x + w]
            cropped_mask = seg_mask[y:y + h, x:x + w]

            # Create RGBA image
            alpha = (cropped_mask * 255).astype(np.uint8)
            rgba = np.dstack([cropped_rgb, alpha])
            segment_img = Image.fromarray(rgba, "RGBA")

            # Light edge feathering for cleaner compositing
            alpha_channel = segment_img.split()[3]
            alpha_channel = alpha_channel.filter(ImageFilter.GaussianBlur(radius=0.5))
            segment_img.putalpha(alpha_channel)

            # Save
            filename = f"segment_{idx:03d}.png"
            segment_img.save(os.path.join(output_dir, filename))

            actual_area = int(np.sum(cropped_mask))
            results.append({
                "index": idx,
                "filename": filename,
                "area": actual_area,
                "width": w,
                "height": h,
                "predicted_iou": round(mask_data.get("predicted_iou", 0), 3),
            })

        # Clean up MPS cache
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return results
