"""
Core segmentation module using SAM 2.1.
Loads the model once and provides a segment() method to extract all objects
from an image as transparent PNGs.
"""

import os
import warnings

import numpy as np
import torch
from PIL import Image, ImageFilter

# Suppress the expected _C import warning when CUDA extensions aren't built
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "sam2.1_hiera_base_plus.pt",
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

    def segment(self, image_path, output_dir, min_area=500, max_dim=4096):
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

        # PIL image no longer needed — pixel data is in original_np
        original.close()
        del original

        with torch.inference_mode():
            all_masks = self.mask_generator.generate(inference_image)

        # Free inference image early when it's a separate allocation
        if scale != 1.0:
            del inference_image

        # Filter by area, eagerly freeing rejected masks' segmentation arrays
        inv_scale_sq = 1.0 / (scale * scale) if scale != 1.0 else 1.0
        masks = []
        for m in all_masks:
            if m["area"] * inv_scale_sq >= min_area:
                masks.append(m)
            else:
                del m["segmentation"]
        del all_masks

        masks.sort(key=lambda m: m["area"], reverse=True)

        # Free excess masks beyond the cap
        for m in masks[200:]:
            del m["segmentation"]
        masks = masks[:200]

        results = []
        for idx, mask_data in enumerate(masks):
            seg_mask = mask_data["segmentation"]
            bbox = mask_data["bbox"]

            if scale != 1.0:
                # Scale bbox to original resolution
                x = int(bbox[0] / scale)
                y = int(bbox[1] / scale)
                w = int(bbox[2] / scale)
                h = int(bbox[3] / scale)

                # Clamp to image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, orig_w - x)
                h = min(h, orig_h - y)
                if w <= 0 or h <= 0:
                    del mask_data["segmentation"]
                    continue

                # Upscale only the bbox region of the mask, not the full image.
                # Map the clamped original-res bbox back to inference coords
                # to extract just the relevant patch.
                inf_x = max(0, int(x * scale))
                inf_y = max(0, int(y * scale))
                inf_r = min(int((x + w) * scale) + 1, seg_mask.shape[1])
                inf_b = min(int((y + h) * scale) + 1, seg_mask.shape[0])
                cropped_mask_small = seg_mask[inf_y:inf_b, inf_x:inf_r]

                mask_pil = Image.fromarray(
                    cropped_mask_small.astype(np.uint8) * 255, "L"
                )
                mask_pil = mask_pil.resize((w, h), Image.NEAREST)
                cropped_mask = np.array(mask_pil) > 127
                mask_pil.close()
                del cropped_mask_small
            else:
                x, y, w, h = (int(v) for v in bbox)

                # Clamp to image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, orig_w - x)
                h = min(h, orig_h - y)
                if w <= 0 or h <= 0:
                    del mask_data["segmentation"]
                    continue

                cropped_mask = seg_mask[y : y + h, x : x + w]

            # Free the full segmentation array now that we have the crop
            del mask_data["segmentation"]

            cropped_rgb = original_np[y : y + h, x : x + w]

            # Build RGBA directly: full-resolution crop + alpha from mask
            rgba = np.empty((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = cropped_rgb
            rgba[:, :, 3] = cropped_mask.astype(np.uint8) * 255
            actual_area = int(np.count_nonzero(cropped_mask))
            del cropped_mask

            segment_img = Image.fromarray(rgba, "RGBA")
            del rgba

            # Light edge feathering for cleaner compositing in slides
            alpha_channel = segment_img.split()[3]
            alpha_channel = alpha_channel.filter(ImageFilter.GaussianBlur(radius=0.5))
            segment_img.putalpha(alpha_channel)

            filename = f"segment_{idx:03d}.png"
            segment_img.save(os.path.join(output_dir, filename))
            segment_img.close()

            results.append(
                {
                    "index": idx,
                    "filename": filename,
                    "area": actual_area,
                    "width": w,
                    "height": h,
                    "predicted_iou": round(mask_data.get("predicted_iou", 0), 3),
                }
            )

        if self.device.type == "mps":
            torch.mps.empty_cache()

        return results
