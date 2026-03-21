"""
Core segmentation module using SAM 2.1.
Loads the model once and provides:
  - segment(): detect masks and save compact data + thumbnails
  - render_segment(): produce a full-res RGBA PNG on demand
"""

import gc
import json
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

# Small thumbnails for the gallery preview
THUMB_MAX = 256


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
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
        print("SAM 2.1 model loaded successfully.")

    def _get_device(self):
        """Try MPS (Apple Silicon GPU), fall back to CPU."""
        if torch.backends.mps.is_available():
            try:
                t = torch.zeros(1, device="mps")
                del t
                return torch.device("mps")
            except Exception:
                print("MPS available but not functional, falling back to CPU.")
        return torch.device("cpu")

    def segment(self, image_path, output_dir, min_area=500, max_dim=1536):
        """
        Detect all masks and save compact data for deferred rendering.

        Saves per segment:
          - masks/<idx>.npz: inference-res boolean mask crop
          - thumbs/segment_<idx>.png: small RGBA thumbnail

        Also saves masks/meta.json with scale, orig size, and
        per-segment metadata.

        Returns:
            List of dicts with segment metadata (index, filename,
            area, width, height, predicted_iou).
        """
        os.makedirs(output_dir, exist_ok=True)
        masks_dir = os.path.join(output_dir, "masks")
        thumbs_dir = os.path.join(output_dir, "thumbs")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(thumbs_dir, exist_ok=True)

        original = Image.open(image_path).convert("RGB")
        orig_w, orig_h = original.size

        # Resize for inference if needed
        scale = 1.0
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            inference_image = np.array(original.resize((new_w, new_h), Image.LANCZOS))
            thumb_source = inference_image
        else:
            inference_image = np.array(original)
            thumb_source = inference_image

        original.close()
        del original

        with torch.inference_mode():
            all_masks = self.mask_generator.generate(inference_image)

        # Free GPU memory immediately
        if self.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

        # Filter by area
        inv_scale_sq = 1.0 / (scale * scale) if scale != 1.0 else 1.0
        masks = []
        for m in all_masks:
            if m["area"] * inv_scale_sq >= min_area:
                masks.append(m)
            else:
                del m["segmentation"]
        del all_masks

        masks.sort(key=lambda m: m["area"], reverse=True)

        for m in masks[200:]:
            del m["segmentation"]
        masks = masks[:200]

        results = []
        for idx, mask_data in enumerate(masks):
            seg_mask = mask_data["segmentation"]
            bbox = mask_data["bbox"]

            # Compute bbox at inference resolution for mask crop
            inf_x = max(0, int(bbox[0]))
            inf_y = max(0, int(bbox[1]))
            inf_w = int(bbox[2])
            inf_h = int(bbox[3])
            inf_r = min(inf_x + inf_w, seg_mask.shape[1])
            inf_b = min(inf_y + inf_h, seg_mask.shape[0])

            if inf_r <= inf_x or inf_b <= inf_y:
                del mask_data["segmentation"]
                continue

            # Crop mask at inference resolution (compact)
            mask_crop = seg_mask[inf_y:inf_b, inf_x:inf_r]

            # Compute full-res bbox
            if scale != 1.0:
                fx = max(0, int(bbox[0] / scale))
                fy = max(0, int(bbox[1] / scale))
                fw = min(int(bbox[2] / scale), orig_w - fx)
                fh = min(int(bbox[3] / scale), orig_h - fy)
            else:
                fx, fy, fw, fh = inf_x, inf_y, inf_w, inf_h
                fw = min(fw, orig_w - fx)
                fh = min(fh, orig_h - fy)

            if fw <= 0 or fh <= 0:
                del mask_data["segmentation"]
                continue

            actual_area = int(np.count_nonzero(mask_crop) * inv_scale_sq)

            # Save compact mask crop
            np.savez_compressed(
                os.path.join(masks_dir, f"{idx}.npz"),
                mask=np.packbits(mask_crop),
                shape=np.array(mask_crop.shape, dtype=np.int32),
            )

            # Generate small thumbnail from inference-res data
            thumb_rgb = thumb_source[inf_y:inf_b, inf_x:inf_r]
            th, tw = thumb_rgb.shape[:2]
            thumb_rgba = np.empty((th, tw, 4), dtype=np.uint8)
            thumb_rgba[:, :, :3] = thumb_rgb
            thumb_rgba[:, :, 3] = mask_crop.astype(np.uint8) * 255

            thumb_img = Image.fromarray(thumb_rgba, "RGBA")
            del thumb_rgba

            # Resize thumbnail
            t_scale = min(THUMB_MAX / tw, THUMB_MAX / th, 1.0)
            if t_scale < 1.0:
                thumb_img = thumb_img.resize(
                    (max(1, int(tw * t_scale)), max(1, int(th * t_scale))),
                    Image.LANCZOS,
                )

            thumb_name = f"segment_{idx:03d}.png"
            thumb_img.save(os.path.join(thumbs_dir, thumb_name))
            thumb_img.close()

            del mask_data["segmentation"]
            del mask_crop

            results.append(
                {
                    "index": idx,
                    "filename": thumb_name,
                    "area": actual_area,
                    "width": fw,
                    "height": fh,
                    "predicted_iou": round(mask_data.get("predicted_iou", 0), 3),
                    "bbox_orig": [fx, fy, fw, fh],
                    "bbox_inf": [inf_x, inf_y, inf_r - inf_x, inf_b - inf_y],
                }
            )

        del thumb_source, inference_image
        gc.collect()

        # Persist metadata for deferred rendering
        meta = {
            "scale": scale,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "segments": results,
        }
        with open(os.path.join(masks_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

        if self.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

        return results

    @staticmethod
    def render_segment(image_path, output_dir, index, meta=None):
        """
        Render a single full-resolution RGBA PNG on demand.

        Pass meta (already-loaded dict) to avoid re-reading meta.json.
        Returns the path to the rendered file, or None on failure.
        """
        masks_dir = os.path.join(output_dir, "masks")

        if meta is None:
            with open(os.path.join(masks_dir, "meta.json")) as f:
                meta = json.load(f)

        seg_info = next((s for s in meta["segments"] if s["index"] == index), None)
        if seg_info is None:
            return None

        scale = meta["scale"]
        fx, fy, fw, fh = seg_info["bbox_orig"]
        inf_x, inf_y, inf_w, inf_h = seg_info["bbox_inf"]

        # Load compact mask
        npz = np.load(os.path.join(masks_dir, f"{index}.npz"))
        shape = tuple(npz["shape"])
        mask_crop = (
            np.unpackbits(npz["mask"])[: shape[0] * shape[1]]
            .reshape(shape)
            .astype(bool)
        )

        # Upscale mask to full resolution if needed
        if scale != 1.0:
            mask_pil = Image.fromarray(mask_crop.astype(np.uint8) * 255, "L")
            mask_pil = mask_pil.resize((fw, fh), Image.NEAREST)
            full_mask = np.array(mask_pil) > 127
            mask_pil.close()
            del mask_crop
        else:
            full_mask = mask_crop

        # Read only the crop region from the original image
        original = Image.open(image_path).convert("RGB")
        cropped_rgb = np.array(original.crop((fx, fy, fx + fw, fy + fh)))
        original.close()

        # Build RGBA
        rgba = np.empty((fh, fw, 4), dtype=np.uint8)
        rgba[:, :, :3] = cropped_rgb
        rgba[:, :, 3] = full_mask.astype(np.uint8) * 255
        del cropped_rgb, full_mask

        segment_img = Image.fromarray(rgba, "RGBA")
        del rgba

        # Edge feathering
        alpha = segment_img.split()[3]
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.5))
        segment_img.putalpha(alpha)

        filename = f"segment_{index:03d}.png"
        out_path = os.path.join(output_dir, filename)
        segment_img.save(out_path)
        segment_img.close()

        return out_path
