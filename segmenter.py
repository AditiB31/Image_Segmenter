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

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from PIL import Image

from config import cfg

# Suppress the expected _C import warning when CUDA extensions aren't built
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

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
            points_per_side=cfg["points_per_side"],
            pred_iou_thresh=cfg["pred_iou_thresh"],
            stability_score_thresh=cfg["stability_score_thresh"],
            min_mask_region_area=cfg["min_mask_region_area"],
        )
        self.predictor = SAM2ImagePredictor(sam2_model)
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

    def _deduplicate_masks(self, masks, iou_thresh=None, containment_thresh=None):
        """Remove near-duplicate and sub-component masks.

        A mask is removed if:
          - IoU with a kept mask > iou_thresh (near-duplicate), OR
          - its overlap with a kept mask > containment_thresh of its own area
            (sub-component of a larger object).

        Masks must be sorted largest-first. Uses spatial grid bucketing for
        O(n*k) average performance.
        """
        if iou_thresh is None:
            iou_thresh = cfg["iou_dedup_thresh"]
        if containment_thresh is None:
            containment_thresh = cfg.get("containment_thresh", 0)
        if len(masks) <= 1:
            return masks

        # Spatial grid: divide image into cells, index kept masks by cell
        cell_size = 64
        grid = {}  # (cx, cy) -> list of indices into `keep`

        def _bbox_cells(bbox):
            """Yield grid cells that this bbox overlaps."""
            x0, y0 = int(bbox[0]) // cell_size, int(bbox[1]) // cell_size
            x1 = int(bbox[0] + bbox[2]) // cell_size
            y1 = int(bbox[1] + bbox[3]) // cell_size
            for gx in range(x0, x1 + 1):
                for gy in range(y0, y1 + 1):
                    yield (gx, gy)

        keep = []
        for mask in masks:
            seg = mask["segmentation"]
            bbox = mask["bbox"]
            is_dup = False

            # Only check masks in overlapping grid cells
            candidate_indices = set()
            for cell in _bbox_cells(bbox):
                if cell in grid:
                    candidate_indices.update(grid[cell])

            for ki in candidate_indices:
                kept = keep[ki]
                kbbox = kept["bbox"]
                ix = max(bbox[0], kbbox[0])
                iy = max(bbox[1], kbbox[1])
                ir = min(bbox[0] + bbox[2], kbbox[0] + kbbox[2])
                ib = min(bbox[1] + bbox[3], kbbox[1] + kbbox[3])
                if ir <= ix or ib <= iy:
                    continue
                bbox_overlap = (ir - ix) * (ib - iy)
                if bbox_overlap / (bbox[2] * bbox[3] + 1e-6) < 0.3:
                    continue
                r0, r1, c0, c1 = int(iy), int(ib), int(ix), int(ir)
                inter = np.logical_and(
                    seg[r0:r1, c0:c1], kept["segmentation"][r0:r1, c0:c1]
                ).sum()
                union = mask["area"] + kept["area"] - inter
                if union > 0 and inter / union > iou_thresh:
                    is_dup = True
                    break
                # Containment check: discard if this mask is mostly inside a larger one
                if containment_thresh > 0 and mask["area"] > 0:
                    if inter / mask["area"] > containment_thresh:
                        is_dup = True
                        break
            if not is_dup:
                idx = len(keep)
                keep.append(mask)
                for cell in _bbox_cells(bbox):
                    grid.setdefault(cell, []).append(idx)
        return keep

    def _load_and_resize(self, image_path, max_dim):
        """Load an image and resize for inference if needed.

        Returns (inference_image, orig_w, orig_h, scale).
        """
        with Image.open(image_path) as original:
            original = original.convert("RGB")
            orig_w, orig_h = original.size

            scale = 1.0
            if max(orig_w, orig_h) > max_dim:
                scale = max_dim / max(orig_w, orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                inference_image = np.array(original.resize((new_w, new_h), Image.LANCZOS))
            else:
                inference_image = np.array(original)
        return inference_image, orig_w, orig_h, scale

    def _run_with_autocast(self, fn):
        """Run fn() with MPS float16 autocast if available, falling back to float32."""
        with torch.inference_mode():
            if self.device.type == "mps":
                try:
                    with torch.autocast("mps", dtype=torch.float16):
                        return fn()
                except RuntimeError:
                    return fn()
            else:
                return fn()

    def _make_thumbnail(self, rgb_crop, mask_crop, thumbs_dir, idx, thumb_max):
        """Generate and save a thumbnail from an RGB crop and mask crop."""
        th, tw = rgb_crop.shape[:2]
        thumb_rgba = np.empty((th, tw, 4), dtype=np.uint8)
        thumb_rgba[:, :, :3] = rgb_crop
        thumb_rgba[:, :, 3] = mask_crop.astype(np.uint8) * 255

        thumb_img = Image.fromarray(thumb_rgba, "RGBA")
        del thumb_rgba

        t_scale = min(thumb_max / tw, thumb_max / th, 1.0)
        if t_scale < 1.0:
            thumb_img = thumb_img.resize(
                (max(1, int(tw * t_scale)), max(1, int(th * t_scale))),
                Image.BILINEAR,
            )

        thumb_name = f"segment_{idx:03d}.png"
        thumb_img.save(os.path.join(thumbs_dir, thumb_name))
        thumb_img.close()
        return thumb_name

    @staticmethod
    def _compute_full_bbox(bbox, scale, orig_w, orig_h):
        """Convert inference-resolution bbox to full-resolution coordinates.

        Returns (fx, fy, fw, fh) or None if degenerate.
        """
        inf_x = max(0, int(bbox[0]))
        inf_y = max(0, int(bbox[1]))
        inf_w = int(bbox[2])
        inf_h = int(bbox[3])

        if scale != 1.0:
            fx = max(0, round(inf_x / scale))
            fy = max(0, round(inf_y / scale))
            fw = min(round(inf_w / scale), orig_w - fx)
            fh = min(round(inf_h / scale), orig_h - fy)
        else:
            fx, fy, fw, fh = inf_x, inf_y, inf_w, inf_h
            fw = min(fw, orig_w - fx)
            fh = min(fh, orig_h - fy)

        if fw <= 0 or fh <= 0:
            return None
        return (fx, fy, fw, fh)

    @staticmethod
    def _save_meta(masks_dir, meta):
        """Atomically write meta.json via tmp + rename."""
        meta_path = os.path.join(masks_dir, "meta.json")
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(meta, f)
        os.replace(tmp_path, meta_path)

    def segment(self, image_path, output_dir, min_area=None, max_dim=None):
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
        if min_area is None:
            min_area = cfg["min_area"]
        if max_dim is None:
            max_dim = cfg["max_dim"]
        max_segments = cfg["max_segments"]
        thumb_max = cfg["thumb_max"]

        os.makedirs(output_dir, exist_ok=True)
        masks_dir = os.path.join(output_dir, "masks")
        thumbs_dir = os.path.join(output_dir, "thumbs")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(thumbs_dir, exist_ok=True)

        inference_image, orig_w, orig_h, scale = self._load_and_resize(image_path, max_dim)
        thumb_source = inference_image

        all_masks = self._run_with_autocast(
            lambda: self.mask_generator.generate(inference_image)
        )

        # Free GPU memory immediately
        if self.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

        # Filter by area and aspect ratio
        inv_scale_sq = 1.0 / (scale * scale) if scale != 1.0 else 1.0
        max_aspect = cfg.get("max_aspect_ratio", 0)
        min_fill = cfg.get("min_fill_ratio", 0)
        masks = []
        for m in all_masks:
            if m["area"] * inv_scale_sq < min_area:
                del m["segmentation"]
                continue
            bw, bh = m["bbox"][2], m["bbox"][3]
            if bw <= 0 or bh <= 0:
                del m["segmentation"]
                continue
            # Reject excessively elongated segments (grid lines, borders)
            if max_aspect > 0:
                ratio = max(bw, bh) / min(bw, bh)
                if ratio > max_aspect:
                    del m["segmentation"]
                    continue
            # Reject low fill ratio segments (text, thin shapes)
            fill = m["area"] / (bw * bh) if (bw * bh) > 0 else 0
            if min_fill > 0 and fill < min_fill:
                del m["segmentation"]
                continue
            # Small segments with very high fill are almost certainly text
            # (e.g. "CME", "OK"). Large solid objects are fine.
            if m["area"] < 20000 and fill > 0.90:
                del m["segmentation"]
                continue
            masks.append(m)
        del all_masks

        # Color-based filters using inference image data.
        # 1. Near-white background: bright (mean > 240) + uniform (std < threshold)
        # 2. Small text detection: small segments with very low saturation
        #    are almost always dark text on a light slide. Icons/graphics
        #    tend to have colour (saturation > 0).
        min_color_std = cfg.get("min_color_std", 0)
        min_sat = cfg.get("min_saturation", 0)
        small_seg_area = 15000  # only apply saturation check to small segments
        if min_color_std > 0 or min_sat > 0:
            filtered = []
            for m in masks:
                seg = m["segmentation"]
                bbox = m["bbox"]
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                crop = inference_image[y:y+h, x:x+w]
                mask_crop = seg[y:y+h, x:x+w]
                pixels = crop[mask_crop]
                if len(pixels) > 0:
                    mean_val = float(np.mean(pixels))
                    std_val = float(np.std(pixels))
                    # Reject near-white flat backgrounds
                    if min_color_std > 0 and mean_val > 240 and std_val < min_color_std:
                        del m["segmentation"]
                        continue
                    # Reject small low-saturation segments (dark text)
                    if min_sat > 0 and m["area"] * inv_scale_sq < small_seg_area:
                        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                        sat_pixels = hsv_crop[:, :, 1][mask_crop]
                        if float(np.mean(sat_pixels)) < min_sat:
                            del m["segmentation"]
                            continue
                filtered.append(m)
            masks = filtered

        masks.sort(key=lambda m: m["area"], reverse=True)

        for m in masks[max_segments:]:
            del m["segmentation"]
        masks = masks[:max_segments]

        # SAM can produce near-identical masks from overlapping point prompts
        masks = self._deduplicate_masks(masks)

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

            mask_crop = seg_mask[inf_y:inf_b, inf_x:inf_r]

            full_bbox = self._compute_full_bbox(bbox, scale, orig_w, orig_h)
            if full_bbox is None:
                del mask_data["segmentation"]
                continue
            fx, fy, fw, fh = full_bbox

            actual_area = int(np.count_nonzero(mask_crop) * inv_scale_sq)

            np.savez_compressed(
                os.path.join(masks_dir, f"{idx}.npz"),
                mask=np.packbits(mask_crop),
                shape=np.array(mask_crop.shape, dtype=np.int32),
            )

            thumb_rgb = thumb_source[inf_y:inf_b, inf_x:inf_r]
            thumb_name = self._make_thumbnail(thumb_rgb, mask_crop, thumbs_dir, idx, thumb_max)

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

        self._save_meta(masks_dir, {
            "scale": scale,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "settings": {"min_area": min_area, "max_dim": max_dim},
            "segments": results,
        })

        if self.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

        return results

    def predict_with_prompts(self, image_path, output_dir, prompts, max_dim=None):
        """
        Segment objects using point/box/contour prompts via SAM2ImagePredictor.

        Args:
            image_path: Path to the image file
            output_dir: Directory to save masks and thumbnails
            prompts: List of dicts, each with optional keys:
                - points: [[x,y], ...] in original image coordinates
                - labels: [1/0, ...] (1=foreground, 0=background)
                - contour: [[x,y], ...] polygon vertices in original coords
                - box: [x1, y1, x2, y2] in original coords
            max_dim: Max image dimension for inference (default from config)

        Returns: List of segment metadata dicts
        """
        if max_dim is None:
            max_dim = cfg["max_dim"]
        thumb_max = cfg["thumb_max"]

        os.makedirs(output_dir, exist_ok=True)
        masks_dir = os.path.join(output_dir, "masks")
        thumbs_dir = os.path.join(output_dir, "thumbs")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(thumbs_dir, exist_ok=True)

        # Load existing segments to determine starting index
        meta_path = os.path.join(masks_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                existing_meta = json.load(f)
            existing_segments = existing_meta.get("segments", [])
            start_idx = max((s["index"] for s in existing_segments), default=-1) + 1
        else:
            existing_segments = []
            start_idx = 0

        inference_image, orig_w, orig_h, scale = self._load_and_resize(image_path, max_dim)

        # Set image on predictor (computes embeddings once)
        self._run_with_autocast(lambda: self.predictor.set_image(inference_image))

        results = []

        for prompt_offset, prompt in enumerate(prompts):
            idx = start_idx + prompt_offset
            predict_kwargs = {"multimask_output": True}
            point_coords = []
            point_labels = []

            # Collect explicit point prompts
            if prompt.get("points"):
                for pt, lbl in zip(prompt["points"], prompt["labels"]):
                    point_coords.append([pt[0] * scale, pt[1] * scale])
                    point_labels.append(lbl)

            # Handle contour: bounding box + sampled foreground points
            if prompt.get("contour") and len(prompt["contour"]) >= 3:
                contour_pts = prompt["contour"]
                xs = [p[0] for p in contour_pts]
                ys = [p[1] for p in contour_pts]
                predict_kwargs["box"] = np.array(
                    [min(xs) * scale, min(ys) * scale,
                     max(xs) * scale, max(ys) * scale],
                    dtype=np.float32,
                )
                # Sample up to 10 points along the contour
                n_samples = min(len(contour_pts), 10)
                step = max(1, len(contour_pts) // n_samples)
                for i in range(0, len(contour_pts), step):
                    pt = contour_pts[i]
                    point_coords.append([pt[0] * scale, pt[1] * scale])
                    point_labels.append(1)

            # Handle explicit box prompt
            if prompt.get("box") and "box" not in predict_kwargs:
                b = prompt["box"]
                predict_kwargs["box"] = np.array(
                    [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale],
                    dtype=np.float32,
                )

            if point_coords:
                predict_kwargs["point_coords"] = np.array(
                    point_coords, dtype=np.float32
                )
                predict_kwargs["point_labels"] = np.array(
                    point_labels, dtype=np.int32
                )

            if len(predict_kwargs) <= 1:
                continue  # no valid prompts

            masks, scores, _ = self._run_with_autocast(
                lambda: self.predictor.predict(**predict_kwargs)
            )

            # Take best mask (highest confidence)
            best = int(np.argmax(scores))
            mask = masks[best] > 0.5

            if not np.any(mask):
                continue

            # Compute bounding box of mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            inf_x, inf_y = int(cmin), int(rmin)
            inf_w = int(cmax - cmin + 1)
            inf_h = int(rmax - rmin + 1)

            full_bbox = self._compute_full_bbox(
                [inf_x, inf_y, inf_w, inf_h], scale, orig_w, orig_h
            )
            if full_bbox is None:
                continue
            fx, fy, fw, fh = full_bbox

            mask_crop = mask[inf_y:inf_y + inf_h, inf_x:inf_x + inf_w]
            inv_scale_sq = 1.0 / (scale * scale) if scale != 1.0 else 1.0
            actual_area = int(np.count_nonzero(mask_crop) * inv_scale_sq)

            np.savez_compressed(
                os.path.join(masks_dir, f"{idx}.npz"),
                mask=np.packbits(mask_crop),
                shape=np.array(mask_crop.shape, dtype=np.int32),
            )

            thumb_rgb = inference_image[inf_y:inf_y + inf_h, inf_x:inf_x + inf_w]
            thumb_name = self._make_thumbnail(thumb_rgb, mask_crop, thumbs_dir, idx, thumb_max)

            del mask_crop

            results.append({
                "index": idx,
                "filename": thumb_name,
                "area": actual_area,
                "width": fw,
                "height": fh,
                "predicted_iou": round(float(scores[best]), 3),
                "bbox_orig": [fx, fy, fw, fh],
                "bbox_inf": [inf_x, inf_y, inf_w, inf_h],
            })

        # Free predictor state and GPU memory
        if hasattr(self.predictor, "reset_predictor"):
            self.predictor.reset_predictor()

        del inference_image
        if self.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

        # Update meta.json
        all_segments = existing_segments + results
        self._save_meta(masks_dir, {
            "scale": scale,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "settings": {"max_dim": max_dim, "mode": "manual"},
            "segments": all_segments,
        })

        return results

    @staticmethod
    def render_segment(
        image_path,
        output_dir,
        index,
        meta=None,
        upscale=None,
        out_path=None,
        image_array=None,
        tight_crop=None,
        tight_crop_padding=None,
    ):
        """
        Render a single full-resolution RGBA PNG on demand.

        Pass meta (already-loaded dict) to avoid re-reading meta.json.
        upscale > 1 resizes the output (e.g. 2 = 2x for sticker quality).
        out_path overrides the default save location.
        tight_crop removes empty transparent space around the segment.
        Returns the path to the rendered file, or None on failure.
        """
        if upscale is None:
            upscale = cfg["upscale"]
        if tight_crop is None:
            tight_crop = cfg["tight_crop"]
        if tight_crop_padding is None:
            tight_crop_padding = cfg["tight_crop_padding"]
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

        # Load compact mask (at inference resolution)
        npz_path = os.path.join(masks_dir, f"{index}.npz")
        try:
            npz = np.load(npz_path)
            shape = tuple(npz["shape"])
            mask_inf = (
                np.unpackbits(npz["mask"])[: shape[0] * shape[1]]
                .reshape(shape)
                .astype(np.uint8)
                * 255
            )
        except (IOError, ValueError, KeyError):
            return None  # corrupt or missing mask file

        # ── Morphological cleanup ────────────────────────────────────────────
        # Remove isolated noise pixels (1-2px) that SAM sometimes produces at
        # mask boundaries.  Adaptive: skip for small segments where MORPH_OPEN
        # would destroy thin structures (text, lines in presentation graphics).
        ks = cfg["morph_kernel_size"]
        morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        mask_area = np.count_nonzero(mask_inf)
        if mask_area > 800:  # only clean masks large enough to survive erosion
            # Close first to fill small holes, then open to remove noise
            mask_inf = cv2.morphologyEx(mask_inf, cv2.MORPH_CLOSE, morph_k)
            mask_inf = cv2.morphologyEx(mask_inf, cv2.MORPH_OPEN, morph_k)

        # ── Contour-based boundary smoothing at inference resolution ──────────
        #
        # Morphological kernels cannot remove blobs that are connected to the
        # main shape via wide pixel bridges (e.g. a white oval frame bleeding
        # into a white slide background).  Instead we:
        #
        #   1. Extract the outer contour of the mask.
        #   2. Apply a circular Gaussian along the contour vertices — this
        #      rounds off protrusions (including attached noise blobs) in
        #      proportion to the object's perimeter so the effect is
        #      resolution-independent.
        #   3. Re-fill the smoothed polygon to get a clean binary mask.
        #
        # Working at inference resolution keeps the contour short (faster) and
        # means the sigma is naturally scaled to the actual object size.
        contours, _ = cv2.findContours(
            mask_inf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            main = max(contours, key=cv2.contourArea)
            pts = main[:, 0, :].astype(np.float64)  # (N, 2)

            # Compactness-aware sigma: compact shapes (circles) need less
            # smoothing; complex shapes (text, irregular edges) need more
            # careful treatment.  Uses the isoperimetric ratio to scale.
            perimeter = len(pts)
            compactness = (4 * np.pi * mask_area) / (perimeter * perimeter + 1e-6)
            # compactness ~1.0 for circles, ~0.1 for very jagged shapes
            base_sigma = perimeter / cfg["contour_sigma_divisor"]
            # Reduce sigma for complex shapes to preserve detail
            sigma = max(
                cfg["contour_sigma_min"],
                min(base_sigma * min(compactness * 2.0, 1.0), cfg["contour_sigma_max"]),
            )
            ks = int(6 * sigma) | 1  # kernel size (always odd)
            pad = ks // 2  # circular wrap-around padding

            t = np.arange(ks) - pad
            kernel = np.exp(-0.5 * t**2 / sigma**2)
            kernel /= kernel.sum()

            smooth = np.empty_like(pts)
            for axis in range(2):
                col = pts[:, axis]
                padded = np.concatenate([col[-pad:], col, col[:pad]])
                # mode='valid' on a symmetric pad of size ks//2 gives length N
                smooth[:, axis] = np.convolve(padded, kernel, mode="valid")

            smooth_pts = np.round(smooth).astype(np.int32).reshape(-1, 1, 2)
            # Only redraw if smoothed contour is non-degenerate (>= 3 points)
            if len(smooth_pts) >= 3:
                mask_inf = np.zeros_like(mask_inf)
                cv2.drawContours(mask_inf, [smooth_pts], -1, 255, cv2.FILLED)

        # ── Upscale to full resolution ────────────────────────────────────────
        # Use CUBIC interpolation for smooth upscaling, then apply sigmoid-like
        # sharpening to tighten the edge and prevent grey fringe/halo artifacts
        # that LANCZOS creates on binary mask boundaries.
        if scale != 1.0:
            full_mask_np = cv2.resize(
                mask_inf, (fw, fh), interpolation=cv2.INTER_CUBIC
            )
            # Sigmoid sharpening: push grey fringe pixels toward 0 or 255
            # to create a crisp but anti-aliased edge
            mid = 127.5
            sharpness = cfg.get("mask_upscale_sharpness", 0.08)
            mask_float = 1.0 / (1.0 + np.exp(-(full_mask_np.astype(np.float32) - mid) * sharpness))
            full_mask_np = (mask_float * 255).astype(np.uint8)
        else:
            full_mask_np = mask_inf

        # ── RGBA assembly ────────────────────────────────────────────────────
        if image_array is not None:
            # Batch path: image already loaded as numpy array (thread-safe read)
            cropped_rgb = image_array[fy : fy + fh, fx : fx + fw].copy()
        else:
            original = Image.open(image_path).convert("RGB")
            cropped_rgb = np.array(original.crop((fx, fy, fx + fw, fy + fh)))
            original.close()

        rgba = np.empty((fh, fw, 4), dtype=np.uint8)
        rgba[:, :, :3] = cropped_rgb
        rgba[:, :, 3] = full_mask_np
        del cropped_rgb

        segment_img = Image.fromarray(rgba, "RGBA")
        del rgba

        # ── Edge feathering via distance transform ─────────────────────────
        # Distance transform computes exact distance from each pixel to the
        # nearest background pixel, producing a smooth alpha gradient that
        # follows the contour shape precisely.  The feather radius scales
        # with segment size and output resolution so the visual edge width
        # stays consistent across upscale factors.
        binary_mask = (full_mask_np > 127).astype(np.uint8)
        del full_mask_np
        dist = cv2.distanceTransform(
            binary_mask, cv2.DIST_L2, cfg["dist_transform_mask_size"]
        )
        feather_px = max(
            cfg["feather_min_px"],
            min(min(fw, fh) * cfg["feather_factor"], cfg["feather_max_px"]),
        )
        del binary_mask
        alpha_float = np.clip(dist / feather_px, 0.0, 1.0)
        del dist
        # Smoothstep t²(3−2t) for natural-looking edge falloff
        alpha_float = alpha_float * alpha_float * (3.0 - 2.0 * alpha_float)
        alpha_np = (alpha_float * 255).astype(np.uint8)
        del alpha_float
        blur_k = cfg["feather_blur_kernel"]
        blur_s = cfg["feather_blur_sigma"]
        if blur_k > 1 and blur_s > 0:
            alpha_np = cv2.GaussianBlur(alpha_np, (blur_k, blur_k), sigmaX=blur_s)
        segment_img.putalpha(Image.fromarray(alpha_np, "L"))

        # ── Tight cropping ────────────────────────────────────────────────
        # Remove fully-transparent rows/columns (e.g. from contour smoothing)
        # to minimise file size and wasted space.  Applied before upscale so
        # the padding scales proportionally with the output resolution.
        if tight_crop:
            alpha_arr = np.array(segment_img.split()[-1])
            rows = np.any(alpha_arr > 0, axis=1)
            cols = np.any(alpha_arr > 0, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                pad = tight_crop_padding
                h, w = alpha_arr.shape
                crop_box = (
                    max(0, cmin - pad),
                    max(0, rmin - pad),
                    min(w, cmax + 1 + pad),
                    min(h, rmax + 1 + pad),
                )
                segment_img = segment_img.crop(crop_box)
            del alpha_arr

        # Upscale for sticker-quality output.  Re-apply light feathering at
        # upscaled resolution so edges stay crisp at the target size.
        if upscale > 1:
            new_size = (segment_img.width * upscale, segment_img.height * upscale)
            segment_img = segment_img.resize(new_size, Image.LANCZOS)
            # Scale blur to upscale factor for consistent edge appearance
            scaled_blur_k = max(1, int(blur_k * upscale) | 1)  # keep odd
            scaled_blur_s = blur_s * upscale
            if scaled_blur_k > 1 and scaled_blur_s > 0:
                alpha_arr = np.array(segment_img.split()[-1])
                alpha_arr = cv2.GaussianBlur(
                    alpha_arr, (scaled_blur_k, scaled_blur_k), sigmaX=scaled_blur_s
                )
                segment_img.putalpha(Image.fromarray(alpha_arr, "L"))

        if out_path is None:
            filename = f"segment_{index:03d}.png"
            out_path = os.path.join(output_dir, filename)

        segment_img.save(out_path, optimize=True)
        segment_img.close()

        return out_path
