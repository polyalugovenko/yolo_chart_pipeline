#!/usr/bin/env python3
"""Detect pie-chart sector boundaries from YOLO masks and polar color profiles."""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from polar_transform import (
    apply_polar_transform,
    detect_sector_boundaries,
    extract_polar_features,
    visualize_boundary_detection,
    visualize_features,
    visualize_polar,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def mask_to_perfect_circle(mask: np.ndarray) -> tuple | None:
    """Approximate a YOLO mask with the enclosing filled circle."""
    mask_bin = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(round(x)), int(round(y)))
    radius = int(round(radius))

    h, w = mask.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)
    return center, radius, circle_mask


def bgr_to_normalized_lab(image_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to normalized Lab: L in [0, 1], a/b in [-1, 1]."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] /= 255.0
    lab[:, :, 1] = (lab[:, :, 1] - 128.0) / 127.0
    lab[:, :, 2] = (lab[:, :, 2] - 128.0) / 127.0
    lab[:, :, 1:] = np.clip(lab[:, :, 1:], -1.0, 1.0)
    return lab


def save_features(features: np.ndarray, output_path: Path) -> None:
    df = pd.DataFrame(
        features,
        columns=[f"feat_{feature_idx}" for feature_idx in range(features.shape[1])],
    )
    df["angle"] = np.arange(features.shape[0])
    df.to_csv(output_path, index=False)


def save_boundary_scores(detection: dict, output_path: Path) -> None:
    df = pd.DataFrame(
        {
            "angle": np.arange(len(detection["score"])),
            "boundary_score": detection["score"],
            "raw_boundary_score": detection["raw_score"],
            "is_boundary": [
                int(angle in detection["boundaries"])
                for angle in range(len(detection["score"]))
            ],
        }
    )
    df.to_csv(output_path, index=False)


def process_image(model: YOLO, image_path: Path, output_dir: Path, cfg: dict) -> dict:
    stats = {"image": image_path.name, "success": False, "n_masks": 0, "n_sectors": 0, "files": []}

    results = model.predict(
        str(image_path),
        conf=cfg.get("conf_threshold", 0.5),
        imgsz=cfg.get("imgsz", 640),
        verbose=False,
    )
    if not results or not results[0].masks:
        logger.warning("No masks: %s", image_path.name)
        return stats

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("Could not read image: %s", image_path)
        return stats

    h_img, w_img = image.shape[:2]
    ring_range = (
        cfg.get("ring_inner_ratio", 0.55),
        cfg.get("ring_outer_ratio", 0.92),
    )

    for mask_idx, mask_obj in enumerate(results[0].masks):
        mask_arr = mask_obj.data[0].cpu().numpy()
        if mask_arr.shape[:2] != (h_img, w_img):
            mask_arr = cv2.resize(mask_arr, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        circle_res = mask_to_perfect_circle(mask_arr)
        if not circle_res:
            logger.warning("Could not fit circle: %s mask %d", image_path.name, mask_idx)
            continue

        center, radius, circle_mask = circle_res
        polar_data = apply_polar_transform(
            image_bgr=image,
            mask=(circle_mask / 255.0).astype(np.float32),
            center=center,
            radius=radius,
            ring_range=ring_range,
        )
        if not polar_data:
            continue

        polar_img = polar_data["polar_img"]
        polar_lab = bgr_to_normalized_lab(polar_img)

        stem = f"{image_path.stem}_mask{mask_idx}"

        polar_path = output_dir / f"{stem}_polar.png"
        visualize_polar(image, polar_data, polar_path)
        stats["files"].append(str(polar_path))

        features = extract_polar_features(polar_lab, polar_data["polar_mask"])
        features_path = output_dir / f"{stem}_features.csv"
        save_features(features, features_path)
        stats["files"].append(str(features_path))

        feature_vis_path = output_dir / f"{stem}_features_detailed.png"
        visualize_features(features, polar_data, feature_vis_path)
        stats["files"].append(str(feature_vis_path))

        detection = detect_sector_boundaries(
            polar_lab,
            polar_mask=polar_data["polar_mask"],
            smooth_window=cfg.get("smooth_window", 7),
            score_window=cfg.get("score_window", 5),
            threshold_factor=cfg.get("threshold_factor", 0.8),
            min_distance_deg=cfg.get("min_boundary_distance_deg", 8),
            min_sector_deg=cfg.get("min_sector_deg", 8),
            merge_color_distance=cfg.get("merge_color_distance", 0.08),
            weak_boundary_ratio=cfg.get("weak_boundary_ratio", 0.0),
            radial_bands=cfg.get("radial_bands", 3),
            max_candidate_peaks=cfg.get("max_candidate_peaks", 0),
            optimize_boundaries=cfg.get("optimize_boundaries", True),
            optimizer_keep_score=cfg.get("optimizer_keep_score", 0.16),
        )

        boundary_vis_path = output_dir / f"{stem}_boundaries.png"
        visualize_boundary_detection(polar_data, detection, boundary_vis_path)
        stats["files"].append(str(boundary_vis_path))

        scores_path = output_dir / f"{stem}_boundary_scores.csv"
        save_boundary_scores(detection, scores_path)
        stats["files"].append(str(scores_path))

        sectors_path = output_dir / f"{stem}_sectors.csv"
        pd.DataFrame(detection["sectors"]).to_csv(sectors_path, index=False)
        stats["files"].append(str(sectors_path))

        stats["n_masks"] += 1
        stats["n_sectors"] += len(detection["sectors"])
        logger.info(
            "%s mask %d: boundaries=%s sectors=%d",
            image_path.name,
            mask_idx,
            detection["boundaries"],
            len(detection["sectors"]),
        )

    stats["success"] = stats["n_masks"] > 0
    return stats


def run_pipeline(cfg: dict):
    weights = Path(cfg.get("weights_path", "weights/best.pt"))
    if not weights.exists():
        logger.error("Weights not found: %s", weights)
        return

    model = YOLO(str(weights))
    src_dir = Path(cfg.get("source_dir", "data/test_images"))
    out_dir = Path(cfg.get("output_dir", "outputs/clustering"))
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        image_path
        for image_path in src_dir.glob("*")
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    logger.info("Found %d images", len(images))

    summary = []
    for image_path in images:
        stats = process_image(model, image_path, out_dir, cfg)
        summary.append(stats)
        logger.info(
            "%s %s: masks=%d sectors=%d",
            "OK" if stats["success"] else "SKIP",
            image_path.name,
            stats["n_masks"],
            stats["n_sectors"],
        )

    summary_path = out_dir / "sector_detection_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    logger.info("Results saved to: %s", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--threshold-factor", type=float, default=None)
    parser.add_argument("--min-boundary-distance", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cl_cfg = cfg.get("clustering", {})
    if args.threshold_factor is not None:
        cl_cfg["threshold_factor"] = args.threshold_factor
    if args.min_boundary_distance is not None:
        cl_cfg["min_boundary_distance_deg"] = args.min_boundary_distance

    run_pipeline(cl_cfg)


if __name__ == "__main__":
    main()
