#!/usr/bin/env python3
"""Polar transform and deterministic sector-boundary detection for pie charts."""

import logging
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def apply_polar_transform(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    center: tuple,
    radius: int,
    angle_steps: int = 360,
    ring_range: tuple | None = None,
) -> dict:
    """Convert a circular chart region to a polar ring with rows=radius and columns=angle."""
    if ring_range is None:
        r_min, r_max = 0, radius
    else:
        r_min = int(ring_range[0] * radius)
        r_max = int(ring_range[1] * radius)

    ring_height = r_max - r_min
    if ring_height <= 0:
        logger.error("Invalid polar ring range: %s", ring_range)
        return {}

    # OpenCV warpPolar returns rows=angle and columns=radius. The rest of the
    # pipeline uses rows=radius and columns=angle, so crop radius then transpose.
    polar_size = (radius, angle_steps)
    polar_img_full = cv2.warpPolar(
        image_bgr,
        polar_size,
        center,
        radius,
        cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR,
    )
    polar_mask_full = cv2.warpPolar(
        (mask > 0.5).astype(np.uint8) * 255,
        polar_size,
        center,
        radius,
        cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST,
    )

    polar_img = np.transpose(polar_img_full[:, r_min:r_max, :], (1, 0, 2))
    polar_mask = np.transpose(polar_mask_full[:, r_min:r_max], (1, 0))
    polar_rgb = cv2.cvtColor(polar_img, cv2.COLOR_BGR2RGB)

    return {
        "polar_img": polar_img,
        "polar_mask": polar_mask,
        "polar_rgb": polar_rgb,
        "angles": np.linspace(0, 360, angle_steps, endpoint=False),
        "radii": np.linspace(r_min, r_max, ring_height),
        "center": center,
        "radius": radius,
        "ring_range": (r_min, r_max),
    }


def visualize_polar(
    original_bgr: np.ndarray,
    polar_data: dict,
    output_path: Path,
    show_grid: bool = True,
    dpi: int = 150,
) -> None:
    """Save the original chart, polar color image, and polar mask side by side."""
    polar_rgb = polar_data["polar_rgb"]
    polar_mask = polar_data["polar_mask"]
    center = polar_data["center"]
    radius = polar_data["radius"]
    ring_range = polar_data["ring_range"]
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Polar transform of pie chart", fontsize=14, fontweight="bold")

    axes[0].imshow(original_rgb)
    axes[0].add_patch(plt.Circle(center, radius, color="lime", fill=False, linewidth=2))
    axes[0].add_patch(plt.Circle(center, ring_range[0], color="orange", fill=False, linestyle="--"))
    axes[0].add_patch(plt.Circle(center, ring_range[1], color="orange", fill=False, linestyle="--"))
    axes[0].plot(center[0], center[1], "r+", markersize=10)
    axes[0].set_title("Original + fitted circle")
    axes[0].set_aspect("equal")
    axes[0].grid(show_grid, alpha=0.3, linestyle=":")

    for ax, img, title, cmap in [
        (axes[1], polar_rgb, "Polar color image", None),
        (axes[2], polar_mask, "Polar mask", "gray"),
    ]:
        ax.imshow(
            img,
            cmap=cmap,
            aspect="auto",
            extent=[0, 360, ring_range[0], ring_range[1]],
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Angle, deg")
        ax.set_ylabel("Radius, px")
        ax.set_xticks(np.arange(0, 361, 45))
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("Polar visualization saved: %s", output_path)


def circular_smooth_1d(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Smooth a circular 1D signal with a moving average."""
    if window <= 1:
        return values.astype(np.float32)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="wrap")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def compute_angle_color_profile(
    polar_lab: np.ndarray,
    polar_mask: np.ndarray | None = None,
    smooth_window: int = 7,
) -> np.ndarray:
    """Return one robust Lab color vector per angle."""
    height, width, _ = polar_lab.shape
    valid = np.ones((height, width), dtype=bool) if polar_mask is None else polar_mask > 0
    fallback = np.median(polar_lab.reshape(-1, 3), axis=0)

    profile = np.zeros((width, 3), dtype=np.float32)
    for angle in range(width):
        pixels = polar_lab[:, angle][valid[:, angle]]
        profile[angle] = np.median(pixels, axis=0) if len(pixels) else fallback

    if smooth_window <= 1:
        return profile

    smoothed = np.empty_like(profile)
    for channel in range(profile.shape[1]):
        smoothed[:, channel] = circular_smooth_1d(profile[:, channel], smooth_window)
    return smoothed


def _circular_distance(a: int, b: int, period: int = 360) -> int:
    dist = abs(int(a) - int(b)) % period
    return min(dist, period - dist)


def _circular_span(start: int, end: int, period: int = 360) -> int:
    return int((end - start) % period)


def _angle_slice(values: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < end:
        return values[start:end]
    return np.concatenate([values[start:], values[:end]], axis=0)


def _sector_mean_color(profile: np.ndarray, start: int, end: int) -> np.ndarray:
    values = _angle_slice(profile, start, end)
    if len(values) == 0:
        return profile[start % len(profile)]
    return np.mean(values, axis=0)


def _build_sectors(boundaries: list[int], profile: np.ndarray) -> list[dict]:
    if len(boundaries) < 2:
        return []

    sectors = []
    for idx, start in enumerate(boundaries):
        end = boundaries[(idx + 1) % len(boundaries)]
        span = _circular_span(start, end, len(profile))
        sectors.append(
            {
                "sector": idx,
                "start_angle": int(start),
                "end_angle": int(end),
                "angle_deg": float(span),
                "percent": float(span / len(profile) * 100.0),
                "mean_l": float(_sector_mean_color(profile, start, end)[0]),
                "mean_a": float(_sector_mean_color(profile, start, end)[1]),
                "mean_b": float(_sector_mean_color(profile, start, end)[2]),
            }
        )
    return sectors


def _segment_values(profile: np.ndarray, start: int, end: int) -> np.ndarray:
    values = _angle_slice(profile, start, end)
    return values if len(values) else profile[start % len(profile)][None, :]


def _segment_sse(profile: np.ndarray, start: int, end: int) -> float:
    values = _segment_values(profile, start, end)
    mean = np.mean(values, axis=0)
    return float(np.sum((values - mean) ** 2))


def _compute_boundary_score(profile: np.ndarray, score_window: int) -> tuple[np.ndarray, np.ndarray]:
    prev_profile = np.roll(profile, score_window, axis=0)
    next_profile = np.roll(profile, -score_window, axis=0)
    central_diff = np.linalg.norm(next_profile - prev_profile, axis=1) / 2.0
    direct_diff = np.linalg.norm(profile - np.roll(profile, 1, axis=0), axis=1)
    raw_score = np.maximum(central_diff, direct_diff)
    score = circular_smooth_1d(raw_score, max(1, score_window))
    return raw_score.astype(np.float32), score.astype(np.float32)


def _robust_threshold(score: np.ndarray, threshold_factor: float, fallback_percentile: float = 85) -> float:
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    threshold = median + threshold_factor * 1.4826 * mad
    if threshold <= median + 1e-6:
        threshold = float(np.percentile(score, fallback_percentile))
    return threshold


def _normalize_score(score: np.ndarray) -> np.ndarray:
    median = np.median(score)
    mad = np.median(np.abs(score - median))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = np.std(score)
    if scale < 1e-6:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - median) / scale, 0.0, None).astype(np.float32)


def _non_max_suppression_circular(
    scores: np.ndarray,
    candidate_angles: np.ndarray,
    min_distance: int,
) -> list[int]:
    selected = []
    for angle in sorted(candidate_angles, key=lambda idx: scores[idx], reverse=True):
        angle = int(angle)
        if all(_circular_distance(angle, other, len(scores)) >= min_distance for other in selected):
            selected.append(angle)
    return sorted(selected)


def _merge_close_boundaries(
    boundaries: list[int],
    scores: np.ndarray,
    min_distance: int,
    profile: np.ndarray,
    distinct_color_distance: float = 0.16,
) -> list[int]:
    merged = sorted(int(b) for b in boundaries)
    if len(merged) < 2:
        return merged

    changed = True
    while changed and len(merged) > 1:
        changed = False
        close_pairs = [
            (i, (i + 1) % len(merged), _circular_distance(merged[i], merged[(i + 1) % len(merged)], len(scores)))
            for i in range(len(merged))
            if _circular_distance(merged[i], merged[(i + 1) % len(merged)], len(scores)) < min_distance
        ]
        if not close_pairs:
            break

        pair_to_merge = None
        for i, j, _ in sorted(close_pairs, key=lambda item: item[2]):
            if len(merged) < 3:
                pair_to_merge = (i, j)
                break

            start = merged[i]
            end = merged[j]
            tiny_color = _sector_mean_color(profile, start, end)
            left_color = _sector_mean_color(profile, merged[i - 1], start)
            right_color = _sector_mean_color(profile, end, merged[(j + 1) % len(merged)])

            if min(np.linalg.norm(tiny_color - left_color), np.linalg.norm(tiny_color - right_color)) < distinct_color_distance:
                pair_to_merge = (i, j)
                break

        if pair_to_merge is None:
            break

        i, j = pair_to_merge
        drop = merged[i] if scores[merged[i]] <= scores[merged[j]] else merged[j]
        merged.remove(drop)
        changed = True

    return sorted(merged)


def _merge_by_sector_quality(
    boundaries: list[int],
    profile: np.ndarray,
    score: np.ndarray,
    min_sector_deg: int,
    merge_color_distance: float,
    weak_boundary_ratio: float,
) -> list[int]:
    boundaries = sorted(int(b) for b in boundaries)
    if len(boundaries) < 3:
        return boundaries

    max_score = float(np.max(score)) if len(score) else 0.0
    weak_score = weak_boundary_ratio * max_score

    changed = True
    while changed and len(boundaries) >= 3:
        changed = False

        for idx, start in enumerate(boundaries):
            end = boundaries[(idx + 1) % len(boundaries)]
            span = _circular_span(start, end, len(score))
            if span and span < min_sector_deg:
                drop = start if score[start] <= score[end] else end
                boundaries.remove(drop)
                changed = True
                break
        if changed:
            continue

        for idx, boundary in enumerate(list(boundaries)):
            prev_boundary = boundaries[idx - 1]
            next_boundary = boundaries[(idx + 1) % len(boundaries)]
            left_color = _sector_mean_color(profile, prev_boundary, boundary)
            right_color = _sector_mean_color(profile, boundary, next_boundary)
            color_dist = float(np.linalg.norm(left_color - right_color))

            if color_dist < merge_color_distance:
                boundaries.remove(boundary)
                changed = True
                break

            if weak_boundary_ratio > 0 and score[boundary] < weak_score and color_dist < merge_color_distance * 2.0:
                boundaries.remove(boundary)
                changed = True
                break

    return sorted(boundaries)


def _optimize_boundaries_by_merge(
    boundaries: list[int],
    profile: np.ndarray,
    score: np.ndarray,
    min_keep_score: float,
    score_weight: float = 0.08,
) -> list[int]:
    """Drop boundaries that add little color separation or segmentation quality."""
    boundaries = sorted(int(b) for b in boundaries)
    if len(boundaries) < 3:
        return boundaries

    score_scale = max(float(np.percentile(score, 95)), 1e-6)
    changed = True
    while changed and len(boundaries) >= 3:
        changed = False
        removal_options = []

        for idx, boundary in enumerate(boundaries):
            prev_boundary = boundaries[idx - 1]
            next_boundary = boundaries[(idx + 1) % len(boundaries)]
            left_values = _segment_values(profile, prev_boundary, boundary)
            right_values = _segment_values(profile, boundary, next_boundary)
            if len(left_values) == 0 or len(right_values) == 0:
                removal_options.append((0.0, boundary))
                continue

            left_color = np.mean(left_values, axis=0)
            right_color = np.mean(right_values, axis=0)
            color_dist = float(np.linalg.norm(left_color - right_color))

            left_sse = _segment_sse(profile, prev_boundary, boundary)
            right_sse = _segment_sse(profile, boundary, next_boundary)
            merged_sse = _segment_sse(profile, prev_boundary, next_boundary)
            merged_span = max(1, _circular_span(prev_boundary, next_boundary, len(profile)))
            sse_gain = max(0.0, merged_sse - left_sse - right_sse) / merged_span

            left_span = _circular_span(prev_boundary, boundary, len(profile))
            right_span = _circular_span(boundary, next_boundary, len(profile))
            small_sector_bonus = 0.08 if min(left_span, right_span) <= 10 and color_dist >= 0.18 else 0.0
            local_score = float(score[boundary]) / score_scale
            keep_score = color_dist + 0.35 * sse_gain + score_weight * local_score + small_sector_bonus
            removal_options.append((keep_score, boundary))

        weakest_score, weakest_boundary = min(removal_options, key=lambda item: item[0])
        if weakest_score >= min_keep_score:
            break

        boundaries.remove(weakest_boundary)
        changed = True

    return sorted(boundaries)


def detect_sector_boundaries(
    polar_lab: np.ndarray,
    polar_mask: np.ndarray | None = None,
    smooth_window: int = 7,
    score_window: int = 5,
    threshold_factor: float = 0.8,
    min_distance_deg: int = 3,
    min_sector_deg: int = 8,
    merge_color_distance: float = 0.08,
    weak_boundary_ratio: float = 0.0,
    radial_bands: int = 3,
    max_candidate_peaks: int = 0,
    optimize_boundaries: bool = True,
    optimizer_keep_score: float = 0.16,
) -> dict:
    """Detect pie-sector boundaries from a polar Lab image."""
    full_profile = compute_angle_color_profile(polar_lab, polar_mask, smooth_window)
    raw_score, full_score = _compute_boundary_score(full_profile, score_window)
    normalized_scores = [_normalize_score(full_score)]

    height = polar_lab.shape[0]
    if radial_bands > 1 and height >= radial_bands * 4:
        for band_idx in range(radial_bands):
            r0 = int(round(band_idx * height / radial_bands))
            r1 = int(round((band_idx + 1) * height / radial_bands))
            if r1 - r0 < 4:
                continue
            band_mask = None if polar_mask is None else polar_mask[r0:r1, :]
            band_profile = compute_angle_color_profile(polar_lab[r0:r1, :, :], band_mask, smooth_window)
            _, band_score = _compute_boundary_score(band_profile, score_window)
            normalized_scores.append(_normalize_score(band_score))

    score = np.mean(np.vstack(normalized_scores), axis=0).astype(np.float32)
    score = circular_smooth_1d(score, max(1, score_window))
    threshold = _robust_threshold(score, threshold_factor, fallback_percentile=75)

    left = np.roll(score, 1)
    right = np.roll(score, -1)
    local_peaks = np.where((score >= left) & (score >= right))[0]
    threshold_candidates = local_peaks[score[local_peaks] >= threshold]

    top_candidates = np.array([], dtype=int)
    if max_candidate_peaks > 0 and len(local_peaks) > 0:
        min_top_score = max(threshold * 0.75, float(np.percentile(score, 70)))
        order = np.argsort(score[local_peaks])[::-1]
        top_candidates = np.array(
            [
                int(angle)
                for angle in local_peaks[order[:max_candidate_peaks]]
                if score[int(angle)] >= min_top_score
            ],
            dtype=int,
        )

    candidates = np.unique(np.concatenate([threshold_candidates, top_candidates]))
    boundaries = _non_max_suppression_circular(score, candidates, min_distance_deg)
    boundaries = _merge_close_boundaries(boundaries, score, min_sector_deg, full_profile)
    boundaries = _merge_by_sector_quality(
        boundaries,
        full_profile,
        score,
        min_sector_deg=min_sector_deg,
        merge_color_distance=merge_color_distance,
        weak_boundary_ratio=weak_boundary_ratio,
    )
    if optimize_boundaries:
        boundaries = _optimize_boundaries_by_merge(
            boundaries,
            full_profile,
            score,
            min_keep_score=optimizer_keep_score,
        )

    return {
        "profile": full_profile,
        "raw_score": raw_score.astype(np.float32),
        "score": score.astype(np.float32),
        "threshold": float(threshold),
        "boundaries": boundaries,
        "sectors": _build_sectors(boundaries, full_profile),
    }


def visualize_boundary_detection(
    polar_data: dict,
    detection: dict,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Save polar image, boundary score, and estimated sector percentages."""
    polar_rgb = polar_data["polar_rgb"]
    ring_range = polar_data["ring_range"]
    score = detection["score"]
    boundaries = detection["boundaries"]
    sectors = detection["sectors"]
    angles = np.arange(len(score))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1, 1]})

    axes[0].imshow(
        polar_rgb,
        aspect="auto",
        extent=[0, 360, ring_range[0], ring_range[1]],
        origin="lower",
        interpolation="nearest",
    )
    axes[0].set_title("Polar ring with detected boundaries")
    axes[0].set_ylabel("Radius, px")
    for angle in boundaries:
        axes[0].axvline(angle, color="red", linewidth=1.5)

    axes[1].plot(angles, score, color="steelblue", linewidth=1)
    axes[1].axhline(detection["threshold"], color="tomato", linestyle="--", linewidth=1)
    for angle in boundaries:
        axes[1].axvline(angle, color="red", alpha=0.7, linewidth=1)
    axes[1].set_xlim(0, 360)
    axes[1].set_title("Boundary score")
    axes[1].grid(True, alpha=0.3, linestyle=":")

    axes[2].axis("off")
    if sectors:
        rows = [
            f"{s['sector']}: {s['start_angle']}..{s['end_angle']} deg, "
            f"{s['angle_deg']:.1f} deg, {s['percent']:.2f}%"
            for s in sectors
        ]
        axes[2].text(0.01, 0.95, "\n".join(rows), va="top", family="monospace", fontsize=10)
    else:
        axes[2].text(0.01, 0.95, "No stable sector boundaries detected", va="top", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("Boundary visualization saved: %s", output_path)
