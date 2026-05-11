#!/usr/bin/env python3
"""
Полярное преобразование для круговых диаграмм.
Преобразует изображение и маску в полярные координаты для анализа секторов.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def apply_polar_transform(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    center: tuple,
    radius: int,
    angle_steps: int = 360,
    ring_range: tuple = None
) -> dict:
    """
    Применяет полярное преобразование к изображению и маске.
    
    Args:
        image_bgr: изображение в BGR формате (OpenCV)
        mask: бинарная маска (любые значения > 0.5 считаются внутри)
        center: (cx, cy) центр круга
        radius: радиус круга в пикселях
        angle_steps: количество шагов по углу (обычно 360 для 1°/шаг)
        ring_range: (r_min_ratio, r_max_ratio) для анализа только кольца, 
                    например (0.5, 0.95) для внешнего кольца
    
    Returns:
        dict с результатами:
        {
            'polar_img': np.ndarray,      # полярное изображение (angle_steps, ring_height, 3)
            'polar_mask': np.ndarray,     # полярная маска (angle_steps, ring_height)
            'polar_rgb': np.ndarray,      # для визуализации (angle_steps, ring_height, 3) в RGB
            'angles': np.ndarray,         # массив углов [0, 360)
            'radii': np.ndarray,          # массив радиусов
            'center': tuple,
            'radius': int,
            'ring_range': tuple
        }
    """
    h, w = image_bgr.shape[:2]
    cx, cy = center
    
    # Настройка ring_range
    if ring_range is None:
        r_min, r_max = 0, radius
    else:
        r_min = int(ring_range[0] * radius)
        r_max = int(ring_range[1] * radius)
    
    ring_height = r_max - r_min
    if ring_height <= 0:
        logger.error("❌ Неверный диапазон кольца")
        return {}
    
    # Размер полярного изображения: (width=углы, height=радиус)
    # OpenCV warpPolar uses width for radius and height for angle.
    # Build (angle, radius), crop the radius columns, then transpose to the
    # project convention: rows = radius, columns = angle.
    polar_size = (radius, angle_steps)
    
    # Полярное преобразование изображения (интерполяция CUBIC для качества)
    polar_img_full = cv2.warpPolar(
        image_bgr, 
        polar_size, 
        center, 
        radius, 
        cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR
    )
    
    
    # Полярное преобразование маски (интерполяция NEAREST для бинарности)
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    polar_mask_full = cv2.warpPolar(
        mask_uint8,
        polar_size,
        center,
        radius,
        cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST
    )
    
    polar_img = np.transpose(polar_img_full[:, r_min:r_max, :], (1, 0, 2))
    polar_mask = np.transpose(polar_mask_full[:, r_min:r_max], (1, 0))

    if polar_img.shape[0] != ring_height or polar_img.shape[1] != angle_steps:
        logger.warning(
            "Unexpected polar shape: got %s, expected (%d, %d, 3)",
            polar_img.shape,
            ring_height,
            angle_steps,
        )

    # polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # polar_mask = cv2.rotate(polar_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Конвертация полярного изображения в RGB для matplotlib
    polar_rgb = cv2.cvtColor(polar_img, cv2.COLOR_BGR2RGB)
    
    # Массивы углов и радиусов для подписей
    angles = np.linspace(0, 360, angle_steps, endpoint=False)
    radii = np.linspace(r_min, r_max, ring_height)
    
    return {
        'polar_img': polar_img,
        'polar_mask': polar_mask,
        'polar_rgb': polar_rgb,
        'angles': angles,
        'radii': radii,
        'center': center,
        'radius': radius,
        'ring_range': (r_min, r_max)
    }


def visualize_polar(
    original_bgr: np.ndarray,
    polar_data: dict,
    output_path: Path,
    show_grid: bool = True,
    dpi: int = 150
):
    """
    Визуализация оригинала и полярного представления.
    """
    polar_rgb = polar_data['polar_rgb']
    polar_mask = polar_data['polar_mask']
    center = polar_data['center']
    radius = polar_data['radius']
    ring_range = polar_data['ring_range']
    
    # Конвертация оригинала в RGB
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Создаём фигуру с 3 подграфиками
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Полярное преобразование круговой диаграммы', fontsize=14, fontweight='bold')
    
    # 1. Оригинал с маской и центром
    ax1 = axes[0]
    ax1.imshow(original_rgb)
    
    circle = plt.Circle(center, radius, color='lime', fill=False, linewidth=2, label=f'R={radius}px')
    ax1.add_patch(circle)
    ax1.plot(center[0], center[1], 'r+', markersize=10, label='Центр')
    
    if ring_range[0] > 0:
        inner_circle = plt.Circle(center, ring_range[0], color='orange', fill=False, 
                                  linewidth=1, linestyle='--', label='Внутр. граница')
        outer_circle = plt.Circle(center, ring_range[1], color='orange', fill=False, 
                                  linewidth=1, linestyle='--', label='Внеш. граница')
        ax1.add_patch(inner_circle)
        ax1.add_patch(outer_circle)
    
    ax1.set_title('Original + mask', fontsize=12)
    ax1.set_xlabel('X (пиксели)')
    ax1.set_ylabel('Y (пиксели)')
    ax1.legend(fontsize=8)
    ax1.grid(show_grid, alpha=0.3, linestyle=':')
    ax1.set_aspect('equal')
    
    # 2. Полярное изображение
    ax2 = axes[1]
    # 🔧 ИСПРАВЛЕНИЕ: правильно настраиваем extent и origin
    im2 = ax2.imshow(polar_rgb, aspect='auto', 
                     extent=[0, 360, ring_range[0], ring_range[1]],  # [left, right, bottom, top]
                     origin='lower',  # 🔧 Важно: origin='lower' чтобы Y рос снизу вверх
                     interpolation='nearest')
    ax2.set_title('Polar image (color)', fontsize=12)
    ax2.set_xlabel('Угол (градусы)')
    ax2.set_ylabel('Радиус (пиксели)')
    
    # 🔧 Настраиваем тики для оси Y (радиус)
    n_yticks = 5
    y_ticks = np.linspace(ring_range[0], ring_range[1], n_yticks, dtype=int)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels([f'{int(t)}' for t in y_ticks])
    
    # 🔧 Настраиваем тики для оси X (углы)
    x_ticks = np.arange(0, 361, 45)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{int(t)}°' for t in x_ticks])
    
    if show_grid:
        ax2.grid(True, alpha=0.3, linestyle=':', color='white')
    
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Полярная маска
    ax3 = axes[2]
    im3 = ax3.imshow(polar_mask, cmap='gray', aspect='auto', 
                     extent=[0, 360, ring_range[0], ring_range[1]],  # 🔧 Тоже исправляем
                     origin='lower',  # 🔧 Важно!
                     vmin=0, vmax=255)
    ax3.set_title('Polar mask', fontsize=12)
    ax3.set_xlabel('Угол (градусы)')
    ax3.set_ylabel('Радиус (пиксели)')
    
    # 🔧 Те же тики что и для polar_rgb
    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels([f'{int(t)}' for t in y_ticks])
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([f'{int(t)}°' for t in x_ticks])
    
    if show_grid:
        ax3.grid(True, alpha=0.3, linestyle=':', color='gray')
    
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Маска')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"💾 Визуализация сохранена: {output_path}")


def extract_polar_features(
    polar_lab: np.ndarray,
    polar_mask: np.ndarray = None
) -> np.ndarray:
    """
    Улучшенный feature extractor с исправленной индексацией.
    """
    H, W, C = polar_lab.shape

    if W != 360:
        raise ValueError(f"Ожидалась ширина 360, получено {W}")

    # =========================================================
    # MASK
    # =========================================================
    if polar_mask is not None:
        valid = polar_mask > 0
    else:
        valid = np.ones((H, W), dtype=bool)

    # =========================================================
    # ROBUST STATISTICS
    # =========================================================
    median_vals = np.zeros((360, 3))
    p10_vals = np.zeros((360, 3))
    p90_vals = np.zeros((360, 3))
    std_vals = np.zeros((360, 3))

    for angle in range(360):
        pixels = polar_lab[:, angle][valid[:, angle]]
        if len(pixels) < 5:
            continue
        median_vals[angle] = np.median(pixels, axis=0)
        p10_vals[angle] = np.percentile(pixels, 10, axis=0)
        p90_vals[angle] = np.percentile(pixels, 90, axis=0)
        std_vals[angle] = np.std(pixels, axis=0)

    # =========================================================
    # ANGULAR DIFFERENCE (центрированная разность для точности)
    # =========================================================
    # Используем центрированную разность: (val[i+1] - val[i-1]) / 2
    # Это даёт более точную локализацию градиента на границе
    prev_vals = np.roll(median_vals, 1, axis=0)
    next_vals = np.roll(median_vals, -1, axis=0)
    angular_diff = np.linalg.norm(next_vals - prev_vals, axis=1) / 2.0


    # =========================================================
    # LOCAL VARIANCE — ✅ ИСПРАВЛЕНА ИНДЕКСАЦИЯ
    # =========================================================
    window = 7
    pad = window // 2

    padded = np.pad(median_vals, ((pad, pad), (0, 0)), mode='wrap')

    local_variance = np.zeros(360)
    robust_contrast = np.zeros(360)

    for i in range(360):
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: смещение на pad
        win = padded[i + pad : i + pad + window]

        local_variance[i] = np.mean(np.var(win, axis=0))

        p10 = np.percentile(win, 10, axis=(0, 1))
        p90 = np.percentile(win, 90, axis=(0, 1))
        robust_contrast[i] = np.mean(p90 - p10)  # усредняем по каналам

    # =========================================================
    # BOUNDARY CONSISTENCY
    # =========================================================
    diff_map = compute_angular_difference_map(polar_lab)
    boundary_consistency = compute_boundary_consistency(diff_map)

    # =========================================================
    # EDGE CONTINUITY — улучшено: используем оба направления градиента
    # =========================================================
    edge_continuity = compute_vertical_edge_continuity(polar_lab)

    # =========================================================
    # RADIAL CONSISTENCY
    # =========================================================
    radial_consistency = 1.0 / (1.0 + np.mean(std_vals, axis=1))

    # =========================================================
    # FEATURE STACK
    # =========================================================
    features = np.column_stack([
        median_vals,                    # 0-2
        std_vals,                       # 3-5
        p90_vals[:, 0] - p10_vals[:, 0],  # 6
        p90_vals[:, 1] - p10_vals[:, 1],  # 7
        p90_vals[:, 2] - p10_vals[:, 2],  # 8
        angular_diff,                   # 9
        local_variance,                 # 10
        robust_contrast,                # 11
        boundary_consistency,           # 12
        edge_continuity,                # 13
        radial_consistency              # 14
    ])

    # В конце extract_polar_features добавить:
    logger.info(f"✅ Признаки: shape={features.shape}, angles=360")
    logger.info(f"📊 Angular diff: min={angular_diff.min():.4f}, "
                f"max={angular_diff.max():.4f}, "
                f"peak_angles={np.argsort(angular_diff)[-5:]}")

    return features.astype(np.float32)


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


def circular_smooth_2d(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Smooth a circular sequence of feature vectors along the angle axis."""
    if window <= 1:
        return values.astype(np.float32)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="wrap")
    smoothed = np.empty_like(values, dtype=np.float32)
    for channel in range(values.shape[1]):
        smoothed[:, channel] = np.convolve(
            padded[:, channel],
            kernel,
            mode="valid",
        )
    return smoothed


def _circular_distance(a: int, b: int, period: int = 360) -> int:
    dist = abs(int(a) - int(b)) % period
    return min(dist, period - dist)


def _non_max_suppression_circular(
    scores: np.ndarray,
    candidate_angles: np.ndarray,
    min_distance: int,
) -> list:
    selected = []
    for angle in sorted(candidate_angles, key=lambda idx: scores[idx], reverse=True):
        angle = int(angle)
        if all(_circular_distance(angle, other, len(scores)) >= min_distance for other in selected):
            selected.append(angle)
    return sorted(selected)


def _merge_close_boundaries(
    boundaries: list,
    scores: np.ndarray,
    min_distance: int,
    profile: np.ndarray = None,
    distinct_color_distance: float = 0.16,
) -> list:
    merged = sorted(int(b) for b in boundaries)
    if len(merged) < 2:
        return merged

    changed = True
    while changed and len(merged) > 1:
        changed = False
        pairs = [
            (
                i,
                (i + 1) % len(merged),
                _circular_distance(merged[i], merged[(i + 1) % len(merged)], len(scores)),
            )
            for i in range(len(merged))
        ]
        close_pairs = [p for p in pairs if p[2] < min_distance]
        if not close_pairs:
            break

        pair_to_merge = None
        for i, j, _ in sorted(close_pairs, key=lambda p: p[2]):
            if profile is None or len(merged) < 3:
                pair_to_merge = (i, j)
                break

            start = merged[i]
            end = merged[j]
            prev_boundary = merged[i - 1]
            next_boundary = merged[(j + 1) % len(merged)]

            tiny_color = _sector_mean_color(profile, start, end)
            left_color = _sector_mean_color(profile, prev_boundary, start)
            right_color = _sector_mean_color(profile, end, next_boundary)
            left_dist = float(np.linalg.norm(tiny_color - left_color))
            right_dist = float(np.linalg.norm(tiny_color - right_color))

            if min(left_dist, right_dist) < distinct_color_distance:
                pair_to_merge = (i, j)
                break

        if pair_to_merge is None:
            break

        i, j = pair_to_merge
        drop = j if scores[merged[i]] >= scores[merged[j]] else i
        merged.pop(drop)
        changed = True

    return sorted(merged)


def compute_angle_color_profile(
    polar_lab: np.ndarray,
    polar_mask: np.ndarray = None,
    smooth_window: int = 5,
) -> np.ndarray:
    """Return one robust Lab color vector per angle."""
    H, W, _ = polar_lab.shape
    if polar_mask is None:
        valid = np.ones((H, W), dtype=bool)
    else:
        valid = polar_mask > 0

    profile = np.zeros((W, 3), dtype=np.float32)
    fallback = np.median(polar_lab.reshape(-1, 3), axis=0)
    for angle in range(W):
        pixels = polar_lab[:, angle][valid[:, angle]]
        if len(pixels) >= 3:
            profile[angle] = np.median(pixels, axis=0)
        else:
            profile[angle] = fallback

    return circular_smooth_2d(profile, smooth_window)


def _sector_channel_mean(profile: np.ndarray, start: int, end: int, channel: int) -> float:
    if start < end:
        values = profile[start:end, channel]
    else:
        values = np.r_[profile[start:, channel], profile[:end, channel]]
    return float(np.mean(values)) if len(values) else 0.0


def _angle_slice(values: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < end:
        return values[start:end]
    return np.concatenate([values[start:], values[:end]], axis=0)


def _sector_mean_color(profile: np.ndarray, start: int, end: int) -> np.ndarray:
    values = _angle_slice(profile, start, end)
    if len(values) == 0:
        return np.zeros(profile.shape[1], dtype=np.float32)
    return np.mean(values, axis=0)


def _build_sectors(boundaries: list, profile: np.ndarray) -> list:
    sectors = []
    if len(boundaries) < 2:
        return sectors

    boundaries = sorted(int(b) for b in boundaries)
    for idx, start in enumerate(boundaries):
        end = boundaries[(idx + 1) % len(boundaries)]
        span = (end - start) % 360
        if span == 0:
            continue
        mean_color = _sector_mean_color(profile, start, end)
        sectors.append(
            {
                "sector": idx + 1,
                "start_angle": int(start),
                "end_angle": int(end),
                "angle_deg": float(span),
                "percent": float(span * 100.0 / 360.0),
                "mean_lab_l": float(mean_color[0]),
                "mean_lab_a": float(mean_color[1]),
                "mean_lab_b": float(mean_color[2]),
            }
        )
    return sectors


def _segment_values(profile: np.ndarray, start: int, end: int) -> np.ndarray:
    return _angle_slice(profile, start, end)


def _segment_sse(profile: np.ndarray, start: int, end: int) -> float:
    values = _segment_values(profile, start, end)
    if len(values) <= 1:
        return 0.0
    mean = np.mean(values, axis=0)
    centered = values - mean
    return float(np.sum(centered * centered))


def _circular_span(start: int, end: int, period: int = 360) -> int:
    span = (int(end) - int(start)) % period
    return period if span == 0 else span


def _optimize_boundaries_by_merge(
    boundaries: list,
    profile: np.ndarray,
    score: np.ndarray,
    min_keep_score: float = 0.16,
    color_weight: float = 1.0,
    score_weight: float = 0.08,
    error_weight: float = 0.45,
    small_sector_bonus: float = 0.04,
    max_iterations: int = 200,
) -> list:
    """
    Remove boundaries that do not meaningfully improve the global segmentation.

    A boundary is useful if the neighboring sectors have different colors, if
    merging them increases within-sector error, or if the local boundary score is
    strong. This keeps small but real sectors better than a pure min-size rule.
    """
    boundaries = sorted(int(b) for b in boundaries)
    if len(boundaries) < 3:
        return boundaries

    score_max = float(np.max(score)) if len(score) else 0.0
    score_scale = score_max if score_max > 1e-6 else 1.0

    for _ in range(max_iterations):
        if len(boundaries) < 3:
            break

        removal_options = []
        n = len(boundaries)
        for idx, boundary in enumerate(boundaries):
            prev_boundary = boundaries[idx - 1]
            next_boundary = boundaries[(idx + 1) % n]

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
            merged_span = max(1, _circular_span(prev_boundary, next_boundary))
            error_gain = max(0.0, (merged_sse - left_sse - right_sse) / merged_span)

            left_span = _circular_span(prev_boundary, boundary)
            right_span = _circular_span(boundary, next_boundary)
            small_span = min(left_span, right_span)
            small_bonus = small_sector_bonus if small_span <= 12 and color_dist > 0.18 else 0.0

            boundary_score = float(score[boundary]) / score_scale
            keep_score = (
                color_weight * color_dist
                + error_weight * np.sqrt(error_gain)
                + score_weight * boundary_score
                + small_bonus
            )
            removal_options.append((keep_score, boundary))

        weakest_score, weakest_boundary = min(removal_options, key=lambda item: item[0])
        if weakest_score >= min_keep_score:
            break

        boundaries.remove(weakest_boundary)

    return sorted(boundaries)


def _compute_boundary_score(profile: np.ndarray, score_window: int) -> tuple[np.ndarray, np.ndarray]:
    prev_profile = np.roll(profile, 1, axis=0)
    next_profile = np.roll(profile, -1, axis=0)
    central_score = np.linalg.norm(next_profile - prev_profile, axis=1) / 2.0
    direct_score = np.linalg.norm(profile - prev_profile, axis=1)
    raw_score = np.maximum(central_score, direct_score)
    return raw_score.astype(np.float32), circular_smooth_1d(raw_score, score_window)


def _robust_threshold(score: np.ndarray, threshold_factor: float, fallback_percentile: float = 85) -> float:
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma < 1e-6:
        robust_sigma = float(np.std(score))

    threshold = median + threshold_factor * robust_sigma
    if threshold <= median:
        threshold = float(np.percentile(score, fallback_percentile))
    return float(threshold)


def _normalize_score(score: np.ndarray) -> np.ndarray:
    median = np.median(score)
    mad = np.median(np.abs(score - median))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = np.std(score)
    if scale < 1e-6:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - median) / scale, 0.0, None).astype(np.float32)


def _merge_by_sector_quality(
    boundaries: list,
    profile: np.ndarray,
    score: np.ndarray,
    min_sector_deg: int,
    merge_color_distance: float,
    weak_boundary_ratio: float,
) -> list:
    boundaries = sorted(int(b) for b in boundaries)
    if len(boundaries) < 3:
        return boundaries

    max_score = float(np.max(score)) if len(score) else 0.0
    weak_score = weak_boundary_ratio * max_score

    changed = True
    while changed and len(boundaries) >= 3:
        changed = False
        n = len(boundaries)

        # First remove tiny sectors by dropping the weaker of their two borders.
        for idx in range(n):
            start = boundaries[idx]
            end = boundaries[(idx + 1) % n]
            span = (end - start) % 360
            if span == 0 or span >= min_sector_deg:
                continue
            drop = start if score[start] <= score[end] else end
            boundaries.remove(drop)
            changed = True
            break
        if changed:
            continue

        # Then remove borders between nearly identical neighboring colors.
        n = len(boundaries)
        for idx, boundary in enumerate(list(boundaries)):
            prev_boundary = boundaries[idx - 1]
            next_boundary = boundaries[(idx + 1) % n]
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


def detect_sector_boundaries(
    polar_lab: np.ndarray,
    polar_mask: np.ndarray = None,
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
    """
    Detect pie-sector boundaries from the angular Lab color profile.

    Returns boundaries in degrees and sector percentages between consecutive
    boundaries. This is a deterministic baseline, not a CNN feature pipeline.
    """
    full_profile = compute_angle_color_profile(
        polar_lab,
        polar_mask=polar_mask,
        smooth_window=smooth_window,
    )

    raw_score, full_score = _compute_boundary_score(full_profile, score_window)
    normalized_scores = [_normalize_score(full_score)]

    H = polar_lab.shape[0]
    if radial_bands > 1 and H >= radial_bands * 4:
        for band_idx in range(radial_bands):
            r0 = int(round(band_idx * H / radial_bands))
            r1 = int(round((band_idx + 1) * H / radial_bands))
            if r1 - r0 < 4:
                continue
            band_mask = None if polar_mask is None else polar_mask[r0:r1, :]
            band_profile = compute_angle_color_profile(
                polar_lab[r0:r1, :, :],
                polar_mask=band_mask,
                smooth_window=smooth_window,
            )
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
    boundaries = _merge_close_boundaries(
        boundaries,
        score,
        min_sector_deg,
        profile=full_profile,
    )
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

    sectors = _build_sectors(boundaries, full_profile)

    return {
        "profile": full_profile,
        "raw_score": raw_score.astype(np.float32),
        "score": score.astype(np.float32),
        "threshold": float(threshold),
        "boundaries": boundaries,
        "sectors": sectors,
    }


def visualize_boundary_detection(
    polar_data: dict,
    detection: dict,
    output_path: Path,
    dpi: int = 150,
):
    """Visualize polar image, boundary score, and estimated sector percentages."""
    polar_rgb = polar_data["polar_rgb"]
    ring_range = polar_data["ring_range"]
    score = detection["score"]
    threshold = detection["threshold"]
    boundaries = detection["boundaries"]
    sectors = detection["sectors"]
    angles = np.arange(len(score))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

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
    axes[1].axhline(threshold, color="tomato", linestyle="--", linewidth=1)
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

    logger.info(f"Boundary visualization saved: {output_path}")


def visualize_features(
    features: np.ndarray,
    polar_data: dict,
    output_path: Path,
    dpi: int = 150
):
    """
    Упрощенная визуализация признаков.
    """
    
    # Проверка размерности
    if features.ndim == 3:
        if features.shape[0] == 1:
            features = features[0]
        else:
            features = features[0]
    
    if features.shape[0] != 360:
        logger.error(f"❌ Ожидалось 360 углов, получено {features.shape[0]}")
        return
    
    # Названия признаков (15 штук)
    feature_names = [
        'Median L', 'Median a', 'Median b',      # 0-2
        'Std L', 'Std a', 'Std b',               # 3-5
        'Spread L', 'Spread a', 'Spread b',      # 6-8
        'Angular Diff',                          # 9
        'Local Variance',                        # 10
        'Robust Contrast',                       # 11
        'Boundary Consistency',                  # 12
        'Edge Continuity',                       # 13
        'Radial Consistency'                     # 14
    ]
    
    # Создаем фигуру: 5 строк × 3 колонки = 15 графиков
    fig, axes = plt.subplots(5, 3, figsize=(18, 25))
    fig.suptitle('Признаки круговой диаграммы (360 углов)', 
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    angles = np.arange(360)
    
    # Визуализация каждого признака
    for i in range(min(len(feature_names), features.shape[1])):
        ax = axes[i]
        
        # Если признак постоянный — показываем горизонтальную линию
        if np.std(features[:, i]) < 1e-6:
            ax.axhline(y=features[0, i], color='gray', linestyle='--')
            ax.text(180, features[0, i], f'const={features[0, i]:.4f}', 
                   ha='center', va='bottom', fontsize=8)
        else:
            ax.plot(angles, features[:, i], linewidth=1, color='steelblue')
        
        ax.set_title(f'{feature_names[i]}', fontsize=10, fontweight='bold')
        ax.set_xlim([0, 360])
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Статистика
        mean_val = np.mean(features[:, i])
        std_val = np.std(features[:, i])
        ax.text(0.95, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
               transform=ax.transAxes, fontsize=7,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Скрываем пустые подграфики если признаков < 15
    for i in range(features.shape[1], len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"💾 Визуализация признаков: {output_path}")


def print_feature_statistics(features: np.ndarray):
    """
    Вывод статистики по всем признакам в консоль.
    """
    # ✅ Обновлённые названия для 15 признаков
    feature_names = [
        'Median L', 'Median a', 'Median b',      # 0-2: медианный цвет
        'Std L', 'Std a', 'Std b',               # 3-5: разброс цвета
        'Spread L', 'Spread a', 'Spread b',      # 6-8: p90-p10
        'Angular Diff',                          # 9: градиент по углу
        'Local Variance',                        # 10: локальная дисперсия
        'Robust Contrast',                       # 11: робастный контраст
        'Boundary Consistency',                  # 12: согласованность границ
        'Edge Continuity',                       # 13: непрерывность рёбер
        'Radial Consistency'                     # 14: однородность по радиусу
    ]
    
    print("\n" + "="*80)
    print("📊 СТАТИСТИКА ПРИЗНАКОВ")
    print("="*80)
    print(f"{'Признак':<25} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Std':>8}")
    print("-"*80)
    
    for i in range(features.shape[1]):
        # 🔧 Защита от выхода за границы списка
        name = feature_names[i] if i < len(feature_names) else f'Feat_{i}'
        
        min_val = np.min(features[:, i])
        max_val = np.max(features[:, i])
        mean_val = np.mean(features[:, i])
        std_val = np.std(features[:, i])
        
        print(f"{name:<25} | {min_val:8.4f} | {max_val:8.4f} | "
              f"{mean_val:8.4f} | {std_val:8.4f}")
    
    print("="*80 + "\n")


def check_feature_quality(features: np.ndarray) -> dict:
    """
    Проверка качества признаков.
    
    Returns:
        dict с результатами проверки
    """
    checks = {}
    
    # 1. Проверка на NaN/Inf
    checks['has_nan'] = np.any(np.isnan(features))
    checks['has_inf'] = np.any(np.isinf(features))
    
    # 2. Проверка диапазонов
    checks['mean_L_range'] = (0 <= np.mean(features[:, 0]) <= 1)
    checks['std_non_negative'] = np.all(features[:, 3:6] >= 0)
    checks['radial_consistency_range'] = (0 <= np.mean(features[:, 6]) <= 1)
    
    # 3. Проверка на постоянные значения (признак не меняется)
    checks['constant_features'] = []
    for i in range(features.shape[1]):
        if np.std(features[:, i]) < 1e-6:
            checks['constant_features'].append(i)
    
    # 4. Проверка корреляций
    corr_matrix = np.corrcoef(features.T)
    checks['high_correlations'] = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > 0.95:
                checks['high_correlations'].append((i, j, corr_matrix[i, j]))
    
    # Вывод результатов
    print("\n" + "="*80)
    print("✅ ПРОВЕРКА КАЧЕСТВА ПРИЗНАКОВ")
    print("="*80)
    
    if checks['has_nan']:
        print("❌ ОБНАРУЖЕНЫ NaN значения!")
    else:
        print("✅ NaN значения отсутствуют")
    
    if checks['has_inf']:
        print("❌ ОБНАРУЖЕНЫ Inf значения!")
    else:
        print("✅ Inf значения отсутствуют")
    
    if checks['mean_L_range']:
        print("✅ Mean L в диапазоне [0, 1]")
    else:
        print("❌ Mean L вне диапазона [0, 1]")
    
    if checks['std_non_negative']:
        print("✅ Std значения неотрицательны")
    else:
        print("❌ Отрицательные Std значения!")
    
    if not checks['constant_features']:
        print("✅ Все признаки имеют вариацию")
    else:
        print(f"⚠️  Постоянные признаки (каналы): {checks['constant_features']}")
    
    if not checks['high_correlations']:
        print("✅ Сильных корреляций (>0.95) не обнаружено")
    else:
        print(f"⚠️  Сильные корреляции между признаками:")
        for i, j, corr in checks['high_correlations']:
            print(f"   - Признаки {i} и {j}: corr={corr:.3f}")
    
    print("="*80 + "\n")
    
    return checks


def compute_angular_difference_map(polar_lab: np.ndarray) -> np.ndarray:
    """
    Difference между соседними углами для каждого радиуса.

    Args:
        polar_lab: (H, 360, 3)

    Returns:
        diff_map: (H, 360)
    """

    shifted = np.roll(polar_lab, -1, axis=1)

    diff = shifted - polar_lab

    diff_map = np.linalg.norm(diff, axis=2)

    return diff_map

def compute_boundary_consistency(
    diff_map: np.ndarray,
    percentile: float = 75
) -> np.ndarray:
    """
    Насколько angular difference согласован по радиусу.

    Args:
        diff_map: (H, 360)

    Returns:
        consistency: (360,)
    """

    threshold = np.percentile(diff_map, percentile)

    strong_edges = diff_map > threshold

    consistency = np.mean(strong_edges, axis=0)

    return consistency

def compute_vertical_edge_continuity(
    polar_lab: np.ndarray
) -> np.ndarray:
    """
    Ищет длинные вертикальные структуры
    (границы секторов).

    Args:
        polar_lab: (H, 360, 3)

    Returns:
        continuity_score: (360,)
    """

    gray = polar_lab[:, :, 0]

    gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)

    sobel = cv2.Sobel(
        gray_u8,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sobel = np.abs(sobel)

    threshold = np.percentile(sobel, 80)

    edges = sobel > threshold

    continuity = np.mean(edges, axis=0)

    return continuity

def standardize_features(X):

    N, T, F = X.shape

    scaler = StandardScaler()

    X_flat = X.reshape(-1, F)

    X_scaled = scaler.fit_transform(X_flat)

    X_scaled = X_scaled.reshape(N, T, F)

    return X_scaled, scaler
