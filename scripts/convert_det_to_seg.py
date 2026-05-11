#!/usr/bin/env python3
"""
Конвертер разметки детекции (bounding box) → сегментации (полигон)
с использованием Segment Anything Model (SAM).

Формат входа: YOLO Detection (<class> <xc> <yc> <w> <h>)
Формат выхода: YOLO Segmentation (<class> <x1> <y1> <x2> <y2> ... <xn> <yn>)
"""

import os
import sys
import yaml
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple

# Инициализация логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/convert_det_to_seg.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def load_sam_model(checkpoint_path: str, device: str = "auto") -> Tuple:
    """Загрузка модели SAM и предиктора."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
        import torch
    except ImportError:
        logger.error("Не установлены зависимости SAM. Выполните: pip install segment-anything torch torchvision")
        sys.exit(1)
    
    # Автоопределение устройства
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Определение типа модели по имени файла
    if "vit_h" in checkpoint_path or "sam_vit_h" in checkpoint_path:
        model_type = "vit_h"
    elif "vit_l" in checkpoint_path or "sam_vit_l" in checkpoint_path:
        model_type = "vit_l"
    else:
        model_type = "vit_b"  # default
    
    logger.info(f"Загрузка SAM модели ({model_type}) с {checkpoint_path} на устройство {device}...")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    logger.info("✅ Модель SAM загружена")
    return predictor, device


def denormalize_bbox(bbox_norm: List[float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Конвертация нормализованных YOLO bbox [xc, yc, w, h] в пиксельные [x1, y1, x2, y2].
    """
    xc, yc, w, h = bbox_norm
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    
    # Ограничение в пределах изображения
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    
    return x1, y1, x2, y2


def mask_to_polygon(mask: np.ndarray, img_w: int, img_h: int, 
                    approx_epsilon_ratio: float = 0.005) -> Optional[List[float]]:
    """
    Конвертация бинарной маски в нормализованный полигон.
    
    Args:
        mask: бинарная маска (H, W)
        img_w, img_h: размеры исходного изображения
        approx_epsilon_ratio: коэффициент аппроксимации полигона (доля периметра)
    
    Returns:
        Список нормализованных координат [x1, y1, x2, y2, ...] или None
    """
    # Преобразование маски в uint8 для OpenCV
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Поиск контуров
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Выбор контура с максимальной площадью
    cnt = max(contours, key=cv2.contourArea)
    
    # Аппроксимация полигона для уменьшения количества вершин
    epsilon = approx_epsilon_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Минимальное количество точек для полигона
    if len(approx) < 3:
        return None
    
    # Нормализация координат
    polygon_norm = []
    for pt in approx:
        x, y = pt[0]
        # Округление до 6 знаков после запятой для компактности
        polygon_norm.append(round(float(x) / img_w, 6))
        polygon_norm.append(round(float(y) / img_h, 6))
    
    return polygon_norm


def compute_iou_bbox_mask(bbox_xyxy: Tuple[int, int, int, int], 
                          mask: np.ndarray) -> float:
    """
    Вычисление IoU между bounding box и маской.
    Используется для валидации качества сегментации.
    """
    x1, y1, x2, y2 = bbox_xyxy
    
    # Маска области bbox
    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True
    
    # Пересечение и объединение
    intersection = np.logical_and(mask, bbox_mask).sum()
    union = np.logical_or(mask, bbox_mask).sum()
    
    if union == 0:
        return 0.0
    return intersection / union


def parse_detection_label(line: str) -> Tuple[int, List[float]]:
    """
    Парсинг строки разметки детекции YOLO.
    
    Returns:
        (class_id, bbox_norm) где bbox_norm = [xc, yc, w, h]
    """
    parts = list(map(float, line.strip().split()))
    if len(parts) < 5:
        raise ValueError(f"Некорректный формат строки детекции: {line}")
    
    cls_id = int(parts[0])
    bbox_norm = parts[1:5]
    return cls_id, bbox_norm


def process_image(predictor, image_path: Path, label_path: Path, 
                  output_dir: Path, cfg: dict) -> dict:
    """
    Обработка одного изображения: генерация полигонов по bbox.
    """
    import torch
    
    stats = {
        "image": image_path.name,
        "objects_processed": 0,
        "objects_success": 0,
        "objects_low_confidence": 0,
        "objects_low_iou": 0,
        "errors": []
    }
    
    # Чтение изображения
    logger.info(f"📷 Обработка изображения: {image_path.name}")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"❌ Не удалось прочитать изображение: {image_path}")
        stats["errors"].append("Failed to read image")
        return stats
    
    img_h, img_w = image.shape[:2]
    logger.info(f"   Размер изображения: {img_w}x{img_h}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Установка изображения в предиктор
    predictor.set_image(image_rgb)
    
    # Чтение разметки детекции
    if not label_path.exists():
        logger.error(f"❌ Файл разметки не найден: {label_path}")
        stats["errors"].append(f"Label file not found: {label_path.name}")
        return stats
    
    # Проверяем, пустой ли файл
    with open(label_path, "r", encoding="utf-8") as f:
        label_content = f.read().strip()
    
    if not label_content:
        logger.warning(f"⚠️ Файл разметки пуст: {label_path}")
        return stats
    
    logger.info(f"   Чтение разметки из: {label_path.name}")
    
    polygons = []
    
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    logger.info(f"   Найдено строк в разметке: {len(lines)}")
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            cls_id, bbox_norm = parse_detection_label(line)
            logger.info(f"   📦 Объект {line_num}: class={cls_id}, bbox={bbox_norm}")
        except ValueError as e:
            logger.warning(f"   ⚠️ Ошибка парсинга строки {line_num}: {e}")
            stats["errors"].append(f"Line {line_num}: {e}")
            continue
        
        stats["objects_processed"] += 1
        
        # Денормализация bbox
        bbox_xyxy = denormalize_bbox(bbox_norm, img_w, img_h)
        logger.info(f"      BBox в пикселях: {bbox_xyxy}")
        
        # ... [код до генерации маски без изменений] ...

        # Генерация маски через SAM
        try:
            masks, scores, _ = predictor.predict(
                box=np.array(bbox_xyxy),
                multimask_output=True
            )
            logger.info(f"      SAM вернул {len(masks)} масок, scores: {scores}")
            
        except Exception as e:
            logger.error(f"      ❌ Ошибка SAM: {e}")
            stats["errors"].append(f"SAM prediction error: {e}")
            continue

        # === ВЫБОР ЛУЧШЕЙ МАСКИ по комбинированному скорингу ===
        best_mask = None
        best_combined_score = -1
        best_score = 0
        best_iou = 0

        for i, (mask, score) in enumerate(zip(masks, scores)):
            iou = compute_iou_bbox_mask(bbox_xyxy, mask)
            circularity = check_circular_shape(mask)  # ← Функция должна быть определена выше!
            
            # Комбинированный score: 40% уверенность SAM, 30% IoU, 30% круговость
            combined_score = 0.4 * score + 0.3 * iou + 0.3 * circularity
            
            logger.info(f"      Маска {i}: score={score:.4f}, IoU={iou:.4f}, circularity={circularity:.4f}, combined={combined_score:.4f}")
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_mask = mask
                best_score = score
                best_iou = iou

        # Если ни одна маска не подошла — берем лучшую по score
        if best_mask is None and len(masks) > 0:
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            best_iou = compute_iou_bbox_mask(bbox_xyxy, best_mask)
            logger.info(f"      ⚠️ Используем маску по score: {best_score:.4f}")

        # === ФИЛЬТРАЦИЯ ПОСЛЕ ВЫБОРА ===
        conf_threshold = cfg.get("confidence_threshold", 0.85)
        if best_score < conf_threshold:
            logger.info(f"      ⚠️ Отклонено: confidence {best_score:.4f} < {conf_threshold}")
            stats["objects_low_confidence"] += 1
            continue

        iou_threshold = cfg.get("iou_threshold", 0.6)
        if best_iou < iou_threshold:
            logger.info(f"      ⚠️ Отклонено: IoU {best_iou:.4f} < {iou_threshold}")
            stats["objects_low_iou"] += 1
            continue

        # Проверка на пустую маску
        if best_mask.sum() == 0:
            logger.warning(f"      ❌ Маска пустая")
            stats["errors"].append("Empty mask")
            continue

        # === КОНВЕРТАЦИЯ В ПОЛИГОН ===
        polygon = mask_to_polygon(
            best_mask, img_w, img_h, 
            approx_epsilon_ratio=cfg.get("approx_epsilon_ratio", 0.005)
        )

        if polygon is None:
            logger.warning(f"      ❌ Не удалось извлечь полигон")
            stats["errors"].append("Failed to extract polygon from mask")
            continue

        logger.info(f"      ✅ Успешно! Полигон: {len(polygon)//2} точек")
        stats["objects_success"] += 1

        # Форматирование строки вывода
        poly_line = [str(cls_id)] + [f"{coord:.6f}" for coord in polygon]
        polygons.append(" ".join(poly_line))

        # ... [остальной код без изменений] ...
    
    # Сохранение результата
    output_path = output_dir / f"{image_path.stem}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        if polygons:
            f.write("\n".join(polygons) + "\n")
            logger.info(f"   💾 Сохранено {len(polygons)} полигонов в {output_path.name}")
        else:
            logger.warning(f"   ⚠️ Нет полигонов для сохранения, файл будет пустым")
    
    # Очистка кэша
    predictor.reset_image()
    
    return stats

def check_circular_shape(mask: np.ndarray) -> float:
    """
    Проверяет, насколько маска похожа на круг.
    Возвращает коэффициент круговости (0-1).
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter == 0:
        return 0.0
    
    # Коэффициент круговости: 4π*Area/Perimeter²
    # Для идеального круга = 1.0
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    return circularity

def convert_dataset(cfg: dict):
    """Основная функция конвертации датасета."""
    
    # === Инициализация ===
    sam_checkpoint = cfg.get("sam_checkpoint", "sam_vit_h_4b8939.pth")
    device = cfg.get("device", "auto")
    
    if not os.path.exists(sam_checkpoint):
        logger.error(f"Файл модели SAM не найден: {sam_checkpoint}")
        logger.info("Скачайте модель с: https://github.com/facebookresearch/segment-anything")
        sys.exit(1)
    
    predictor, _ = load_sam_model(sam_checkpoint, device)
    
    # === Пути ===
    src_root = Path(cfg["dataset"]["source_root"])
    dest_root = Path(cfg["dataset"]["dest_root"])
    output_dir = dest_root / "labels_segmentation"  # Отдельная папка для сегментации
    output_dir.mkdir(parents=True, exist_ok=True)
    
    src_images = src_root / "images"
    src_labels = src_root / "labels"  # Детекция
    img_exts = set(cfg["dataset"].get("img_extensions", [".jpg", ".jpeg", ".png", ".bmp"]))
    
    # === Индексация файлов разметки ===
    logger.info("🔍 Индексация файлов разметки детекции...")
    label_index = {p.stem: p for p in src_labels.glob("*.txt")}
    
    # === Сбор пар изображений ===
    logger.info("🔍 Сбор валидных пар (изображение + разметка)...")
    pairs = []
    for img_path in src_images.iterdir():
        if img_path.suffix.lower() in img_exts:
            if img_path.stem in label_index:
                pairs.append((img_path, label_index[img_path.stem]))
            else:
                logger.warning(f"⚠️ Нет разметки для {img_path.name}")
    
    logger.info(f"✅ Найдено {len(pairs)} валидных пар")
    
    # === Обработка ===
    logger.info("🚀 Запуск конвертации...")
    
    total_stats = {
        "images_processed": 0,
        "total_objects": 0,
        "successful_conversions": 0,
        "low_confidence": 0,
        "low_iou": 0,
        "errors": []
    }
    
    for img_path, lbl_path in pairs:
        stats = process_image(predictor, img_path, lbl_path, output_dir, cfg)
        
        total_stats["images_processed"] += 1
        total_stats["total_objects"] += stats["objects_processed"]
        total_stats["successful_conversions"] += stats["objects_success"]
        total_stats["low_confidence"] += stats["objects_low_confidence"]
        total_stats["low_iou"] += stats["objects_low_iou"]
        total_stats["errors"].extend([f"{stats['image']}: {e}" for e in stats["errors"]])
        
        if total_stats["images_processed"] % 10 == 0:
            logger.info(f"📊 Прогресс: {total_stats['images_processed']}/{len(pairs)} изображений")
    
    # === Отчёт ===
    logger.info("\n" + "="*60)
    logger.info("📈 ОТЧЁТ О КОНВЕРТАЦИИ")
    logger.info("="*60)
    logger.info(f"Изображений обработано: {total_stats['images_processed']}")
    logger.info(f"Объектов всего: {total_stats['total_objects']}")
    logger.info(f"Успешных конвертаций: {total_stats['successful_conversions']}")
    logger.info(f"Отклонено (низкий confidence): {total_stats['low_confidence']}")
    logger.info(f"Отклонено (низкий IoU): {total_stats['low_iou']}")
    
    if total_stats["errors"]:
        logger.warning(f"Ошибок: {len(total_stats['errors'])}")
        for err in total_stats["errors"][:10]:  # Показываем первые 10
            logger.warning(f"  • {err}")
        if len(total_stats["errors"]) > 10:
            logger.warning(f"  ... и ещё {len(total_stats['errors']) - 10}")
    
    # Сохранение отчёта
    report_path = Path(cfg["training"]["project"]) / "conversion_report.yaml"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        yaml.dump(total_stats, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"💾 Отчёт сохранён: {report_path}")
    
    # Вычисление процента успеха
    if total_stats["total_objects"] > 0:
        success_rate = total_stats["successful_conversions"] / total_stats["total_objects"] * 100
        logger.info(f"✅ Успешность конвертации: {success_rate:.1f}%")
    
    logger.info("✨ Конвертация завершена")


def main():
    parser = argparse.ArgumentParser(description="Конвертер детекция → сегментация (SAM)")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="Путь к конфигурационному файлу")
    parser.add_argument("--sam-checkpoint", type=str, default=None,
                        help="Переопределение пути к SAM checkpoint")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Порог confidence для фильтрации масок")
    parser.add_argument("--iou-threshold", type=float, default=None,
                        help="Порог IoU для валидации")
    args = parser.parse_args()
    
    # Загрузка конфигурации
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Применение аргументов командной строки (переопределение)
    if args.sam_checkpoint:
        cfg["sam_checkpoint"] = args.sam_checkpoint
    if args.confidence is not None:
        cfg["confidence_threshold"] = args.confidence
    if args.iou_threshold is not None:
        cfg["iou_threshold"] = args.iou_threshold
    
    # Установка значений по умолчанию, если не заданы
    cfg.setdefault("confidence_threshold", 0.85)
    cfg.setdefault("iou_threshold", 0.6)
    cfg.setdefault("approx_epsilon_ratio", 0.005)
    
    convert_dataset(cfg)


if __name__ == "__main__":
    main()