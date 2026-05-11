#!/usr/bin/env python3
"""
Визуализация результатов инференса YOLO (det/seg).
Работает напрямую с объектами Results из ultralytics.
"""

import os
import sys
import yaml
import logging
import argparse
import random
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def visualize_result(
    image_path: Path,
    result,
    output_path: Path,
    class_names: List[str] = None,
    show_confidence: bool = True,
    alpha: float = 0.4
):
    """
    Визуализация одного результата инференса.
    
    Args:
        image_path: путь к исходному изображению
        result: объект Results из ultralytics
        output_path: куда сохранить визуализацию
        class_names: список имён классов
        show_confidence: показывать ли confidence
        alpha: прозрачность заливки масок
    """
    # Чтение изображения
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Не удалось прочитать: {image_path}")
        return False
    
    img_h, img_w = image.shape[:2]
    
    # Цвета для классов
    if class_names is None:
        class_names = ["pie_chart"]
    random.seed(42)
    colors = [(random.randint(100, 255), 
               random.randint(100, 255), 
               random.randint(100, 255)) for _ in range(len(class_names))]
    
    # === ОБРАБОТКА МАСОК (сегментация) ===
    if result.masks is not None and len(result.masks) > 0:
        for i, (mask, box) in enumerate(zip(result.masks, result.boxes)):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            color = colors[cls_id % len(colors)]
            
            # Маска в формате HxW
            mask_array = mask.data[0].cpu().numpy()  # (H, W)
            mask_uint8 = (mask_array * 255).astype(np.uint8)
            
            # Находим контур для отрисовки
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            cnt = max(contours, key=cv2.contourArea)
            points = cnt.reshape(-1, 2)
            
            # Заливка (полупрозрачная)
            overlay = image.copy()
            cv2.fillPoly(overlay, [cnt], color)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
            # Контур
            cv2.polylines(image, [cnt], True, color, thickness=2)
            
            # Подпись
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            label = f"{cls_name}" + (f" {conf:.2f}" if show_confidence else "")
            
            # Позиция текста (верхняя левая точка контура)
            x, y = points[0]
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(image, (x, y - text_h - baseline - 5), 
                         (x + text_w, y - 5), color, cv2.FILLED)
            cv2.putText(image, label, (x, y - baseline - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # === ОБРАБОТКА BOXES (детекция) ===
    elif result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            color = colors[cls_id % len(colors)]
            
            # Координаты bbox
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Рисуем прямоугольник
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
            
            # Подпись
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            label = f"{cls_name}" + (f" {conf:.2f}" if show_confidence else "")
            
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w, y1 - 5), color, cv2.FILLED)
            cv2.putText(image, label, (x1, y1 - baseline - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Сохранение
    cv2.imwrite(str(output_path), image)
    logger.info(f"Сохранено: {output_path}")
    return True


def visualize_inference_results(cfg: dict, sample_size: Optional[int] = None):
    """
    Визуализация результатов инференса из папки predictions.
    """
    # Пути
    output_dir = Path(cfg["inference"]["output_dir"])
    pred_dir = output_dir / "predictions"
    
    if not pred_dir.exists():
        logger.error(f"Папка с предсказаниями не найдена: {pred_dir}")
        return
    
    # Загрузка модели (нужна для доступа к class names)
    weights = cfg["inference"]["weights"]
    if not Path(weights).exists():
        logger.warning(f"Веса не найдены: {weights}, используем дефолтные имена классов")
        class_names = cfg.get("training", {}).get("class_names", ["pie_chart"])
        model = None
    else:
        model = YOLO(weights)
        class_names = list(model.names.values())
    
    # Выходная папка для визуализаций
    vis_output_dir = output_dir / "visualization"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Поиск изображений с предсказаниями
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pred_images = sorted([f for f in pred_dir.glob("*") 
                         if f.suffix.lower() in img_exts and "_vis" not in f.name])
    
    if sample_size is not None:
        import random
        random.seed(42)
        pred_images = random.sample(pred_images, min(sample_size, len(pred_images)))
    
    logger.info(f"Визуализация {len(pred_images)} изображений...")
    
    # Обработка
    success = 0
    for img_path in pred_images:
        output_path = vis_output_dir / f"{img_path.stem}_vis{img_path.suffix}"
        
        # Если модель загружена — можно получить Results напрямую
        if model is not None:
            # Повторный инференс для получения Results объектов
            results = model.predict(
                source=str(img_path),
                conf=cfg["inference"]["conf"],
                imgsz=cfg["training"]["imgsz"],
                verbose=False
            )
            if results and len(results) > 0:
                if visualize_result(img_path, results[0], output_path, class_names):
                    success += 1
        else:
            # Без модели — просто копируем изображение (fallback)
            import shutil
            shutil.copy2(img_path, output_path)
            logger.info(f"Скопировано (без модели): {output_path}")
            success += 1
    
    logger.info(f"\n✅ Визуализация завершена: {success}/{len(pred_images)}")
    logger.info(f"📁 Результаты: {vis_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Визуализация результатов инференса YOLO")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--sample", type=int, default=None, help="Количество изображений")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    visualize_inference_results(cfg, sample_size=args.sample)


if __name__ == "__main__":
    main()