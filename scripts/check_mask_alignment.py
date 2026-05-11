#!/usr/bin/env python3
"""
Проверка совпадения маски YOLO с оригинальным изображением.
Скрипт рисует полупрозрачную маску поверх фото и сохраняет результат.
"""

import os
import sys
import yaml
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# Настройка логгера
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_single_image(model, image_path: Path, output_dir: Path, conf: float = 0.5):
    """
    Проверка маски для одного изображения.
    """
    logger.info(f"🔍 Проверка: {image_path.name}")
    
    # 1. Чтение оригинала
    img_original = cv2.imread(str(image_path))
    if img_original is None:
        logger.error(f"❌ Не удалось прочитать изображение")
        return

    h_img, w_img = img_original.shape[:2]
    
    # 2. Инференс
    # Важно: verbose=False, чтобы не засорять консоль таблицами
    results = model.predict(str(image_path), conf=conf, imgsz=640, verbose=False)
    
    if not results or not results[0].masks:
        logger.warning(f"⚠️ Маски не найдены (conf={conf})")
        return

    # 3. Обработка каждой маски (если на фото несколько диаграмм)
    for i, mask_obj in enumerate(results[0].masks):
        # Получаем маску как numpy array (H, W)
        mask = mask_obj.data[0].cpu().numpy()
        
        #  КРИТИЧЕСКИ ВАЖНО: Приводим маску к размеру оригинала
        # YOLO может вернуть маску 640x640, а фото быть 1024x768
        if mask.shape[0] != h_img or mask.shape[1] != w_img:
            logger.debug(f"   Ресайз маски: {mask.shape} -> {img_original.shape[:2]}")
            # INTER_NEAREST важен для бинарных масок, чтобы не было размытия на границах
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        
        # Бинаризация (на всякий случай)
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # --- Визуализация ---
        
        # А) Создаем цветную маску (Зеленый цвет для наглядности)
        # BGR формат для OpenCV: (0, 255, 0)
        color_mask = np.zeros_like(img_original)
        color_mask[mask_binary == 1] = [0, 255, 0] 
        
        # Б) Рисуем контур (Белый, толщина 3)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_mask, contours, -1, (255, 255, 255), thickness=3)
        
        # В) Накладываем на оригинал (Alpha blending)
        # alpha=0.5 делает маску полупрозрачной
        overlay = cv2.addWeighted(img_original, 1.0, color_mask, 0.5, 0)
        
        # Г) Сохранение
        out_filename = f"{image_path.stem}_mask_check_{i}.jpg"
        out_path = output_dir / out_filename
        cv2.imwrite(str(out_path), overlay)
        
        logger.info(f"   ✅ Сохранено: {out_filename} (Пикселей в маске: {np.sum(mask_binary)})")

def main():
    parser = argparse.ArgumentParser(description="Проверка выравнивания маски YOLO")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Путь к best.pt")
    parser.add_argument("--source", type=str, default=None, help="Папка с картинками")
    parser.add_argument("--conf", type=float, default=0.5, help="Порог уверенности")
    args = parser.parse_args()

    # Загрузка конфига
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    # Пути
    weights_path = args.weights or cfg.get("clustering", {}).get("weights_path", "weights/best.pt")
    source_dir = args.source or cfg.get("clustering", {}).get("source_dir", "data/test_images")
    output_dir = Path("outputs/mask_check")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка модели
    if not Path(weights_path).exists():
        logger.error(f"❌ Файл весов не найден: {weights_path}")
        return
        
    logger.info(f"🚀 Загрузка модели: {weights_path}")
    model = YOLO(weights_path)
    
    # Обработка изображений
    images = sorted(Path(source_dir).glob("*"))
    images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    
    if not images:
        logger.error(f"❌ Изображения не найдены в {source_dir}")
        return

    logger.info(f"📊 Найдено {len(images)} изображений")
    
    for img_path in images:
        check_single_image(model, img_path, output_dir, args.conf)
        
    logger.info(f" Готово. Результаты в: {output_dir}")

if __name__ == "__main__":
    main()