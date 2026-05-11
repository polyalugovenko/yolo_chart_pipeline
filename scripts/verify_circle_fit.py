#!/usr/bin/env python3
"""
Проверка: оригинальная маска YOLO vs аппроксимированный идеальный круг.
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def mask_to_perfect_circle(mask: np.ndarray) -> tuple:
    mask_bin = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(round(x)), int(round(y)))
    radius = int(round(radius))
    h, w = mask.shape[:2]
    perfect = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(perfect, center, radius, 255, -1)
    return center, radius, perfect

def verify_fit(model, image_path: Path, output_dir: Path, conf: float = 0.5):
    img = cv2.imread(str(image_path))
    if img is None: return
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model.predict(str(image_path), conf=conf, imgsz=640, verbose=False)
    if not results or not results[0].masks:
        logger.warning(f"⚠️ Нет масок: {image_path.name}")
        return
        
    for i, mask_obj in enumerate(results[0].masks):
        mask = mask_obj.data[0].cpu().numpy()
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        circle_res = mask_to_perfect_circle(mask)
        if circle_res is None: continue
        (cx, cy), r, perfect_mask = circle_res
        
        # --- Визуализация сравнения ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Оригинал
        axes[0].imshow(img_rgb)
        axes[0].set_title("Оригинал")
        axes[0].axis('off')
        
        # 2. Оригинальная маска (зелёная)
        overlay1 = img_rgb.copy()
        overlay1[mask > 0.5] = [0, 255, 0]
        axes[1].imshow(overlay1)
        axes[1].set_title(f"Маска YOLO ({np.sum(mask>0.5)} px)")
        axes[1].axis('off')
        
        # 3. Идеальный круг (синий)
        overlay2 = img_rgb.copy()
        overlay2[perfect_mask == 255] = [0, 0, 255]
        axes[2].imshow(overlay2)
        axes[2].set_title(f"Идеальный круг (r={r}, центр=({cx},{cy}))")
        axes[2].axis('off')
        
        plt.tight_layout()
        out_path = output_dir / f"{image_path.stem}_circle_fit_{i}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"✅ Сохранено сравнение: {out_path.name}")
        
        # IoU для оценки качества аппроксимации
        inter = np.logical_and(mask > 0.5, perfect_mask == 255).sum()
        union = np.logical_or(mask > 0.5, perfect_mask == 255).sum()
        iou = inter / union if union > 0 else 0
        logger.info(f"   IoU маска↔круг: {iou:.3f} {'✅ Отлично' if iou > 0.85 else '⚠️ Есть расхождения'}")

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", default="data/test_images")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()
    
    out_dir = Path("outputs/circle_verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(args.weights)
    images = sorted(Path(args.source).glob("*"))
    images = [f for f in images if f.suffix.lower() in {".jpg",".jpeg",".png"}]
    
    for img in images:
        verify_fit(model, img, out_dir, args.conf)
        
    print(f"\n📁 Все сравнения сохранены в: {out_dir}")

if __name__ == "__main__":
    main()