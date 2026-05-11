import os
import yaml
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_inference(cfg):
    weights = cfg["inference"]["weights"]
    conf = cfg["inference"]["conf"]
    source = Path(cfg["inference"]["source"])
    output_dir = Path(cfg["inference"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(weights).exists():
        raise FileNotFoundError(f"Веса не найдены: {weights}")
        
    model = YOLO(weights)
    logging.info(f"🚀 Загрузка модели: {weights}")
    
    images = sorted(source.glob("*"))
    images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    
    results = model.predict(
        source=[str(p) for p in images],
        conf=conf,
        imgsz=cfg["training"]["imgsz"],
        save=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
        verbose=True
    )
    
    total = 0
    class_dist = {}
    for res in results:
        n = len(res.boxes) if res.boxes is not None else 0
        total += n
        if res.boxes is not None:
            for cls_id in res.boxes.cls.cpu().numpy():
                name = model.names[int(cls_id)]
                class_dist[name] = class_dist.get(name, 0) + 1
                
    report = {
        "weights": weights,
        "confidence": conf,
        "total_images": len(images),
        "total_objects": total,
        "class_distribution": class_dist
    }
    report_path = output_dir / "inference_report.yaml"
    with open(report_path, "w", encoding="utf-8") as f:
        yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
    logging.info(f" Всего объектов: {total} | Отчёт: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_inference(cfg)