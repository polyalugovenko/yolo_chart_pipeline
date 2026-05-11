import os
import yaml
import shutil
import logging
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import KFold
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_kfold(cfg):
    task = cfg["training"]["task"]
    model_name = cfg["training"]["model"]
    batch = cfg["training"]["batch"]
    imgsz = cfg["training"]["imgsz"]
    n_folds = cfg["training"]["n_folds"]
    project = Path(cfg["training"]["project"])
    aug = cfg["training"]["augmentation"]
    
    data_root = Path(cfg["dataset"]["dest_root"])
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    base_yaml = data_root / "data.yaml"
    
    with open(base_yaml, "r") as f:
        base_cfg = yaml.safe_load(f)
    class_names = base_cfg.get("names", [])
    
    image_files = sorted(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    logging.info(f"📊 Всего изображений: {len(image_files)}")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg["seed"])
    fold_metrics = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.get("device") == "cpu":
        device = "cpu"
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        epochs = cfg["training"]["epochs"]
        fold_dir = project / f"fold_{fold}"
        fold_images = fold_dir / "images"
        fold_labels = fold_dir / "labels"
        
        for split, indices in [("train", train_idx), ("val", val_idx)]:
            (fold_images / split).mkdir(parents=True, exist_ok=True)
            (fold_labels / split).mkdir(parents=True, exist_ok=True)
            for idx in indices:
                img_path = image_files[idx]
                lbl_path = labels_dir / f"{img_path.stem}.txt"
                shutil.copy2(img_path, fold_images / split / img_path.name)
                if lbl_path.exists():
                    shutil.copy2(lbl_path, fold_labels / split / lbl_path.name)
                    
        fold_yaml_path = fold_dir / "data.yaml"
        with open(fold_yaml_path, "w") as f:
            yaml.dump({
                "path": str(fold_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": len(class_names),
                "names": class_names
            }, f, default_flow_style=False)
            
        logging.info(f"🚀 Фолд {fold+1}/{n_folds} | Запуск обучения...")
        model = YOLO(model_name)
        results = model.train(
            data=str(fold_yaml_path),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=str(project / f"fold_{fold}_runs"),
            name="train",
            exist_ok=True,
            verbose=False,
            save=True,
            device=device,
            mosaic=aug["mosaic"],
            mixup=aug["mixup"],
            perspective=aug["perspective"],
            shear=aug["shear"]
        )
        
        metrics = results.results_dict
        fold_metrics.append({
            "fold": fold + 1,
            "metrics": metrics
        })
        map_key = f"metrics/mAP50-95(B)"
        logging.info(f"✅ Фолд {fold+1} завершён. mAP50-95: {metrics.get(map_key, 'N/A'):.4f}")
        
    # Агрегация
    logging.info("\n" + "="*50)
    logging.info("📈 СВОДНЫЕ РЕЗУЛЬТАТЫ")
    logging.info("="*50)
    summary = {}
    for key in ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"]:
        vals = [fm["metrics"].get(key, 0) for fm in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_fold": vals}
        logging.info(f"{key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")
        
    report_path = project / "kfold_summary.yaml"
    with open(report_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    logging.info(f"💾 Отчёт сохранён: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_kfold(cfg)