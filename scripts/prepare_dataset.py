import os
import shutil
import random
import uuid
import yaml
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def prepare_dataset(cfg):
    src_root = Path(cfg["dataset"]["source_root"])
    dest_root = Path(cfg["dataset"]["dest_root"])
    img_exts = set(cfg["dataset"]["img_extensions"])
    
    dest_images = dest_root / "images"
    dest_labels = dest_root / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    
    # Индексация разметки
    label_dir = src_root / "labels"
    if not label_dir.exists():
        logging.error(f"Директория разметки не найдена: {label_dir}")
        return
    label_index = {p.stem: p for p in label_dir.glob("*.txt")}
    
    pairs = []
    img_dir = src_root / "images"
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() in img_exts:
            if img_path.stem in label_index:
                pairs.append((img_path, label_index[img_path.stem], img_path.suffix))
            else:
                logging.warning(f"Нет разметки для {img_path.name}")
                
    logging.info(f"Найдено валидных пар: {len(pairs)}")
    
    random.seed(cfg["seed"])
    random.shuffle(pairs)
    
    for img_path, lbl_path, ext in pairs:
        new_name = str(uuid.uuid4())
        shutil.copy2(img_path, dest_images / f"{new_name}{ext}")
        shutil.copy2(lbl_path, dest_labels / f"{new_name}.txt")
        
    # Генерация data.yaml
    data_cfg = {
        "path": str(dest_root.resolve()),
        "train": "images",
        "val": "images",
        "nc": 1,
        "names": ["pie_chart"]
    }
    yaml_path = dest_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
        
    logging.info(f"✅ Датасет подготовлен: {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prepare_dataset(cfg)