#!/usr/bin/env python3
"""
Training script for YOLOv26-GPR (DINOv3 Multi-Scale Cross-Attention)
Task: GPR subsurface void (Cavity) detection

Usage:
    python train_gpr.py
    python train_gpr.py --scale s --epochs 150 --batch 8
    python train_gpr.py --scale m --epochs 200 --batch 4 --imgsz 640
"""

import argparse
import warnings
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import set_model_scale

warnings.filterwarnings("ignore")

DATA_CONFIG  = "data_all.yaml"
MODEL_CONFIG = "ultralytics/cfg/models/v26/yolov26_gpr.yaml"


def train(scale: str, epochs: int, batch: int, imgsz: int, data: str):
    print("=" * 60)
    print(f"scale={scale}  epochs={epochs}  batch={batch}  imgsz={imgsz}")
    print("DINOv3 ViT frozen — training CNN + CrossFusion")
    print("=" * 60)

    set_model_scale(scale)
    model = YOLO(MODEL_CONFIG)

    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        # Optimizer
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=5,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation (GPR-safe)
        fliplr=0.5,
        flipud=0.0,       # DO NOT flip vertically — destroys GPR depth order
        degrees=0.0,      # DO NOT rotate — destroys hyperbola shape
        translate=0.1,
        scale=0.3,
        mosaic=1.0,
        copy_paste=0.3,
        hsv_h=0.0,        # GPR is not color-sensitive
        hsv_s=0.0,
        hsv_v=0.2,

        # Misc
        project="runs/gpr",
        name=f"train_{scale}",
        exist_ok=True,
        plots=True,
        save_period=20,
        patience=50,
    )

    best = Path(f"runs/gpr/train_{scale}/weights/best.pt")
    print(f"\nDone. Best weights: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv26-GPR void detector")
    parser.add_argument("--scale",  default="s", choices=["n", "s", "m", "l"],
                        help="Model scale (default: s)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--data",   default=DATA_CONFIG)
    args = parser.parse_args()

    train(args.scale, args.epochs, args.batch, args.imgsz, args.data)


if __name__ == "__main__":
    main()
