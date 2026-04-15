#!/usr/bin/env python3
"""
Training script for YOLOv26-GPR (DINOv3 Multi-Scale Cross-Attention)
Task: GPR subsurface void (Cavity) detection
Dataset: ~2700 images, triple 9-channel input, 3 scan orientations

Usage:
    python train_gpr.py                          # default: scale s, 100 epochs
    python train_gpr.py --scale m                # larger model
    python train_gpr.py --epochs 150 --batch 8
    python train_gpr.py --phase2                 # phase 2: unfreeze DINOv3
"""

import argparse
import warnings
from pathlib import Path
from ultralytics import YOLO

warnings.filterwarnings("ignore")

DATA_CONFIG  = "data_all.yaml"
MODEL_CONFIG = "ultralytics/cfg/models/v26/yolov26_gpr.yaml"


def phase1(scale: str, epochs: int, batch: int, imgsz: int):
    """Phase 1: DINOv3 frozen — train CNN + CrossFusion layers only."""
    print("=" * 60)
    print(f"Phase 1  |  scale={scale}  epochs={epochs}  batch={batch}")
    print("DINOv3 ViT frozen — training CNN + CrossFusion")
    print("=" * 60)

    model = YOLO(MODEL_CONFIG)

    model.train(
        data=DATA_CONFIG,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        # Optimizer
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=5,

        # Loss weights  (single class: focus on box quality)
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation  (GPR-safe: no vertical flip, no large rotation)
        fliplr=0.5,
        flipud=0.0,         # DO NOT flip vertically — destroys GPR depth order
        degrees=0.0,        # DO NOT rotate — destroys hyperbola shape
        translate=0.1,
        scale=0.3,
        mosaic=1.0,
        copy_paste=0.3,     # paste extra void patches (helps rare class)
        hsv_h=0.0,          # GPR is not color-sensitive
        hsv_s=0.0,
        hsv_v=0.2,          # slight brightness variation OK

        # Misc
        project="runs/gpr",
        name=f"phase1_{scale}",
        exist_ok=True,
        plots=True,
        save_period=20,
        patience=30,
    )

    best = Path(f"runs/gpr/phase1_{scale}/weights/best.pt")
    print(f"\nPhase 1 done. Best weights: {best}")
    return best


def phase2(weights: str, epochs: int, batch: int, imgsz: int, scale: str):
    """Phase 2: Unfreeze DINOv3 — fine-tune entire model with low LR."""
    print("=" * 60)
    print(f"Phase 2  |  fine-tuning from {weights}")
    print("DINOv3 ViT UNFROZEN — fine-tuning all layers")
    print("=" * 60)

    model = YOLO(weights)

    # Unfreeze DINOv3 ViT weights
    from ultralytics.nn.modules import DINOv3FPN
    for layer in model.model.model:
        if isinstance(layer, DINOv3FPN):
            layer.unfreeze = True
            for p in layer.dino_model.parameters():
                p.requires_grad = True
            print(f"  Unfroze DINOv3FPN — {sum(p.numel() for p in layer.dino_model.parameters()):,} params now trainable")

    model.train(
        data=DATA_CONFIG,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        # Much lower LR for fine-tuning
        optimizer="AdamW",
        lr0=0.0002,          # 10x lower than phase 1
        lrf=0.01,
        weight_decay=0.005,
        warmup_epochs=3,

        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Same GPR-safe augmentation
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.3,
        mosaic=0.5,          # reduce mosaic in phase 2
        copy_paste=0.1,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,

        project="runs/gpr",
        name=f"phase2_{scale}",
        exist_ok=True,
        plots=True,
        save_period=10,
        patience=20,
    )

    best = Path(f"runs/gpr/phase2_{scale}/weights/best.pt")
    print(f"\nPhase 2 done. Best weights: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv26-GPR void detector")
    parser.add_argument("--scale",    default="s",   choices=["n","s","m","l"],
                        help="Model scale (s recommended for ~2700 imgs)")
    parser.add_argument("--epochs",   type=int, default=100, help="Epochs for phase 1")
    parser.add_argument("--batch",    type=int, default=16,  help="Batch size")
    parser.add_argument("--imgsz",    type=int, default=640, help="Image size")
    parser.add_argument("--phase2",   action="store_true",   help="Run phase 2 after phase 1")
    parser.add_argument("--weights",  default=None,
                        help="Start phase 2 from existing weights (skip phase 1)")
    parser.add_argument("--data",     default=None,          help="Dataset yaml path")
    args = parser.parse_args()

    if args.data:
        global DATA_CONFIG
        DATA_CONFIG = args.data

    if args.weights:
        # Jump straight to phase 2
        phase2(args.weights, epochs=50, batch=args.batch,
               imgsz=args.imgsz, scale=args.scale)
    else:
        best = phase1(args.scale, args.epochs, args.batch, args.imgsz)
        if args.phase2 and best.exists():
            phase2(str(best), epochs=50, batch=max(1, args.batch // 2),
                   imgsz=args.imgsz, scale=args.scale)


if __name__ == "__main__":
    main()
