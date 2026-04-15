#!/usr/bin/env python3
"""
Fine-tune YOLOv26-GPR from best.pt checkpoint.

Changes vs original training:
  - Loads best.pt as starting point (mAP50=0.287 @ epoch 88)
  - Unfreezes DINOv3 last 3 transformer layers (9-11) + norm
  - Lower lr=0.0001 to avoid catastrophic forgetting
  - cos_lr=True for smoother convergence

Usage:
    python train_gpr_finetune.py
    python train_gpr_finetune.py --best "C:/Users/USER/Downloads/best (1).pt"
    python train_gpr_finetune.py --best runs/gpr/train_m_pretrained/weights/best.pt
"""

import argparse
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")

DATA_CONFIG   = "data_all.yaml"
DEFAULT_BEST  = "runs/best (1).pt"


def partial_unfreeze_dinov3(model, n_unfreeze: int = 3):
    """
    Unfreeze last n_unfreeze transformer layers of DINOv3 + final norm.
    Keeps all other DINOv3 layers frozen.
    """
    dino_fpn = model.model.model[0]   # DINOv3FPN is always layer 0

    # Count total transformer layers
    layers = dino_fpn.dino_model.model.layer
    total  = len(layers)
    unfreeze_idx = list(range(total - n_unfreeze, total))

    print(f"\nPartial DINOv3 unfreeze: layers {unfreeze_idx} + norm")

    for i in unfreeze_idx:
        for p in layers[i].parameters():
            p.requires_grad = True
        print(f"  Unfrozen: layer {i}")

    # Unfreeze final norm
    for p in dino_fpn.dino_model.norm.parameters():
        p.requires_grad = True
    print("  Unfrozen: norm")

    # Allow grad in DINOv3FPN forward pass
    dino_fpn.freeze = False

    # Count trainable params
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.model.parameters())
    print(f"  Trainable: {trainable:,} / {total_p:,} params ({100*trainable/total_p:.1f}%)\n")


def finetune(best_pt: str, epochs: int, batch: int, imgsz: int, data: str, n_unfreeze: int):
    best_path = Path(best_pt)
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt not found: {best_path}")

    print("=" * 60)
    print(f"Fine-tuning from: {best_path.name}")
    print(f"epochs={epochs}  batch={batch}  imgsz={imgsz}")
    print(f"DINOv3 unfreeze last {n_unfreeze} layers  lr=0.0001  cos_lr")
    print("=" * 60)

    # 1. Load checkpoint
    model = YOLO(str(best_path))

    # 2. Partially unfreeze DINOv3
    partial_unfreeze_dinov3(model, n_unfreeze=n_unfreeze)

    # 3. Fine-tune
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        # Optimizer — low lr to preserve learned features
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=3,
        cos_lr=True,

        # Loss weights (same as original)
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation (GPR-safe, same as original)
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.3,
        mosaic=1.0,
        copy_paste=0.3,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,

        # Misc
        project="runs/gpr",
        name="finetune_m_dino_partial",
        exist_ok=True,
        plots=True,
        save_period=20,
        patience=50,
    )

    best_out = Path("runs/gpr/finetune_m_dino_partial/weights/best.pt")
    print(f"\nDone. Best weights: {best_out}")
    return best_out


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv26-GPR from best.pt")
    parser.add_argument("--best",      default=DEFAULT_BEST, help="Path to best.pt checkpoint")
    parser.add_argument("--epochs",    type=int, default=150)
    parser.add_argument("--batch",     type=int, default=6)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--data",      default=DATA_CONFIG)
    parser.add_argument("--n_unfreeze", type=int, default=3,
                        help="Number of DINOv3 transformer layers to unfreeze from the end (default 3)")
    args = parser.parse_args()

    finetune(args.best, args.epochs, args.batch, args.imgsz, args.data, args.n_unfreeze)


if __name__ == "__main__":
    main()
