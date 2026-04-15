#!/usr/bin/env python3
"""
Training script for YOLOv26-GPR with pretrained YOLOv26s backbone + DINOv3.

Loads yolo26s.pt pretrained weights into compatible CNN layers, then trains
with DINOv3 frozen.  CrossFusion layers and DINOv3FPN are always random-init
(no equivalent in yolo26s.pt).

Usage:
    python train_gpr_pretrained.py
    python train_gpr_pretrained.py --scale s --epochs 150 --batch 6
    python train_gpr_pretrained.py --scale m --epochs 150 --batch 4
"""

import argparse
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import set_model_scale

warnings.filterwarnings("ignore")

DATA_CONFIG  = "data_all.yaml"
MODEL_CONFIG = "ultralytics/cfg/models/v26/yolov26_gpr.yaml"
# auto-map scale → pretrained .pt (override with --pretrained if needed)
PRETRAINED_MAP = {
    "n": "yolo26n.pt",
    "s": "yolo26s.pt",
    "m": "yolo26m.pt",
    "l": "yolo26l.pt",
    "x": "yolo26x.pt",
}


def load_pretrained_backbone(our_model, pretrained_path: str) -> int:
    """
    Partial-load yolo26s.pt weights into our yolov26_gpr model.
    Only copies parameters whose key name AND shape match exactly.
    Returns number of matched layers.
    """
    print(f"\nLoading pretrained backbone from: {pretrained_path}")
    try:
        pretrained_sd = YOLO(pretrained_path).model.state_dict()
    except FileNotFoundError:
        fallback = "yolo26s.pt"
        print(f"  ⚠️  {pretrained_path} not found — falling back to {fallback}")
        pretrained_sd = YOLO(fallback).model.state_dict()

    our_sd = our_model.state_dict()
    matched, skipped = {}, []

    for k, v in pretrained_sd.items():
        if k in our_sd:
            if our_sd[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(f"  shape mismatch  {k}: pretrained={tuple(v.shape)} ours={tuple(our_sd[k].shape)}")
        else:
            skipped.append(f"  key not found   {k}")

    our_sd.update(matched)
    our_model.load_state_dict(our_sd)

    n_total = len(our_sd)
    n_matched = len(matched)
    print(f"  Matched  : {n_matched}/{n_total} layers  ({100*n_matched/n_total:.1f}%)")
    print(f"  Skipped  : {len(skipped)} layers (DINOv3FPN, CrossFusion, 9ch stem — expected)")
    return n_matched


def train(scale: str, epochs: int, batch: int, imgsz: int, data: str, pretrained: str = None):
    print("=" * 60)
    print(f"scale={scale}  epochs={epochs}  batch={batch}  imgsz={imgsz}")
    print("YOLOv26s pretrained + DINOv3 frozen — training CNN + CrossFusion")
    print("=" * 60)

    # 1. Build model from YAML
    set_model_scale(scale)
    model = YOLO(MODEL_CONFIG)

    # 2. Partial-load pretrained backbone (auto-select by scale if not specified)
    pt = pretrained or PRETRAINED_MAP.get(scale, "yolo26s.pt")
    load_pretrained_backbone(model.model, pt)

    # 3. Train
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,          # ต่ำกว่า scratch เล็กน้อย เพราะ backbone pretrained แล้ว
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=3,    # warmup สั้นลง backbone ไม่ต้อง warm นาน

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation (GPR-safe)
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
        name=f"train_{scale}_pretrained",
        exist_ok=True,
        plots=True,
        save_period=20,
        patience=50,
    )

    best = Path(f"runs/gpr/train_{scale}_pretrained/weights/best.pt")
    print(f"\nDone. Best weights: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv26-GPR with pretrained backbone")
    parser.add_argument("--scale",  default="s", choices=["n", "s", "m", "l"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch",  type=int, default=6)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--data",   default=DATA_CONFIG)
    parser.add_argument("--pretrained", default=None,
                        help="Path or hub name for yolo26s weights")
    args = parser.parse_args()

    train(args.scale, args.epochs, args.batch, args.imgsz, args.data, args.pretrained)


if __name__ == "__main__":
    main()
