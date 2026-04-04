#!/usr/bin/env python3
"""
Training script for YOLOv26 Triple Input with DINOv3 backbone.

This script trains YOLOv26 Triple Input models with DINOv3 feature extraction
for enhanced performance in civil engineering applications.

Key differences from YOLOv26:
  - Uses C2PSA instead of A2C2f attention blocks
  - Uses SPPF with improved parameters
  - Backbone: Conv → C3k2 → SPPF → C2PSA pipeline

Usage:
    # P0 DINOv3 input preprocessing only
    python train_triple_dinov3_v26.py --data dataset.yaml --integrate initial

    # No DINOv3 integration (standard triple input)
    python train_triple_dinov3_v26.py --data dataset.yaml --integrate nodino

    # P3 DINOv3 feature enhancement only (after P3 stage)
    python train_triple_dinov3_v26.py --data dataset.yaml --integrate p3

    # Dual DINOv3 integration (P0 preprocessing + P3 enhancement)
    python train_triple_dinov3_v26.py --data dataset.yaml --integrate p0p3

    # With different DINOv3 sizes
    python train_triple_dinov3_v26.py --data dataset.yaml --dinov3-size base --freeze-dinov3
    python train_triple_dinov3_v26.py --data dataset.yaml --dinov3-size large --variant s
"""

import argparse
import yaml
import torch
from pathlib import Path
import warnings
from ultralytics import YOLO


# Map DINOv3 size names to HuggingFace model IDs
DINOV3_MODEL_MAP = {
    "small":      "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base":       "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large":      "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "giant":      "facebook/dinov3-vitg16-pretrain-lvd1689m",
    "sat_large":  "facebook/dinov3-vitl16-pretrain-sat493m",
    "sat_giant":  "facebook/dinov3-vitg16-pretrain-sat493m",
}

# YOLOv26 model config paths
MODEL_CONFIGS = {
    "nodino":  "ultralytics/cfg/models/v26/yolo26_triple.yaml",
    "initial": "ultralytics/cfg/models/v26/yolo26_triple_dinov3.yaml",
    "p3":      "ultralytics/cfg/models/v26/yolo26_triple_dinov3_p3.yaml",
    "p0p3":    "ultralytics/cfg/models/v26/yolo26_triple_dinov3_p0p3.yaml",
}


def train_triple_dinov3_v26(
    data_config: str,
    dinov3_size: str = "base",
    freeze_dinov3: bool = True,
    pretrained_path: str = None,
    epochs: int = 100,
    batch_size: int = 8,
    imgsz: int = 224,
    patience: int = 50,
    name: str = "yolo26_triple_dinov3",
    device: str = "0",
    integrate: str = "initial",
    variant: str = "s",
    save_period: int = -1,
    **kwargs
):
    """
    Train YOLOv26 Triple Input with DINOv3 backbone.

    Args:
        data_config:     Path to dataset YAML configuration
        dinov3_size:     DINOv3 model size (small, base, large, giant, sat_large, sat_giant)
        freeze_dinov3:   Whether to freeze DINOv3 backbone weights
        pretrained_path: Path to pretrained YOLOv26 model (optional)
        epochs:          Number of training epochs
        batch_size:      Batch size (smaller recommended due to DINOv3 memory usage)
        imgsz:           Input image size (224 recommended for DINOv3)
        patience:        Early stopping patience
        name:            Experiment name (saved under runs/detect/)
        device:          CUDA device ("0", "0,1", "cpu")
        integrate:       DINOv3 integration strategy:
                           "initial" - P0 input preprocessing (DINOv3 replaces TripleInputConv)
                           "nodino"  - No DINOv3, standard TripleInputConv only
                           "p3"      - DINOv3 enhancement after P3 backbone stage
                           "p0p3"    - Dual integration: P0 + P3
        variant:         YOLOv26 model scale (n, s, m, l, x)
        save_period:     Save weights every N epochs (-1 = best/last only)
        **kwargs:        Additional YOLO training arguments

    Returns:
        Training results object
    """

    print("=" * 60)
    print("  YOLOv26 Triple Input + DINOv3 Training")
    print("=" * 60)
    print(f"  Data Config    : {data_config}")
    print(f"  YOLOv26 Variant: {variant}")
    print(f"  DINOv3 Size    : {dinov3_size}")
    print(f"  Integration    : {integrate}")
    print(f"  Freeze DINOv3  : {freeze_dinov3}")
    print(f"  Pretrained     : {pretrained_path or 'None (train from scratch)'}")
    print(f"  Epochs         : {epochs}")
    print(f"  Batch Size     : {batch_size}")
    print(f"  Image Size     : {imgsz}")
    print(f"  Device         : {device}")
    print(f"  Save Period    : {'Best/Last only' if save_period == -1 else f'Every {save_period} epochs'}")
    print("-" * 60)

    # ── Step 1: Validate integration strategy ────────────────────────────────
    if integrate not in MODEL_CONFIGS:
        print(f"ERROR: Unknown integration strategy '{integrate}'")
        print(f"       Available options: {list(MODEL_CONFIGS.keys())}")
        return None

    model_config = MODEL_CONFIGS[integrate]
    if not Path(model_config).exists():
        print(f"ERROR: Model config not found: {model_config}")
        return None

    # ── Step 2: Check DINOv3 dependencies ────────────────────────────────────
    if integrate != "nodino":
        try:
            import transformers  # noqa: F401
            import timm          # noqa: F401
            print("Step 1: DINOv3 packages available (transformers, timm)")
        except ImportError as e:
            print(f"ERROR: Missing required packages: {e}")
            print("       Install with: pip install transformers timm huggingface_hub")
            return None

        # HuggingFace authentication check
        try:
            from ultralytics.nn.modules.dinov3 import setup_huggingface_auth
            auth_success, auth_source = setup_huggingface_auth()
            if not auth_success:
                print("WARNING: HuggingFace authentication not configured.")
                print("         DINOv3 models may fail to download without a token.")
                print("         Set HUGGINGFACE_HUB_TOKEN env var or run: huggingface-cli login")
        except Exception as e:
            print(f"WARNING: Auth setup warning: {e}")
    else:
        print("Step 1: Skipping DINOv3 setup (nodino mode)")

    # ── Step 3: Build model configuration ────────────────────────────────────
    print(f"\nStep 2: Loading model config for integration='{integrate}'...")

    with open(model_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["scale"] = variant
    config["ch"] = 9  # Triple input: 3 RGB images concatenated

    if integrate != "nodino":
        dino_model_name = DINOV3_MODEL_MAP.get(dinov3_size, DINOV3_MODEL_MAP["base"])
        print(f"         DINOv3 model: {dino_model_name}")
        print(f"         DINOv3 frozen: {freeze_dinov3}")

        if integrate == "initial":
            # P0 only: update DINOv3 model name and freeze setting in backbone[0]
            config["backbone"][0][-1][0] = dino_model_name
            config["backbone"][0][-1][3] = freeze_dinov3

        elif integrate == "p3":
            # P3 only: update P3FeatureEnhancer output channels by variant width
            width_scale = {"n": 0.25, "s": 0.5, "m": 1.0, "l": 1.0, "x": 1.5}.get(variant, 1.0)
            config["backbone"][5][-1][1] = 256  # base output channels (YOLO will scale)

        elif integrate == "p0p3":
            # Dual P0 + P3: update both DINOv3 entries
            config["backbone"][0][-1][0] = dino_model_name   # P0 model name
            config["backbone"][0][-1][3] = freeze_dinov3      # P0 freeze
            config["backbone"][4][-1][0] = dino_model_name   # P3 model name
            config["backbone"][4][-1][3] = freeze_dinov3      # P3 freeze

    # Write temporary config with scale embedded
    temp_config = f"temp_yolo26_triple_{integrate}_{variant}_{dinov3_size}.yaml"
    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"         Temp config saved: {temp_config}")

    # ── Step 4: Initialize model ──────────────────────────────────────────────
    print("\nStep 3: Initializing YOLOv26 model...")
    try:
        if pretrained_path:
            print(f"         Loading pretrained weights from: {pretrained_path}")
            from load_pretrained_triple import load_pretrained_weights_to_triple_model
            model = load_pretrained_weights_to_triple_model(
                pretrained_path=pretrained_path,
                triple_model_config=temp_config
            )
        else:
            model = YOLO(temp_config, task="detect")

        # Force scale in model yaml
        if hasattr(model, "model") and hasattr(model.model, "yaml"):
            model.model.yaml["scale"] = variant

        print("         Model initialized successfully")

    except Exception as e:
        print(f"ERROR: Model initialization failed: {e}")
        Path(temp_config).unlink(missing_ok=True)
        return None

    # ── Step 5: Configure training hyperparameters ────────────────────────────
    print("\nStep 4: Starting training...")

    # Augmentation restrictions for triple-channel input
    train_kwargs = dict(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        name=name,
        device=device,
        save_period=save_period,
        # Disable augmentations incompatible with 9-channel input
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        # Smaller learning rate for DINOv3 fine-tuning
        lr0=0.001 if integrate != "nodino" else 0.01,
        lrf=0.01,
        warmup_epochs=3,
    )
    train_kwargs.update(kwargs)

    try:
        results = model.train(**train_kwargs)
        print("\nTraining complete!")
        return results
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        raise
    finally:
        # Clean up temp config
        Path(temp_config).unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv26 Triple Input with DINOv3 backbone"
    )
    parser.add_argument("--data",         type=str, required=True,
                        help="Path to dataset YAML config")
    parser.add_argument("--integrate",    type=str, default="initial",
                        choices=["initial", "nodino", "p3", "p0p3"],
                        help="DINOv3 integration strategy")
    parser.add_argument("--dinov3-size",  type=str, default="base",
                        choices=list(DINOV3_MODEL_MAP.keys()),
                        help="DINOv3 model size")
    parser.add_argument("--freeze-dinov3", action="store_true", default=True,
                        help="Freeze DINOv3 backbone (default: True)")
    parser.add_argument("--no-freeze",    dest="freeze_dinov3", action="store_false",
                        help="Unfreeze DINOv3 backbone for full fine-tuning")
    parser.add_argument("--pretrained",   type=str, default=None,
                        help="Path to pretrained YOLOv26 weights (optional)")
    parser.add_argument("--epochs",       type=int, default=100)
    parser.add_argument("--batch",        type=int, default=8)
    parser.add_argument("--imgsz",        type=int, default=224)
    parser.add_argument("--patience",     type=int, default=50)
    parser.add_argument("--name",         type=str, default="yolo26_triple_dinov3")
    parser.add_argument("--device",       type=str, default="0")
    parser.add_argument("--variant",      type=str, default="s",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv26 model scale")
    parser.add_argument("--save-period",  type=int, default=-1,
                        help="Save weights every N epochs (-1 = best/last only)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_triple_dinov3_v26(
        data_config=args.data,
        dinov3_size=args.dinov3_size,
        freeze_dinov3=args.freeze_dinov3,
        pretrained_path=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        name=args.name,
        device=args.device,
        integrate=args.integrate,
        variant=args.variant,
        save_period=args.save_period,
    )
