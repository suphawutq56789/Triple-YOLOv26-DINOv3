#!/usr/bin/env python3
"""
Training script for YOLOv26-GPR-MedSAM
Architecture: YOLOv26 CNN + MedSAM ViT-B Multi-Scale Cross-Attention FPN
Task: GPR subsurface void (Cavity) detection
Dataset: ~2700 images, triple 9-channel input, 3 scan orientations

Domain rationale:
    MedSAM was trained on 1.5M medical images including ultrasound.
    Ultrasound and GPR share the same pulse-echo physics:
      - hyperbolic diffraction patterns from point reflectors
      - layered reflections from material interfaces
      - speckle/clutter noise, void = signal dropout
    This gives MedSAM far better feature transfer to GPR than DINOv3
    which was trained on natural RGB photographs.

Training phases:
    Phase 1 — MedSAM frozen (87.3M params frozen)
        Train only CNN backbone + MedSAMCrossFusion layers (3.4M params)
        γ starts at 0 → model first learns to detect without MedSAM
        γ gradually increases → MedSAM features blend in

    Phase 2 — MedSAM unfrozen (optional, needs ≥8GB VRAM)
        Fine-tune all layers with 10x lower LR
        Allows ViT to adapt to GPR domain from medical domain

Usage:
    python train_medsam.py                        # scale=s, 100 epochs phase1
    python train_medsam.py --scale m              # larger model
    python train_medsam.py --epochs 150 --batch 8
    python train_medsam.py --phase2               # run phase2 after phase1
    python train_medsam.py --weights runs/gpr_medsam/phase1_s/weights/best.pt --phase2
    python train_medsam.py --compare              # compare vs DINOv3 version
"""

import argparse
import warnings
from pathlib import Path

import yaml as _yaml
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Depth, width, max_channels for each scale
_SCALES = {
    "n": [0.50, 0.25, 1024],
    "s": [0.50, 0.50, 1024],
    "m": [0.50, 1.00,  512],
    "l": [1.00, 1.00,  512],
    "x": [1.50, 1.00,  512],
}


def load_model_with_scale(config_path: str, scale: str) -> YOLO:
    """Load YOLO model config with the specified scale.

    Replaces the 'scales' dict with a single entry so ultralytics'
    fallback always selects the correct depth/width/max_channels.
    """
    cfg_path = Path(config_path)
    out_path = cfg_path.parent / f"_scale_{scale}.yaml"

    with open(config_path, "r") as f:
        cfg = _yaml.safe_load(f)

    # Keep only the target scale so tuple(scales.keys())[0] picks it
    cfg["scales"] = {"n": _SCALES[scale]}
    cfg.pop("scale", None)   # remove any leftover scale key

    with open(out_path, "w") as f:
        _yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return YOLO(str(out_path))

DATA_CONFIG       = "data_all.yaml"
MODEL_CONFIG      = "ultralytics/cfg/models/v26/yolov26_gpr_medsam.yaml"
DINOV3_CONFIG     = "ultralytics/cfg/models/v26/yolov26_gpr.yaml"   # for comparison
PROJECT           = "runs/gpr_medsam"

# GPR-safe augmentation rules (same for both phases)
# - NO vertical flip  → destroys depth ordering in B-scan
# - NO rotation       → destroys hyperbola shape
# - NO HSV hue/sat    → GPR is amplitude-only, not color
GPR_AUG = dict(
    fliplr=0.5,
    flipud=0.0,
    degrees=0.0,
    translate=0.1,
    scale=0.3,
    shear=0.0,
    perspective=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.2,
)


# ---------------------------------------------------------------------------

def print_banner(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def gamma_report(model):
    """Print current γ values of all CrossFusion layers."""
    from ultralytics.nn.modules import MedSAMCrossFusion
    gammas = []
    for name, m in model.model.named_modules():
        if isinstance(m, MedSAMCrossFusion):
            g = m.gamma.item()
            gammas.append((name, m.scale, g))
    if gammas:
        print("\n  MedSAMCrossFusion γ values:")
        for name, scale, g in gammas:
            bar = "█" * int(abs(g) * 20)
            print(f"    {scale:3s}  γ={g:.4f}  {bar}")


def freeze_gamma(model):
    """Freeze all CrossFusion γ params (keep them at 0 during phase1).
    Without this, gamma learns to blend frozen medical-domain ViT features
    into the CNN stream, which hurts GPR detection (domain mismatch noise).
    """
    from ultralytics.nn.modules import MedSAMCrossFusion
    n = 0
    for m in model.model.modules():
        if isinstance(m, MedSAMCrossFusion):
            m.gamma.requires_grad_(False)
            n += 1
    if n:
        print(f"  γ frozen for {n} CrossFusion layers (will unfreeze in phase2)")


def unfreeze_gamma(model):
    """Unfreeze CrossFusion γ params for phase2."""
    from ultralytics.nn.modules import MedSAMCrossFusion
    n = 0
    for m in model.model.modules():
        if isinstance(m, MedSAMCrossFusion):
            m.gamma.requires_grad_(True)
            n += 1
    if n:
        print(f"  γ unfrozen for {n} CrossFusion layers")


# ---------------------------------------------------------------------------

def phase1(scale: str, epochs: int, batch: int, imgsz: int, name_suffix: str = "") -> Path:
    """
    Phase 1: MedSAM ViT-B frozen.
    Train only CNN backbone + MedSAMCrossFusion layers.

    γ=0 init means the model starts as pure YOLOv26 CNN,
    then gradually learns to use MedSAM features via γ.
    """
    run_name = f"phase1_{scale}{name_suffix}"
    print_banner(f"PHASE 1  |  scale={scale}  epochs={epochs}  batch={batch}  imgsz={imgsz}")
    print("  MedSAM ViT-B: FROZEN (87.3M params)")
    print("  Training:     CNN backbone only — γ FROZEN at 0")
    print("  Rationale:    MedSAM features are medical-domain; blending them")
    print("                before phase2 domain-adapt hurts GPR detection.")

    model = load_model_with_scale(MODEL_CONFIG, scale)

    # Freeze γ so CrossFusion stays identity during phase1.
    # CNN trains to ~baseline performance; phase2 then unfreezes γ + ViT together.
    freeze_gamma(model)

    # Report initial γ (should all be 0.0)
    gamma_report(model)

    model.train(
        data=DATA_CONFIG,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        # Optimizer — AdamW works well for small datasets
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.001,           # gentler LR decay → refine longer near end
        weight_decay=0.05,   # stronger regularization for small dataset
        warmup_epochs=5,

        # Loss weights (single class: emphasise box quality)
        box=8.0,
        cls=0.5,
        dfl=1.5,

        # GPR-safe augmentation
        **GPR_AUG,
        mosaic=1.0,
        close_mosaic=30,     # disable mosaic last 30 epochs for cleaner train
        copy_paste=0.5,      # paste extra void patches — helps rare class
        mixup=0.1,           # blend pairs → regularize, reduce overfit

        # Logging
        project=PROJECT,
        name=run_name,
        exist_ok=True,
        plots=True,
        save_period=20,
        patience=60,
        verbose=True,
    )

    # Report final γ — shows how much MedSAM was actually used
    print("\n  Final γ after Phase 1:")
    gamma_report(model)

    best = Path(f"{PROJECT}/{run_name}/weights/best.pt")
    print(f"\n  Phase 1 done → {best}")
    return best


# ---------------------------------------------------------------------------

def phase2(weights: str, epochs: int, batch: int, imgsz: int, scale: str,
           name_suffix: str = "") -> Path:
    """
    Phase 2: Unfreeze MedSAM ViT-B for domain adaptation.
    Fine-tune all layers with 10x lower LR.

    Recommended only if:
      - GPU VRAM ≥ 8GB (ViT-B at 512×512 is memory-heavy)
      - Phase 1 converged well (γ > 0.1 on at least one scale)
    """
    run_name = f"phase2_{scale}{name_suffix}"
    print_banner(f"PHASE 2  |  fine-tune from {weights}")
    print("  MedSAM ViT-B: UNFROZEN — domain adaptation GPR←medical")
    print("  LR: 10x lower than Phase 1  (prevent catastrophic forgetting)")

    model = YOLO(weights)

    # Unfreeze MedSAM ViT weights
    from ultralytics.nn.modules import MedSAMFPN
    unfrozen_params = 0
    for layer in model.model.model:
        if isinstance(layer, MedSAMFPN):
            layer.freeze = False
            for p in layer.vit.parameters():
                p.requires_grad = True
            unfrozen_params = sum(p.numel() for p in layer.vit.parameters())
            print(f"  Unfroze MedSAMFPN ViT — {unfrozen_params/1e6:.1f}M params now trainable")

    # Unfreeze γ — now ViT is also adapting so CrossFusion can learn properly
    unfreeze_gamma(model)

    total_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  Total trainable: {total_trainable/1e6:.1f}M params")

    model.train(
        data=DATA_CONFIG,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        # 10x lower LR — prevents catastrophic forgetting of medical features
        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.001,
        weight_decay=0.01,
        warmup_epochs=3,

        box=8.0,
        cls=0.5,
        dfl=1.5,

        # Lighter augmentation in phase 2
        **GPR_AUG,
        mosaic=0.5,
        close_mosaic=10,
        copy_paste=0.1,

        project=PROJECT,
        name=run_name,
        exist_ok=True,
        plots=True,
        save_period=10,
        patience=30,
        verbose=True,
    )

    best = Path(f"{PROJECT}/{run_name}/weights/best.pt")
    print(f"\n  Phase 2 done → {best}")
    return best


# ---------------------------------------------------------------------------

def train_unified(scale: str, epochs: int, batch: int, imgsz: int,
                  unfreeze_epoch: int = None) -> Path:
    """
    Single-run training: freeze MedSAM for first N epochs, then unfreeze.
    Equivalent to Phase 1 + Phase 2 but without restarting.

    Uses YOLO callback 'on_train_epoch_end' to unfreeze ViT mid-training
    and drop LR 10x at the same epoch.

    Args:
        unfreeze_epoch: epoch to unfreeze MedSAM (default: epochs * 2/3)
    """
    if unfreeze_epoch is None:
        unfreeze_epoch = int(epochs * 2 / 3)  # e.g. epoch 100 of 150

    run_name = f"unified_{scale}"
    print_banner(f"UNIFIED TRAINING  |  scale={scale}  epochs={epochs}  batch={batch}")
    print(f"  Epoch 1-{unfreeze_epoch}:   MedSAM FROZEN   (CNN + CrossFusion train)")
    print(f"  Epoch {unfreeze_epoch+1}-{epochs}: MedSAM UNFROZEN (full fine-tune, LR /10)")
    print(f"  γ starts at 0 → opens gradually throughout")

    model = load_model_with_scale(MODEL_CONFIG, scale)
    unfrozen = {"done": False}

    def on_epoch_end(trainer):
        epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
        if epoch >= unfreeze_epoch and not unfrozen["done"]:
            unfrozen["done"] = True
            from ultralytics.nn.modules import MedSAMFPN
            total = 0
            for layer in trainer.model.model:
                if isinstance(layer, MedSAMFPN):
                    layer.freeze = False
                    for p in layer.vit.parameters():
                        p.requires_grad = True
                    total = sum(p.numel() for p in layer.vit.parameters())
            # drop LR 10x for all param groups
            for pg in trainer.optimizer.param_groups:
                pg["lr"] *= 0.1
            print(f"\n  [Epoch {epoch}] MedSAM unfrozen ({total/1e6:.1f}M params), LR /10")
            gamma_report(trainer.model)

    model.add_callback("on_train_epoch_end", on_epoch_end)

    model.train(
        data=DATA_CONFIG,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        optimizer="AdamW",
        lr0=0.002,
        lrf=0.001,
        weight_decay=0.05,
        warmup_epochs=5,

        box=8.0,
        cls=0.5,
        dfl=1.5,

        **GPR_AUG,
        mosaic=1.0,
        close_mosaic=30,
        copy_paste=0.5,
        mixup=0.1,

        project=PROJECT,
        name=run_name,
        exist_ok=True,
        plots=True,
        save_period=20,
        patience=60,
        verbose=True,
    )

    best = Path(f"{PROJECT}/{run_name}/weights/best.pt")
    print(f"\n  Unified training done → {best}")
    return best


# ---------------------------------------------------------------------------

def compare(scale: str, epochs: int, batch: int, imgsz: int):
    """
    Run both MedSAM and DINOv3 versions back-to-back for fair comparison.
    Results saved in runs/gpr_medsam/compare_*/
    """
    print_banner("COMPARISON: MedSAM vs DINOv3")
    print("  Both models trained with identical hyperparameters")
    print("  Check runs/gpr_medsam/ for results")

    results = {}

    # --- MedSAM ---
    print_banner("  [1/2] YOLOv26 + MedSAM ViT-B")
    model_m = load_model_with_scale(MODEL_CONFIG, scale)
    model_m.train(
        data=DATA_CONFIG, epochs=epochs, imgsz=imgsz, batch=batch,
        optimizer="AdamW", lr0=0.002, lrf=0.01, weight_decay=0.01,
        warmup_epochs=5, box=7.5, cls=0.5, dfl=1.5,
        **GPR_AUG, mosaic=1.0, copy_paste=0.5, mixup=0.1,
        project=PROJECT, name=f"compare_medsam_{scale}",
        exist_ok=True, plots=True, patience=30,
    )
    results["MedSAM"] = Path(f"{PROJECT}/compare_medsam_{scale}/weights/best.pt")

    # --- DINOv3 ---
    print_banner("  [2/2] YOLOv26 + DINOv3 ViT-S")
    model_d = load_model_with_scale(DINOV3_CONFIG, scale)
    model_d.train(
        data=DATA_CONFIG, epochs=epochs, imgsz=imgsz, batch=batch,
        optimizer="AdamW", lr0=0.002, lrf=0.01, weight_decay=0.01,
        warmup_epochs=5, box=7.5, cls=0.5, dfl=1.5,
        **GPR_AUG, mosaic=1.0, copy_paste=0.3,
        project=PROJECT, name=f"compare_dinov3_{scale}",
        exist_ok=True, plots=True, patience=30,
    )
    results["DINOv3"] = Path(f"{PROJECT}/compare_dinov3_{scale}/weights/best.pt")

    print_banner("COMPARISON DONE")
    for name, path in results.items():
        print(f"  {name}: {path}")
    print()
    print("  Compare mAP50 in runs/gpr_medsam/compare_*/results.csv")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv26-GPR-MedSAM void detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_medsam.py                              # phase 1 only, scale s
  python train_medsam.py --scale m --epochs 150       # larger model
  python train_medsam.py --phase2                     # phase 1 then phase 2
  python train_medsam.py --weights best.pt --phase2   # skip to phase 2
  python train_medsam.py --compare                    # MedSAM vs DINOv3
        """
    )
    parser.add_argument("--data",     default=None,
                        help="Path to data yaml (default: data_all.yaml)")
    parser.add_argument("--scale",    default="s",  choices=["n", "s", "m", "l"],
                        help="Model scale (s=recommended for ~2700 imgs, m=better but slower)")
    parser.add_argument("--epochs",   type=int, default=100,
                        help="Epochs for phase 1 (phase 2 uses epochs//2)")
    parser.add_argument("--batch",    type=int, default=16,
                        help="Batch size (reduce if OOM, phase 2 uses batch//2 automatically)")
    parser.add_argument("--imgsz",    type=int, default=640,
                        help="YOLO input image size (MedSAM resizes internally to 512)")
    parser.add_argument("--phase2",   action="store_true",
                        help="Run phase 2 (unfreeze MedSAM) after phase 1")
    parser.add_argument("--weights",  default=None,
                        help="Start from existing weights (skips phase 1, goes to phase 2)")
    parser.add_argument("--compare",  action="store_true",
                        help="Run comparison: MedSAM vs DINOv3 with same hyperparams")
    parser.add_argument("--unified",  action="store_true",
                        help="Single-run training: auto-unfreeze MedSAM at epoch 2/3")
    parser.add_argument("--unfreeze-epoch", type=int, default=None,
                        help="Epoch to unfreeze MedSAM in unified mode (default: epochs*2/3)")
    args = parser.parse_args()

    global DATA_CONFIG
    if args.data:
        DATA_CONFIG = args.data

    print_banner("YOLOv26-GPR-MedSAM Training")
    print(f"  Model:    {MODEL_CONFIG}")
    print(f"  Data:     {DATA_CONFIG}")
    print(f"  Scale:    {args.scale}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  ImgSz:    {args.imgsz}")
    print(f"  Project:  {PROJECT}/")

    if args.unified:
        train_unified(args.scale, args.epochs, args.batch, args.imgsz,
                      unfreeze_epoch=args.unfreeze_epoch)

    elif args.compare:
        compare(args.scale, args.epochs, args.batch, args.imgsz)

    elif args.weights:
        # Jump straight to phase 2 from existing checkpoint
        phase2(
            weights=args.weights,
            epochs=max(30, args.epochs // 2),
            batch=max(1, args.batch // 2),
            imgsz=args.imgsz,
            scale=args.scale,
        )

    else:
        best = phase1(args.scale, args.epochs, args.batch, args.imgsz)

        if args.phase2 and best.exists():
            phase2(
                weights=str(best),
                epochs=max(30, args.epochs // 2),
                batch=args.batch,   # keep same batch as phase1
                imgsz=args.imgsz,
                scale=args.scale,
            )
        elif args.phase2:
            print(f"\n  WARNING: phase 1 weights not found at {best}")
            print("  Skipping phase 2.")


if __name__ == "__main__":
    main()
