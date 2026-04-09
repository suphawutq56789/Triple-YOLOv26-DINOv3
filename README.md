<div align="center">

# YOLOv26-GPR

### Multi-Scale Cross-Attention Architecture for GPR Subsurface Void Detection

**Research Group, Department of Civil Engineering**
**King Mongkut's University of Technology Thonburi (KMUTT)**

</div>

---

## Overview

This repository implements **YOLOv26-GPR**, a novel object detection architecture for **Ground Penetrating Radar (GPR) subsurface void detection** in civil engineering inspection.

Two backbone variants are provided:

| Variant | Backbone | Pretrain Data | Domain Gap to GPR |
|---------|----------|--------------|-------------------|
| `yolov26_gpr.yaml` | **DINOv3 ViT-S/16** | 1.6B natural images | High |
| `yolov26_gpr_medsam.yaml` | **MedSAM ViT-B** | 1.5M medical images (incl. ultrasound) | Low |

**Domain rationale for MedSAM:** Ultrasound and GPR share the same pulse-echo physics — hyperbolic diffraction patterns, layered reflections, and void = signal dropout. MedSAM features transfer significantly better than DINOv3 natural-image features.

See [`FLOWCHART.md`](FLOWCHART.md) for full Mermaid diagrams of pipeline, architecture, and training phases.

---

## Architecture

### Core Design: ViT Feature Pyramid + CNN Cross-Attention

```
9-ch Triple GPR Input (3 scan orientations × 3ch)
        |
        v
+--------------------------------+
|  ViT FPN  (layer 0)            |  <- ViT runs ONCE
|  DINOv3FPN  OR  MedSAMFPN      |     Passthrough (output = input)
|  Caches P3 / P4 / P5 features  |
+--------------------------------+
        |
        v
  YOLOv26 CNN Backbone  (Stem -> P3 -> P4 -> P5)
        |              |           |
        v              v           v
  CrossFusion × 3  (one per FPN level)
  CNN(Q) × ViT(K,V) cross-attention
  gamma-gated residual  (gamma=0 init -> stable training)
        |
        v
  PAFPN Neck -> Detect Head -> Cavity boxes
```

**Evolution:**
- Previous (YOLOv12-based): ViT → 64ch → CNN — sequential, single scale
- **This work:** ViT cached features injected at P3 + P4 + P5 via cross-attention — multi-scale, parallel

### Modules

| Module | Role |
|--------|------|
| `DINOv3FPN` | Runs DINOv3 ViT-S once; caches P3/P4/P5; passthrough |
| `DINOv3CrossFusion` | CNN(Q) × DINOv3(K,V) cross-attention; gamma-gated residual |
| `MedSAMFPN` | Runs MedSAM ViT-B once; hooks P3/P4/P5 from blocks 4/8/12; passthrough |
| `MedSAMCrossFusion` | CNN(Q) × MedSAM(K,V) cross-attention; gamma-gated residual |

### Gamma-Gated Residual

```
out = CNN_feature + γ × CrossAttn(Q=CNN, K=V=ViT)
```

`γ` is a learnable parameter initialized to **0** — the model starts as pure YOLOv26 CNN and gradually learns how much ViT knowledge to blend in. This ensures stable training on small GPR datasets.

---

## Application: GPR Void Detection

**Task:** Detect subsurface voids (cavities) in concrete/road structures using GPR B-scan images.

**Input:** 9-channel triple input — 3 GPR scan orientations × 3 channels each

**Dataset:** ~910 images per orientation × 3 orientations = ~2,700 total images

**Class:** `Cavity` (void/hollow beneath surface)

GPR B-scan characteristics this architecture addresses:
- Hyperbolic reflection patterns from subsurface objects
- Depth-ordered signal (vertical axis = time/depth)
- Low contrast, high noise environments
- Small dataset size → requires strong pretrained features

---

## Model Variants

### DINOv3 Variant (`yolov26_gpr.yaml`)

| Scale | Total Params | Trainable | Frozen (DINOv3) | Recommended for |
|-------|-------------|-----------|----------------|-----------------|
| `n` | 25.3M | 3.3M | 22M | Fast testing |
| `s` | 28.6M | 6.5M | 22M | **Training ~2700 imgs** |
| `m` | 41.2M | 19.1M | 22M | Higher accuracy |

DINOv3 backbone options:

| Backbone | dim | Pretrain data |
|----------|-----|--------------|
| `dinov3-vits16-pretrain-lvd1689m` | 384 | LVD-1.6B images (default) |
| `dinov3-vitb16-pretrain-lvd1689m` | 768 | LVD-1.6B images |
| `dinov3-vitl16-pretrain-sat493m` | 1024 | Satellite 493M |

### MedSAM Variant (`yolov26_gpr_medsam.yaml`)

| Scale | Total Params | Trainable | Frozen (MedSAM) | Recommended for |
|-------|-------------|-----------|----------------|-----------------|
| `n` | 90.0M | 3.4M | 87.3M | Fast testing |
| `s` | 90.7M | 3.4M | 87.3M | **Training ~2700 imgs** |
| `m` | 106.1M | 18.8M | 87.3M | Higher accuracy |

MedSAM: `wanglab/medsam-vit-base` (ViT-B, dim=768, patch16, trained on 1.5M medical images)

---

## Installation

```bash
git clone https://github.com/suphawutq56789/Triple-YOLOv26-DINOv3-.git
cd Triple-YOLOv26-DINOv3-

pip install torch torchvision ultralytics transformers timm huggingface_hub
```

---

## Training

### Data config (`data_all.yaml`)

```yaml
path: /path/to/your/GPR/dataset

train: images/primary/train
val:   images/primary/val
test:  images/primary/test

triple_input: true
nc: 1
names: ['Cavity']
```

### DINOv3 Variant

```bash
# Phase 1 only (DINOv3 frozen, 100 epochs)
python train_gpr.py

# Phase 1 + Phase 2 (unfreeze DINOv3)
python train_gpr.py --phase2

# Larger model
python train_gpr.py --scale m --epochs 120 --phase2
```

### MedSAM Variant

```bash
# Phase 1 only (MedSAM frozen, 100 epochs)
python train_medsam.py

# Phase 1 + Phase 2 (unfreeze MedSAM — needs ≥8GB VRAM)
python train_medsam.py --phase2

# Larger model
python train_medsam.py --scale m --epochs 150 --batch 8 --phase2

# Resume from checkpoint → Phase 2
python train_medsam.py --weights runs/gpr_medsam/phase1_s/weights/best.pt --phase2

# Compare MedSAM vs DINOv3 (identical hyperparameters)
python train_medsam.py --compare --scale s --epochs 100
```

### Two-Phase Training Strategy

**Phase 1 — ViT Frozen**
- Train only CNN backbone + CrossFusion layers (3–6M params)
- γ starts at 0 → model first learns to detect without ViT
- γ gradually opens → ViT features blend in
- AdamW, LR = 0.002, 100 epochs, warmup 5 epochs

**Phase 2 — ViT Unfrozen** *(optional)*
- Fine-tune entire model including ViT backbone
- LR = 0.0002 (10× lower — prevents catastrophic forgetting)
- 50 epochs, reduced augmentation

### GPR-Safe Augmentation Rules

| Augmentation | Setting | Reason |
|--------------|---------|--------|
| Horizontal flip | `fliplr=0.5` | Hyperbola is left-right symmetric |
| Vertical flip | `flipud=0.0` | **DISABLED** — destroys depth ordering |
| Rotation | `degrees=0.0` | **DISABLED** — destroys hyperbola shape |
| HSV hue/sat | `hsv_h/s=0.0` | **DISABLED** — GPR is amplitude only, not color |
| Brightness | `hsv_v=0.2` | Simulate gain variation OK |
| Copy-paste | `copy_paste=0.3` | Augment rare void class |

---

## Inference

```python
from ultralytics import YOLO

# DINOv3 model
model = YOLO("runs/gpr/phase1_s/weights/best.pt")

# MedSAM model
model = YOLO("runs/gpr_medsam/phase1_s/weights/best.pt")

# Single image
results = model.predict("gpr_bscan.jpg", conf=0.25)

# With test-time augmentation (recommended for GPR)
results = model.predict("gpr_bscan.jpg", augment=True, conf=0.25)

# Batch predict with save
results = model.predict("path/to/test/images/", conf=0.25, save=True)
```

---

## GPR Preprocessing (recommended)

```python
import numpy as np
import cv2

def preprocess_gpr_bscan(img: np.ndarray) -> np.ndarray:
    """Background removal + Automatic Gain Control for GPR B-scans."""
    img = img.astype(np.float32)

    # 1. Background removal (remove direct wave / air arrival)
    img -= img.mean(axis=0, keepdims=True)

    # 2. AGC: compensate signal amplitude decay with depth
    for col in range(img.shape[1]):
        rms = np.sqrt(np.mean(img[:, col] ** 2)) + 1e-8
        img[:, col] /= rms

    # 3. Normalize to [0, 255]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)
```

---

## Repository Structure

```
Triple-YOLOv26-DINOv3/
|-- ultralytics/
|   |-- cfg/models/v26/
|   |   |-- yolov26_gpr.yaml              <- DINOv3 architecture
|   |   |-- yolov26_gpr_medsam.yaml       <- MedSAM architecture (NEW)
|   |   `-- yolov26_triple_dinov3*.yaml   <- Other variants
|   `-- nn/modules/
|       `-- dinov3.py                     <- DINOv3FPN + DINOv3CrossFusion
|                                            + MedSAMFPN + MedSAMCrossFusion (NEW)
|-- train_gpr.py                          <- DINOv3 two-phase training
|-- train_medsam.py                       <- MedSAM two-phase training (NEW)
|-- FLOWCHART.md                          <- Mermaid architecture diagrams (NEW)
|-- data_all.yaml                         <- Dataset config
`-- README.md
```

---

## Citation

```bibtex
@misc{yolov26gpr2025,
  title     = {YOLOv26-GPR: Multi-Scale Cross-Attention with MedSAM/DINOv3 for GPR Void Detection},
  author    = {KMUTT Civil Engineering Research Group},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/suphawutq56789/Triple-YOLOv26-DINOv3-}
}
```

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — base detection framework
- [MedSAM (wanglab)](https://github.com/bowang-lab/MedSAM) — medical image encoder
- [DINOv3 (Meta AI)](https://github.com/facebookresearch/dinov3) — pretrained ViT backbone
- [triple_YOLO13](https://github.com/Sompote/triple_YOLO13) — triple input concept

---

<div align="center">
<b>Department of Civil Engineering, KMUTT</b><br>
GPR Subsurface Void Detection Research
</div>
