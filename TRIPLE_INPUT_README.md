# YOLOv26 Triple Input Implementation

This document describes the implementation of triple input functionality for YOLOv26, similar to the architecture found in triple_YOLO13. The triple input feature allows the model to process three related images simultaneously as a 9-channel input, potentially improving object detection accuracy through enhanced contextual understanding.

## Overview

The triple input architecture processes three related images:
1. **Primary image** (channels 0-2): Main RGB image
2. **Detail image 1** (channels 3-5): Additional contextual image
3. **Detail image 2** (channels 6-8): Additional contextual image

These are concatenated into a 9-channel input tensor and processed by a specialized `TripleInputConv` module that splits them into separate branches, processes each independently, and then fuses the features.

## Key Components

### 1. TripleInputConv Module (`ultralytics/nn/modules/conv.py`)

A specialized convolution layer that handles 9-channel input:

```python
class TripleInputConv(nn.Module):
    """Triple input convolution layer that processes 9-channel input as three separate 3-channel images."""
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        # c1: Input channels (must be 9)
        # c2: Output channels
        # Creates three separate Conv layers + fusion layer
```

**Features:**
- Splits 9-channel input into three 3-channel branches
- Processes each branch independently with separate Conv layers
- Fuses features from all branches using a 1x1 convolution
- Maintains the same interface as standard Conv layers

### 2. Model Configuration (`ultralytics/cfg/models/v26/yolov26_triple.yaml`)

Modified YOLOv26 configuration that uses `TripleInputConv` as the first layer:

```yaml
backbone:
  - [-1, 1, TripleInputConv, [64, 3, 2]] # 0-P1/2 - Triple input layer
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]     # Standard layers follow
  # ... rest of backbone
```

### 3. Dataset Support

Enhanced dataset loading with triple input support:

- **Base dataset** (`ultralytics/data/base.py`): Added `load_triple_images()` method
- **YOLO dataset** (`ultralytics/data/dataset.py`): Added `use_triple_input` flag
- **Smart fallback**: Uses primary image if detail images are missing

### 4. Dataset Configuration (`ultralytics/cfg/datasets/coco_triple.yaml`)

Example dataset configuration:

```yaml
path: ../datasets/coco_triple
train: images/train2017
val: images/val2017
triple_input: true  # Enable triple input mode
nc: 80
names: ...
```

## Dataset Structure

For triple input datasets with proper train/validation/test splits, organize your data as follows:

```
dataset_root/
├── images/
│   ├── train/                  # Training images
│   │   ├── image1.jpg          # Primary images
│   │   ├── image2.jpg
│   │   ├── detail1/
│   │   │   ├── image1.jpg      # Detail images 1
│   │   │   └── image2.jpg
│   │   └── detail2/
│   │       ├── image1.jpg      # Detail images 2
│   │       └── image2.jpg
│   ├── val/                    # Validation images (used during training)
│   │   └── (same structure as train)
│   └── test/                   # Test images (used for final evaluation)
│       └── (same structure as train)
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    ├── val/
    │   └── (same structure as train)
    └── test/
        └── (same structure as train)
```

**Smart Fallback Feature:**
- If `detail1/` or `detail2/` directories don't exist, the primary image is used as fallback
- This allows gradual migration to triple input format

## Usage Examples

### 1. Training with Proper Validation

```bash
# Train with validation during training
python train_with_validation.py \
    --model ultralytics/cfg/models/v26/yolov26_triple.yaml \
    --data test_triple_dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --patience 50
```

### 2. Final Model Evaluation on Test Set

```bash
# After training, evaluate on test set for final performance metrics
python test_model_evaluation.py \
    --model runs/detect/yolov26_triple_training/weights/best.pt \
    --data test_triple_dataset.yaml \
    --split test
```

### 3. Python API Usage

```python
from ultralytics import YOLO

# Load triple input model
model = YOLO('ultralytics/cfg/models/v26/yolov26_triple.yaml')

# Train with automatic validation during training
model.train(
    data='test_triple_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    val=True,        # Enable validation during training
    patience=50      # Early stopping patience
)

# After training, evaluate on test set
test_results = model.val(
    data='test_triple_dataset.yaml',
    split='test'     # Use test set for final evaluation
)
```

### 4. Manual Triple Input Processing

```python
import torch
from ultralytics.nn.modules.conv import TripleInputConv

# Create 9-channel input (3 RGB images concatenated)
primary = torch.randn(1, 3, 640, 640)
detail1 = torch.randn(1, 3, 640, 640)
detail2 = torch.randn(1, 3, 640, 640)
triple_input = torch.cat([primary, detail1, detail2], dim=1)  # Shape: [1, 9, 640, 640]

# Process with TripleInputConv
conv = TripleInputConv(c1=9, c2=64, k=3, s=2)
output = conv(triple_input)  # Shape: [1, 64, 320, 320]
```

### 5. Creating a Custom Triple Input Dataset

```python
import cv2
import numpy as np
from pathlib import Path

def create_triple_dataset(base_dir):
    \"\"\"Create a triple input dataset structure.\"\"\"
    base_dir = Path(base_dir)
    
    # Create directories for train/val/test splits
    for split in ['train', 'val', 'test']:
        (base_dir / f'images/{split}').mkdir(parents=True, exist_ok=True)
        (base_dir / f'images/{split}/detail1').mkdir(parents=True, exist_ok=True)
        (base_dir / f'images/{split}/detail2').mkdir(parents=True, exist_ok=True)
        (base_dir / f'labels/{split}').mkdir(parents=True, exist_ok=True)
    
    # Add your image processing logic here
    print(f"Dataset structure created at {base_dir}")

# Usage
create_triple_dataset("my_triple_dataset")
```

## Model Architecture Details

### Input Processing Flow

1. **Input**: 9-channel tensor `[B, 9, H, W]`
2. **Split**: Three 3-channel tensors `[B, 3, H, W]` each
3. **Branch Processing**: Each branch processed by separate Conv layers
4. **Feature Fusion**: Concatenate and fuse with 1x1 conv
5. **Output**: Standard feature maps `[B, C, H', W']`

### Memory Considerations

- Triple input requires 3x more memory for input images
- Feature fusion adds computational overhead
- Consider using smaller batch sizes for training

### Performance Impact

- **Benefits**: Enhanced contextual understanding, better detection accuracy
- **Costs**: Increased memory usage, slightly slower inference
- **Trade-off**: Improved accuracy vs computational efficiency

## Implementation Details

### Key Files Modified

1. **`ultralytics/nn/modules/conv.py`**: Added `TripleInputConv` class
2. **`ultralytics/nn/modules/__init__.py`**: Exported `TripleInputConv`
3. **`ultralytics/nn/tasks.py`**: Added special parsing for `TripleInputConv`
4. **`ultralytics/data/base.py`**: Added `load_triple_images()` method
5. **`ultralytics/data/dataset.py`**: Added triple input support

### Validation Tests

Run the test suite to validate the implementation:

```bash
python test_triple_input.py
```

Expected output:
```
YOLOv26 Triple Input Validation Tests
==================================================
TripleInputConv Module: ✓ PASSED
Model Configuration: ✓ PASSED
Dataset Configuration: ✓ PASSED
Dummy Dataset Creation: ✓ PASSED

Overall: 4/4 tests passed
🎉 All tests passed! YOLOv26 Triple Input is ready to use.
```

## Comparison with Standard YOLOv26

| Feature | Standard YOLOv26 | Triple Input YOLOv26 |
|---------|------------------|---------------------|
| Input Channels | 3 (RGB) | 9 (3 × RGB) |
| First Layer | `Conv(3, 64, 3, 2)` | `TripleInputConv(9, 64, 3, 2)` |
| Memory Usage | Baseline | ~3x for inputs |
| Dataset Format | Single images | Primary + 2 detail images |
| Fallback Support | N/A | Uses primary if details missing |

## Troubleshooting

### Common Issues

1. **"TripleInputConv expects 9 input channels"**
   - Ensure you're using the triple input model configuration
   - Check that your dataset has `triple_input: true`

2. **Missing detail images**
   - Create `detail1/` and `detail2/` subdirectories
   - Or rely on smart fallback (uses primary image)

3. **Memory errors**
   - Reduce batch size
   - Use smaller image sizes
   - Consider gradient checkpointing

### Debug Commands

```bash
# Test TripleInputConv module
python -c "from ultralytics.nn.modules.conv import TripleInputConv; print('✓ Import successful')"

# Test model creation
python -c "from ultralytics import YOLO; model = YOLO('ultralytics/cfg/models/v26/yolov26_triple.yaml'); print('✓ Model created')"

# Run full validation
python test_triple_input.py
```

## Future Enhancements

Potential improvements for the triple input implementation:

1. **Adaptive Fusion**: Learn optimal fusion weights during training
2. **Attention Mechanisms**: Add attention between the three branches
3. **Progressive Loading**: Dynamically add detail images during training
4. **Compression**: Optimize memory usage for large-scale training
5. **Multi-Scale**: Support different scales for detail images

## Conclusion

The YOLOv26 triple input implementation provides a flexible and powerful extension to the standard YOLO architecture. It maintains backward compatibility while adding sophisticated multi-image processing capabilities. The smart fallback feature ensures easy adoption, and the modular design allows for future enhancements.

For questions or issues, refer to the test suite and debug commands provided above.