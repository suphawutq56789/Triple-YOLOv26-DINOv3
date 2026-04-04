#!/usr/bin/env python3
"""
Training script for YOLOv26 Triple Input with pretrained weights.

This script demonstrates how to use pretrained YOLOv26 weights to initialize
a triple input model and fine-tune it for civil engineering applications.

Usage:
    python train_triple_pretrained.py --pretrained yolov26n.pt --data test_triple_dataset.yaml
    python train_triple_pretrained.py --pretrained yolov26s.pt --data test_triple_dataset.yaml --epochs 50
"""

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
from load_pretrained_triple import load_pretrained_weights_to_triple_model

def train_triple_with_pretrained(pretrained_path, data_config, model_config=None,
                                epochs=100, batch_size=16, imgsz=640,
                                patience=50, name="yolov26_triple_pretrained", device='0'):
    """
    Train triple input model with pretrained weights.
    
    Args:
        pretrained_path (str): Path to pretrained YOLOv26 model
        data_config (str): Path to dataset configuration
        model_config (str): Path to triple model config (optional)
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        imgsz (int): Image size
        patience (int): Early stopping patience
        name (str): Experiment name
        
    Returns:
        Training results
    """
    
    # Default triple model config
    if model_config is None:
        model_config = "ultralytics/cfg/models/v26/yolov26_triple.yaml"
    
    print("YOLOv26 Triple Input Training with Pretrained Weights")
    print("=" * 60)
    print(f"Pretrained Model: {pretrained_path}")
    print(f"Model Config: {model_config}")
    print(f"Data Config: {data_config}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print("-" * 60)
    
    # Step 1: Load pretrained weights into triple model
    print("\n🔄 Step 1: Loading pretrained weights...")
    model = load_pretrained_weights_to_triple_model(
        pretrained_path=pretrained_path,
        triple_model_config=model_config,
        save_path=None  # Don't save intermediate model
    )
    
    # Step 2: Validate model with dummy input
    print("\n🔍 Step 2: Validating model...")
    try:
        test_input = torch.randn(1, 9, imgsz, imgsz)
        with torch.no_grad():
            output = model.model(test_input)
        print(f"✓ Model validation passed - output shapes: {[x.shape for x in output] if isinstance(output, (list, tuple)) else output.shape}")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return None
    
    # Step 3: Start training with fine-tuning
    print(f"\n🚀 Step 3: Starting training with fine-tuning...")
    print("Note: Using lower learning rate for fine-tuning pretrained weights")
    
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        save=True,
        save_period=10,
        name=name,
        verbose=True,
        val=True,
        plots=False,  # Disable plots for triple input
        cache=False,  # Disable caching for triple input
        device=device,  # Use specified device
        workers=0,    # Disable multiprocessing for triple input

        # Disable augmentations that require multiple images
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        # Disable color augmentations for 9-channel input
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        auto_augment=None,

        lr0=0.001,    # Lower learning rate for fine-tuning
        lrf=0.1,      # Final learning rate factor
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        optimizer='AdamW',  # AdamW often works better for fine-tuning
    )
    
    print(f"\n✅ Training completed!")
    print(f"Best model saved to: runs/detect/{name}/weights/best.pt")
    print(f"Last model saved to: runs/detect/{name}/weights/last.pt")
    
    # Step 4: Validate on validation set
    print(f"\n📊 Step 4: Final validation...")
    val_results = model.val(data=data_config, split='val')
    print(f"Validation mAP50: {val_results.box.map50:.4f}")
    print(f"Validation mAP50-95: {val_results.box.map:.4f}")
    
    # Step 5: Instructions for test evaluation
    print(f"\n🧪 Step 5: Test set evaluation")
    print("To evaluate on test set after training, run:")
    print(f"python test_model_evaluation.py --model runs/detect/{name}/weights/best.pt --data {data_config} --split test")
    
    return results

def compare_models(pretrained_path, data_config, epochs=50):
    """
    Compare training from scratch vs pretrained initialization.
    
    Args:
        pretrained_path (str): Path to pretrained model
        data_config (str): Path to dataset config
        epochs (int): Number of epochs for comparison
    """
    print("Comparing training from scratch vs pretrained initialization")
    print("=" * 70)
    
    # Train from scratch
    print("\n🆕 Training from scratch...")
    scratch_model = YOLO("ultralytics/cfg/models/v26/yolov26_triple.yaml")
    scratch_results = scratch_model.train(
        data=data_config,
        epochs=epochs,
        batch=8,  # Smaller batch for comparison
        name="triple_from_scratch",
        verbose=False
    )
    
    # Train with pretrained
    print(f"\n🔄 Training with pretrained weights from {pretrained_path}...")
    pretrained_results = train_triple_with_pretrained(
        pretrained_path=pretrained_path,
        data_config=data_config,
        epochs=epochs,
        batch_size=8,
        name="triple_pretrained"
    )
    
    # Compare results
    print("\n📈 Comparison Results:")
    print("-" * 30)
    
    if scratch_results and pretrained_results:
        scratch_map = scratch_results.metrics.get('metrics/mAP50(B)', 0)
        pretrained_map = pretrained_results.metrics.get('metrics/mAP50(B)', 0)
        
        print(f"From Scratch mAP50:  {scratch_map:.4f}")
        print(f"Pretrained mAP50:    {pretrained_map:.4f}")
        print(f"Improvement:         {pretrained_map - scratch_map:.4f}")
        
        if pretrained_map > scratch_map:
            print("✅ Pretrained initialization improved performance!")
        else:
            print("⚠️ Pretrained initialization didn't improve performance")
    else:
        print("⚠️ Could not complete comparison")

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv26 Triple Input with pretrained weights')
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained YOLOv26 model (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset configuration (.yaml file)')
    parser.add_argument('--model', type=str,
                       help='Path to triple model config (default: ultralytics/cfg/models/v26/yolov26_triple.yaml)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    parser.add_argument('--name', type=str, default= 'yolov26_triple_pretrained',
                       help='Experiment name (default: yolov26_triple_pretrained)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu, use "0" for GPU)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare pretrained vs from-scratch training')
    
    args = parser.parse_args()

    # Validate input files - create pretrained if not exists
    if not Path(args.pretrained).exists():
        print(f"⚠️ Pretrained model not found: {args.pretrained}")
        print(f"🔧 Creating empty pretrained weights...")

        # Extract variant from filename (e.g., yolov26n.pt -> n, yolov26s.pt -> s)
        import re
        variant_match = re.search(r'yolov26([nsmx]|l)\.pt', args.pretrained)
        if variant_match:
            variant = variant_match.group(1)
        else:
            variant = 'n'  # default to nano

        # Create pretrained weights
        from ultralytics import YOLO
        import yaml

        config_path = 'ultralytics/cfg/models/v26/yolov26.yaml'
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        cfg['scale'] = variant
        cfg['ch'] = 3  # Standard YOLO uses 3 channels

        temp_cfg = f'temp_yolov26_{variant}.yaml'
        with open(temp_cfg, 'w') as f:
            yaml.dump(cfg, f)

        model = YOLO(temp_cfg)
        model.model.yaml['scale'] = variant
        model.save(args.pretrained)

        Path(temp_cfg).unlink(missing_ok=True)
        print(f"✅ Created {args.pretrained} with variant '{variant}'")
    
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")
    
    if args.model and not Path(args.model).exists():
        raise FileNotFoundError(f"Model config not found: {args.model}")
    
    # Run comparison or regular training
    if args.compare:
        compare_models(args.pretrained, args.data, epochs=min(args.epochs, 50))
    else:
        train_triple_with_pretrained(
            pretrained_path=args.pretrained,
            data_config=args.data,
            model_config=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            name=args.name,
            device=args.device
        )

if __name__ == "__main__":
    main()