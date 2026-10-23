"""
RoadTrace AI - Training Script for Road Segmentation

Usage:
  python scripts/train.py --data_root data/bdd100k_real --epochs 10 --batch_size 4

Features:
  - Mixed precision training (AMP)
  - Checkpoint saving/resuming
  - IoU and Dice metrics
  - Early stopping
  - Automatic batch size reduction on OOM
  - Training log export
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RoadSegmentationNet, create_model
from dataset import create_dataloaders


class SegmentationMetrics:
    """Compute IoU and Dice metrics for segmentation."""
    
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix with batch predictions."""
        preds = preds.cpu().long()
        targets = targets.cpu().long()
        
        for cls in range(self.num_classes):
            pred_cls = (preds == cls)
            target_cls = (targets == cls)
            self.confusion[cls, cls] += (pred_cls & target_cls).sum().item()
            for other in range(self.num_classes):
                if other != cls:
                    self.confusion[cls, other] += (pred_cls & (targets == other)).sum().item()
    
    def compute_iou(self) -> dict:
        """Compute per-class and mean IoU."""
        ious = {}
        class_names = ['background', 'direct_drivable', 'alt_drivable']
        
        for cls in range(self.num_classes):
            tp = self.confusion[cls, cls].item()
            fp = self.confusion[cls, :].sum().item() - tp
            fn = self.confusion[:, cls].sum().item() - tp
            
            iou = tp / (tp + fp + fn + 1e-8)
            ious[class_names[cls]] = round(iou, 4)
        
        ious['mean_iou'] = round(sum(ious.values()) / len(ious), 4)
        return ious
    
    def compute_dice(self) -> dict:
        """Compute per-class and mean Dice score."""
        dices = {}
        class_names = ['background', 'direct_drivable', 'alt_drivable']
        
        for cls in range(self.num_classes):
            tp = self.confusion[cls, cls].item()
            fp = self.confusion[cls, :].sum().item() - tp
            fn = self.confusion[:, cls].sum().item() - tp
            
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            dices[class_names[cls]] = round(dice, 4)
        
        dices['mean_dice'] = round(sum(dices.values()) / len(dices), 4)
        return dices


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: str,
    epoch: int,
    use_amp: bool = True,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    metrics = SegmentationMetrics()
    
    start_time = time.time()
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update metrics
        preds = outputs.argmax(dim=1)
        metrics.update(preds, masks)
        
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Avg: {total_loss/num_batches:.4f} "
                  f"Time: {elapsed:.1f}s")
    
    avg_loss = total_loss / max(num_batches, 1)
    ious = metrics.compute_iou()
    dices = metrics.compute_dice()
    
    return {
        'loss': round(avg_loss, 4),
        'iou': ious,
        'dice': dices,
        'time': round(time.time() - start_time, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = True,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    metrics = SegmentationMetrics()
    
    start_time = time.time()
    
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        num_batches += 1
        
        preds = outputs.argmax(dim=1)
        metrics.update(preds, masks)
    
    avg_loss = total_loss / max(num_batches, 1)
    ious = metrics.compute_iou()
    dices = metrics.compute_dice()
    
    return {
        'loss': round(avg_loss, 4),
        'iou': ious,
        'dice': dices,
        'time': round(time.time() - start_time, 1),
    }


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("RoadTrace AI - Road Segmentation Training")
    print("=" * 60)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    # Paths
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("\nCreating model...")
    model = create_model(pretrained=True, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dataloaders
    print(f"\nLoading dataset from: {args.data_root}")
    batch_size = args.batch_size
    
    try:
        train_loader, val_loader = create_dataloaders(
            data_root=args.data_root,
            img_size=args.img_size,
            batch_size=batch_size,
            subset_size=args.subset,
            num_workers=args.num_workers,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease download the BDD100K dataset first.")
        print("See docs/DATASET_REPORT.md for instructions.")
        sys.exit(1)

    generated_prefixes = ('train_road_', 'train_floor_', 'train_stairs_', 'train_outdoor_')
    generated_images = [
        os.path.basename(image_path)
        for image_path, _ in train_loader.dataset.pairs
        if os.path.basename(image_path).startswith(generated_prefixes)
    ]
    if generated_images and not args.allow_synthetic:
        raise ValueError(
            f"Refusing to train on generated demo data ({len(generated_images)} files detected). "
            "Use real labeled driving images, or pass --allow_synthetic only for a demo run."
        )
    
    # Loss function with increased background penalty for negative rejection
    class_weights = torch.tensor([0.8, 1.8, 1.4], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_iou = 0
    training_log = []
    
    resume_path = checkpoint_dir / 'last_checkpoint.pth'
    if args.resume and resume_path.exists():
        print(f"\nResuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_iou = ckpt.get('best_val_iou', 0)
        training_log = ckpt.get('training_log', [])
        print(f"Resumed at epoch {start_epoch}, best val IoU: {best_val_iou:.4f}")
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs (starting from {start_epoch})...")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {args.img_size}x{args.img_size}")
    print(f"  Subset: {args.subset or 'full'}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Mixed precision: {use_amp}")
    print()
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Train
        try:
            train_result = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, 
                device, epoch, use_amp
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\nOOM Error! Reducing batch size from {batch_size} to {batch_size // 2}")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                train_loader, val_loader = create_dataloaders(
                    data_root=args.data_root,
                    img_size=args.img_size,
                    batch_size=batch_size,
                    subset_size=args.subset,
                    num_workers=args.num_workers,
                )
                # Retry this epoch
                train_result = train_one_epoch(
                    model, train_loader, criterion, optimizer, scaler, 
                    device, epoch, use_amp
                )
            else:
                raise
        
        # Validate
        val_result = validate(model, val_loader, criterion, device, use_amp)
        
        scheduler.step()
        
        # Log
        epoch_log = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'train': train_result,
            'val': val_result,
        }
        training_log.append(epoch_log)
        
        print(f"  Train - Loss: {train_result['loss']:.4f}, "
              f"mIoU: {train_result['iou']['mean_iou']:.4f}, "
              f"mDice: {train_result['dice']['mean_dice']:.4f}")
        print(f"  Val   - Loss: {val_result['loss']:.4f}, "
              f"mIoU: {val_result['iou']['mean_iou']:.4f}, "
              f"mDice: {val_result['dice']['mean_dice']:.4f}")
        print(f"  Val IoU per class: {val_result['iou']}")
        
        # Save checkpoint
        is_best = val_result['iou']['mean_iou'] > best_val_iou
        if is_best:
            best_val_iou = val_result['iou']['mean_iou']
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'training_log': training_log,
                'config': vars(args),
            }, checkpoint_dir / 'best_model.pth')
            print(f"  [*] New best model saved (mIoU: {best_val_iou:.4f})")
        else:
            patience_counter += 1
        
        # Always save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_iou': best_val_iou,
            'training_log': training_log,
            'config': vars(args),
        }, checkpoint_dir / 'last_checkpoint.pth')
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {patience_counter} epochs without improvement")
            break
        
        print()
    
    # Save training log
    log_path = checkpoint_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\nTraining log saved to {log_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Best validation mIoU: {best_val_iou:.4f}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Best model: {checkpoint_dir / 'best_model.pth'}")
    print("=" * 60)
    
    return best_val_iou


def parse_args():
    parser = argparse.ArgumentParser(description='RoadTrace AI - Train Road Segmentation')
    parser.add_argument('--data_root', type=str, default='data/bdd100k_real',
                       help='Path to BDD100K dataset root')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size (square)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use subset of training data')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='DataLoader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints_real',
                       help='Directory for checkpoints')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--allow_synthetic', action='store_true',
                       help='Allow generated demo data (not suitable for real-world accuracy claims)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
