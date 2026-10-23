"""
RoadTrace AI - Evaluation and Visualization Script

Usage:
  python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth --data_root data/bdd100k --output_dir docs/predictions
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RoadSegmentationNet
from dataset import BDD100KDrivableDataset


# Color map for visualization
CLASS_COLORS = {
    0: (0, 0, 0),        # Background: black
    1: (0, 200, 100),    # Direct drivable: green
    2: (0, 100, 200),    # Alternative drivable: blue
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class mask to RGB visualization."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay colored mask on image."""
    mask_rgb = colorize_mask(mask)
    # Only overlay where mask is non-zero
    overlay = image.copy()
    mask_region = mask > 0
    overlay[mask_region] = (
        image[mask_region] * (1 - alpha) + mask_rgb[mask_region] * alpha
    ).astype(np.uint8)
    return overlay


def estimate_curve(mask: np.ndarray) -> dict:
    """Estimate road curve direction from segmentation mask."""
    h, w = mask.shape
    drivable = (mask > 0).astype(np.float32)
    
    if drivable.sum() < 100:
        return {'direction': 'unknown', 'confidence': 0.0}
    
    # Compute centerline at different heights
    centers = []
    for row_frac in [0.3, 0.5, 0.7, 0.9]:
        row = int(h * row_frac)
        row_data = drivable[row]
        if row_data.sum() > 5:
            positions = np.where(row_data > 0)[0]
            center = (positions[0] + positions[-1]) / 2
            centers.append((row_frac, center / w))
    
    if len(centers) < 3:
        return {'direction': 'straight', 'confidence': 0.3}
    
    # Fit a line through centers and measure deviation
    fracs, x_positions = zip(*centers)
    x_positions = np.array(x_positions)
    
    # Compare top and bottom center positions
    drift = x_positions[-1] - x_positions[0]  # positive = shifts right (curves left)
    
    threshold = 0.08
    if abs(drift) < threshold:
        direction = 'straight'
    elif drift > 0:
        direction = 'curves_left'
    else:
        direction = 'curves_right'
    
    confidence = min(abs(drift) / 0.3, 1.0)
    
    return {
        'direction': direction,
        'confidence': round(confidence, 2),
        'drift': round(drift, 4),
        'centerline': list(zip([float(f) for f in fracs], [float(x) for x in x_positions])),
    }


def evaluate(args):
    """Run evaluation and save predictions."""
    print("=" * 60)
    print("RoadTrace AI - Model Evaluation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = RoadSegmentationNet(pretrained=False)
    
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
        print(f"  Best val IoU: {ckpt.get('best_val_iou', '?')}")
    else:
        print(f"  WARNING: Checkpoint not found, using random weights")
    
    model = model.to(device)
    model.eval()
    
    # Load validation data
    val_images = os.path.join(args.data_root, 'images', '10k', 'val')
    val_masks = os.path.join(args.data_root, 'drivable_maps', 'labels', 'val')
    
    dataset = BDD100KDrivableDataset(
        images_dir=val_images,
        masks_dir=val_masks,
        img_size=args.img_size,
        subset_size=args.num_samples,
        augment=False,
    )
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run predictions
    print(f"\nGenerating {len(dataset)} predictions...")
    results = []
    
    for idx in range(min(len(dataset), args.num_samples)):
        image_tensor, gt_mask_tensor = dataset[idx]
        
        # Inference
        with torch.no_grad():
            input_batch = image_tensor.unsqueeze(0).to(device)
            output = model(input_batch)
            pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
        
        gt_mask = gt_mask_tensor.numpy()
        
        # Denormalize image for visualization
        img = image_tensor.numpy()
        img = img * np.array([0.229, 0.224, 0.225])[:, None, None] + \
              np.array([0.485, 0.456, 0.406])[:, None, None]
        img = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        
        # Create visualizations
        pred_overlay = create_overlay(img, pred_mask)
        gt_overlay = create_overlay(img, gt_mask)
        
        # Curve estimation
        curve_info = estimate_curve(pred_mask)
        
        # Save
        img_name = f"sample_{idx:03d}"
        Image.fromarray(pred_overlay).save(output_dir / f"{img_name}_pred.png")
        Image.fromarray(gt_overlay).save(output_dir / f"{img_name}_gt.png")
        Image.fromarray(colorize_mask(pred_mask)).save(output_dir / f"{img_name}_mask.png")
        
        # Compute IoU for this sample
        intersection = ((pred_mask > 0) & (gt_mask > 0)).sum()
        union = ((pred_mask > 0) | (gt_mask > 0)).sum()
        iou = intersection / (union + 1e-8)
        
        results.append({
            'sample': img_name,
            'iou': round(float(iou), 4),
            'curve': curve_info,
        })
        
        print(f"  [{idx+1}/{args.num_samples}] IoU: {iou:.4f}, "
              f"Curve: {curve_info['direction']} ({curve_info['confidence']:.2f})")
    
    # Save results summary
    mean_iou = np.mean([r['iou'] for r in results])
    summary = {
        'num_samples': len(results),
        'mean_iou': round(float(mean_iou), 4),
        'samples': results,
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Mean IoU: {mean_iou:.4f}")
    print(f"  Results saved to: {output_dir}")
    
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate road segmentation model')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth')
    parser.add_argument('--data_root', type=str, default='data/bdd100k')
    parser.add_argument('--output_dir', type=str, default='docs/predictions')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
