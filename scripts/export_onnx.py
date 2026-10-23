"""
RoadTrace AI - Export trained model to ONNX for browser inference

Usage:
  python scripts/export_onnx.py --checkpoint models/checkpoints/best_model.pth --output public/models/road_seg.onnx
"""

import os
import sys
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RoadSegmentationNet


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    img_size: int = 256,
    opset_version: int = 13,
    simplify: bool = True,
):
    """Export trained model to ONNX format."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    device = 'cpu'  # Export on CPU for compatibility
    
    # Create model and load weights
    model = RoadSegmentationNet(pretrained=False)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")
        print(f"  Best val IoU: {ckpt.get('best_val_iou', '?')}")
    else:
        print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
        print(f"  Exporting untrained model (for testing pipeline only)")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )
    
    # Verify
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output size: {file_size:.1f} MB")
    
    # Optional: simplify the ONNX model
    if simplify:
        try:
            import onnxsim
            import onnx
            
            print("  Simplifying ONNX model...")
            model_onnx = onnx.load(output_path)
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "ONNX simplification failed"
            onnx.save(model_onnx, output_path)
            
            simplified_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Simplified size: {simplified_size:.1f} MB")
        except ImportError:
            print("  onnxsim not installed, skipping simplification")
        except Exception as e:
            print(f"  Simplification failed: {e}")
    
    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("  Verifying with ONNX Runtime...")
        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        test_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        result = session.run([output_name], {input_name: test_input})
        
        print(f"  Input shape:  {test_input.shape}")
        print(f"  Output shape: {result[0].shape}")
        print(f"  Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")
        print("  [OK] ONNX Runtime verification passed!")
    except ImportError:
        print("  onnxruntime not installed, skipping verification")
    
    print(f"\nExport complete: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='Export road segmentation model to ONNX')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='frontend/public/models/road_seg.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--no_simplify', action='store_true',
                       help='Skip ONNX simplification')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        img_size=args.img_size,
        simplify=not args.no_simplify,
    )
