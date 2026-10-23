"""
RoadTrace AI - Obstacle Detection using YOLOv8n

Uses ultralytics YOLOv8 nano model (pretrained on COCO).
License: AGPL-3.0 (ultralytics) - free for non-commercial use.

Usage:
  python scripts/obstacle_detection.py --image test_image.jpg
  python scripts/obstacle_detection.py --export  # Export to ONNX for browser
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# COCO classes relevant for road/traffic scenarios
ROAD_OBSTACLE_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    11: 'stop sign',
}


def setup_yolo():
    """Import and return ultralytics YOLO class."""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("Install with: pip install ultralytics")
        sys.exit(1)


def detect_obstacles(
    image_path: str,
    model_path: str = 'yolov8n.pt',
    conf_threshold: float = 0.3,
    classes: list = None,
) -> dict:
    """Run obstacle detection on an image."""
    YOLO = setup_yolo()
    
    # Load model
    model = YOLO(model_path)
    
    # Filter to road-relevant classes
    if classes is None:
        classes = list(ROAD_OBSTACLE_CLASSES.keys())
    
    # Run inference
    results = model(image_path, conf=conf_threshold, classes=classes, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': ROAD_OBSTACLE_CLASSES.get(cls_id, f'class_{cls_id}'),
                    'confidence': round(conf, 3),
                    'bbox': [round(x1), round(y1), round(x2), round(y2)],
                })
    
    return {
        'image': image_path,
        'num_detections': len(detections),
        'detections': detections,
    }


def export_yolo_onnx(
    output_path: str = 'frontend/public/models/yolov8n.onnx',
    img_size: int = 416,
):
    """Export YOLOv8n to ONNX for browser inference."""
    YOLO = setup_yolo()
    
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    print(f"Exporting to ONNX (input size: {img_size}x{img_size})...")
    export_path = model.export(
        format='onnx',
        imgsz=img_size,
        simplify=True,
        opset=13,
        dynamic=False,
    )
    
    # Move to desired output location
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if export_path != output_path:
        import shutil
        shutil.copy2(export_path, output_path)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported to: {output_path}")
    print(f"  Size: {file_size:.1f} MB")
    
    return output_path


def visualize_detections(
    image_path: str,
    detections: list,
    output_path: str = None,
    road_mask: np.ndarray = None,
):
    """Draw detection results on image and optionally check overlap with road mask."""
    try:
        import cv2
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Color based on whether object overlaps drivable area
            color = (0, 255, 0)  # Green by default
            if road_mask is not None:
                # Check if bottom center of bbox is on drivable area
                cx = int((x1 + x2) / 2)
                cy = int(y2)
                if 0 <= cy < road_mask.shape[0] and 0 <= cx < road_mask.shape[1]:
                    if road_mask[cy, cx] > 0:
                        color = (0, 0, 255)  # Red = obstacle on road
                        label += " [ON ROAD]"
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Saved visualization: {output_path}")
        
        return img
        
    except ImportError:
        print("OpenCV not available for visualization")
        return None


def test_detection(args):
    """Test obstacle detection on sample image(s)."""
    print("=" * 60)
    print("RoadTrace AI - Obstacle Detection Test")
    print("=" * 60)
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        
        print(f"\nTesting on: {args.image}")
        result = detect_obstacles(args.image, conf_threshold=args.conf)
        
        print(f"Detections: {result['num_detections']}")
        for det in result['detections']:
            print(f"  {det['class_name']}: {det['confidence']:.3f} "
                  f"@ [{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]")
        
        # Visualize if output path specified
        if args.output:
            visualize_detections(args.image, result['detections'], args.output)
    
    elif args.export:
        export_yolo_onnx(
            output_path=args.export_path,
            img_size=args.img_size,
        )
    
    else:
        print("Specify --image <path> to test or --export to export ONNX")


def parse_args():
    parser = argparse.ArgumentParser(description='RoadTrace AI - Obstacle Detection')
    parser.add_argument('--image', type=str, help='Test image path')
    parser.add_argument('--output', type=str, help='Output visualization path')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--export', action='store_true', help='Export YOLOv8n to ONNX')
    parser.add_argument('--export_path', type=str, default='frontend/public/models/yolov8n.onnx')
    parser.add_argument('--img_size', type=int, default=416)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test_detection(args)
