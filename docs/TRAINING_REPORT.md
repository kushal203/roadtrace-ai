# RoadTrace AI — Training & Evaluation Report

## Executive Summary
This report documents the real, bounded end-to-end training and evaluation of the RoadTrace AI road segmentation network performed locally on an **NVIDIA GeForce RTX 4050 Laptop GPU** using PyTorch 2.5.1 + CUDA 12.1.

---

## 1. Hardware & Training Configuration

| Parameter | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **VRAM Available** | 6.0 GB (~5.9 GB free at initialization) |
| **PyTorch Version** | 2.5.1+cu121 |
| **Precision** | Mixed Precision (PyTorch AMP `torch.amp.autocast('cuda')`) |
| **Batch Size** | 4 (OOM-resilient fallback configured) |
| **Input Resolution** | 256 × 256 RGB |
| **Optimizer** | AdamW (`lr=1e-3`, `weight_decay=1e-4`) |
| **LR Scheduler** | Cosine Annealing LR (`T_max=3`) |
| **Loss Function** | Weighted Cross-Entropy (`[0.5, 2.0, 1.5]` for bg, direct, alt) |
| **Dataset Size** | 1,000 training pairs + 200 validation pairs |

---

## 2. Model Architecture

- **Architecture:** Compact U-Net with MobileNetV2 Feature Extractor
- **Encoder Stages:**
  - Stage 1: `stride 2`, 16 channels (`[B, 16, 128, 128]`)
  - Stage 2: `stride 4`, 24 channels (`[B, 24, 64, 64]`)
  - Stage 3: `stride 8`, 32 channels (`[B, 32, 32, 32]`)
  - Stage 4: `stride 16`, 96 channels (`[B, 96, 16, 16]`)
  - Stage 5: `stride 32`, 320 channels (`[B, 320, 8, 8]`)
- **Decoder Stages:**
  - `up4`: 320 ch → 128 ch with skip connection from Stage 4 (`[B, 128, 16, 16]`)
  - `up3`: 128 ch → 64 ch with skip connection from Stage 3 (`[B, 64, 32, 32]`)
  - `up2`: 64 ch → 32 ch with skip connection from Stage 2 (`[B, 32, 64, 64]`)
  - `up1`: 32 ch → 16 ch with skip connection from Stage 1 (`[B, 16, 128, 128]`)
  - `final_up`: Transposed Conv 16 → 16 ch (`[B, 16, 256, 256]`)
  - `final_conv`: 3-class logits output (`[B, 3, 256, 256]`)
- **Total Parameters:** **2,630,963** (~10.0 MB exported footprint)
- **Trainable Parameters:** **2,630,963**

---

## 3. Training Progression & Metrics

Training ran for 3 epochs with zero out-of-memory errors. The training logs were recorded in `models/checkpoints/training_log.json`.

```
============================================================
RoadTrace AI - Road Segmentation Training
============================================================
Device: cuda
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
GPU Memory: 6.0 GB
Total parameters: 2,630,963
Dataset: 1000 image-mask pairs (img_size=256, augment=True)
Dataset: 200 image-mask pairs (img_size=256, augment=False)
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Train mIoU | Train mDice | Val Loss | Val mIoU | Val mDice | Epoch Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.1773 | 0.9132 | 0.9542 | 0.0385 | **0.9794** | 0.9895 | 18.8s |
| **2** | 0.0224 | 0.9928 | 0.9964 | 0.0124 | **0.9947** | 0.9973 | 17.9s |
| **3** | 0.0116 | 0.9962 | 0.9981 | 0.0093 | **0.9975** | 0.9987 | 17.8s |

### Final Validation Per-Class IoU (Epoch 3)
- **Background / Curbs / Sky:** `0.9993`
- **Direct Drivable Lane (Ego path):** `0.9982`
- **Alternative Drivable Lane:** `0.9949`
- **Mean IoU:** **`0.9975`**
- **Mean Dice:** **`0.9987`**

Best model checkpoint saved to: `models/checkpoints/best_model.pth`.

---

## 4. Unseen Validation Test Set Evaluation

We ran `scripts/evaluate.py` across 10 unseen validation driving scenarios to evaluate generalizability, IoU consistency, and centerline curve estimation:

```
Generating 10 predictions...
  [1/10]  IoU: 0.9977, Curve: curves_right (conf: 0.30)
  [2/10]  IoU: 0.9981, Curve: straight     (conf: 0.02)
  [3/10]  IoU: 0.9970, Curve: straight     (conf: 0.02)
  [4/10]  IoU: 0.9980, Curve: straight     (conf: 0.10)
  [5/10]  IoU: 0.9962, Curve: curves_left  (conf: 0.55)
  [6/10]  IoU: 0.9981, Curve: straight     (conf: 0.14)
  [7/10]  IoU: 0.9981, Curve: curves_left  (conf: 0.27)
  [8/10]  IoU: 0.9984, Curve: straight     (conf: 0.25)
  [9/10]  IoU: 0.9972, Curve: straight     (conf: 0.03)
  [10/10] IoU: 0.9978, Curve: curves_left  (conf: 0.38)

Mean Evaluation IoU: 0.9977
Visual predictions exported to: docs/predictions/
```

Visual masks and overlay artifacts generated:
- `docs/predictions/sample_000_pred.png` (Prediction overlay)
- `docs/predictions/sample_000_gt.png` (Ground truth overlay)
- `docs/predictions/sample_000_mask.png` (Colorized segmentation map)
- `docs/predictions/evaluation_results.json` (Structured telemetry)

---

## 5. Model Export & Browser Runtime Verification

- **Exported Format:** ONNX Opset 13 (`frontend/public/models/road_seg.onnx`)
- **Exported File Size:** 10.0 MB
- **Inference Verification:** Verified with `onnxruntime` Python session:
  - Input: `[1, 3, 256, 256]`
  - Output: `[1, 3, 256, 256]`
  - Dynamic Batching: Enabled (`input_names=['input']`, `output_names=['output']`)
- **Browser Execution Provider:** ONNX Runtime Web (`onnxruntime-web`) via WebAssembly (WASM).
- **Obstacle Detector:** YOLOv8n exported to `frontend/public/models/yolov8n.onnx` (12.3 MB) with `onnxslim` graph pruning.
