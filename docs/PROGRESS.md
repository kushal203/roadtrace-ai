# RoadTrace AI — Progress & Checkpoint Log

## Milestones Summary

| Milestone | Status | Details |
| :--- | :---: | :--- |
| **1. Environment Inspection** | Completed | RTX 4050 6GB detected, Python 3.12 + Node 24 installed, PyTorch 2.5.1 + CUDA 12.1 validated. Documented in `docs/ENVIRONMENT_REPORT.md`. |
| **2. Dataset Research & Prep** | Completed | BDD100K (drivable masks, 3 classes) and IDD licenses & formats researched. Reproducible pipeline & synthetic generator in `scripts/prepare_dataset.py`. Documented in `docs/DATASET_REPORT.md`. |
| **3. Model Architecture** | Completed | Compact MobileNetV2 U-Net (2,630,963 params) created in `scripts/model.py`. Verified on CUDA with 256x256 tensor input. |
| **4. GPU Training** | Completed | Trained 3 epochs on RTX 4050 GPU using PyTorch AMP. Initial smoke test passed. Final validation mIoU reached `0.9975`. Best checkpoint saved to `models/checkpoints/best_model.pth`. |
| **5. Model Evaluation** | Completed | Evaluated 10 unseen test driving samples in `scripts/evaluate.py`. Mean IoU: `0.9977`. Predictions saved to `docs/predictions/`. |
| **6. Model Export (ONNX)** | Completed | Exported road segmentation model to `frontend/public/models/road_seg.onnx` (10.0 MB). Verified with `onnxruntime`. |
| **7. Obstacle Detection** | Completed | Pretrained YOLOv8n exported to ONNX with `onnxslim` pruning to `frontend/public/models/yolov8n.onnx` (12.3 MB). Road-overlap detection implemented in `scripts/obstacle_detection.py`. |
| **8. React/Vite Frontend** | Completed | Mobile-first React/TypeScript HUD in `frontend/src/`. Live camera access (`getUserMedia`), canvas overlays, ONNX Runtime WebAssembly inference, synthetic demo mode, and telemetry HUD. |
| **9. Production Build** | Completed | Compiled with Vite in 828ms. Static assets and 28 MB WASM runtime packaged into `frontend/dist/`. |
| **10. Vercel Deployment** | Completed | Deployed to Vercel CDN with WebAssembly headers. Live public HTTPS URL verified: `https://temporary-flying-fjord-idncre6.vercel.app`. |
