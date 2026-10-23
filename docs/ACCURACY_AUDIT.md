# RoadTrace AI — Phase 1 Accuracy & Architectural Audit

## Executive Summary
This audit rigorously investigates the reported metrics (**99.75% validation mIoU** and **99.87% mDice**) for RoadTrace AI, identifies the root causes of real-world misclassifications (specifically on stairs, floors, and indoor environments), diagnoses the ~1.8 FPS mobile performance bottleneck, and explains why curve estimation remains perpetually stuck on "Calculating...".

---

## 1. Key Audit Findings & Root Causes

### Root Cause 1: Synthetic Data Monoculture & False High IoU
- **Dataset Composition:** 100% of the 1,000 training pairs and 200 validation pairs in `data/bdd100k/` were procedurally synthesized via Pillow drawings in `scripts/prepare_dataset.py` (`synth_00000.jpg` to `synth_00999.jpg`).
- **The Metric Reality:** The reported **99.75% validation mIoU** and **99.87% mDice** are mathematically genuine on the synthetic dataset, but have **zero real-world generalization**.
- **Absence of Negative Scenes:** The dataset contained zero real road asphalt textures, zero indoor surfaces, zero staircases, zero tile floors, and zero blank walls.

### Root Cause 2: Empirical Reproduction of the Staircase & Indoor Floor Failure
Using the audit test harness [`scripts/audit_model.py`](file:///C:/Users/yella/.gemini/antigravity/scratch/roadtrace-ai/scripts/audit_model.py), the trained checkpoint `models/checkpoints/best_model.pth` was tested against controlled non-road test inputs:

| Non-Road Test Input | Background Pixels | Direct Drivable | Alt Drivable | Total False-Positive Road | Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Simulated Staircase** | 63.5% | 17.0% | 19.5% | **36.5%** | **82.9%** |
| **Indoor Floor & Wall** | 53.0% | 1.8% | 45.2% | **47.0%** | **90.3%** |
| **Blank White Wall** | 99.6% | 0.4% | 0.0% | **0.4%** | 86.3% |

**Verdict:** Because the synthetic generator consistently placed drivable road trapezoids in the bottom 50% of images, the network's spatial bias treats any horizontal or perspective ground plane (stair treads, hardwood/tile flooring) as drivable road.

### Root Cause 3: The Stuck "Calculating..." State Bug
- In `frontend/src/hooks/useInference.ts` (line 220):
  ```typescript
  setResults({
    mask: newMask,
    boxes: newBoxes,
    curveInfo: null, // will be computed during drawing
  })
  ```
- In `frontend/src/components/CameraView.tsx`: `drawCenterline` computes `curveInfo` internally for canvas overlay, but **never passes it back up to `App.tsx`**.
- In `frontend/src/components/StatsDisplay.tsx`:
  ```tsx
  {curveInfo ? (
    <span>{curveInfo.direction}</span>
  ) : (
    <span className="text-sm font-mono text-slate-500">Calculating...</span>
  )}
  ```
- **Verdict:** Because `results.curveInfo` is initialized to `null` and never set during live camera frames, `StatsDisplay` is **permanently stuck on "Calculating..."**.

### Root Cause 4: Mobile Performance Bottleneck (~1.7–1.8 FPS)
Profiling `frontend/src/hooks/useInference.ts` and `frontend/src/utils/drawing.ts` reveals three severe computational bottlenecks on mobile devices:
1. **Full-Resolution Pixel Loop in JavaScript:** `drawSegmentationMask` creates a 1280×720 `ImageData` buffer and runs a **921,600 iteration loop** in JavaScript on every animation frame to populate RGBA values.
2. **Synchronous 640×640 Letterbox Preprocessing:** `preprocessForDetection` allocates canvas contexts, fetches full image data, and allocates a 3.68 MB `Float32Array` on the main UI thread.
3. **Single-Thread WASM Inference:** Both MobileNetV2 U-Net and YOLOv8n run sequentially on a single WebAssembly thread without WebGPU or multi-threaded SIMD pipeline overlap, consuming ~550ms total per frame (~1.8 FPS).

### Root Cause 5: Absence of Minimum Confidence & Validity Gating
- `drawCenterline` in `drawing.ts` attempts to extract a centerline if as few as 2 rows have even a single road pixel.
- Isolated noise or false positives on stairs create immediate wild centerlines, leading to severe visual jitter and false directional warnings.

---

## 2. Quantitative Confusion Matrix (Validation Split)

```
Recalculated on 200 validation samples (13,107,200 total pixels):
                      Predicted Background   Predicted Direct   Predicted Alt
True Background             10,085,224              92               488
True Direct Drivable             2,630       2,048,438               845
True Alt Drivable                3,398             215           965,870
```
- Background IoU: `0.9993`
- Direct Drivable IoU: `0.9982`
- Alternative Drivable IoU: `0.9949`
- Combined Drivable IoU: `0.9978`
*(Reflects synthetic pattern memorization rather than real driving domain generalization).*

---

## 3. Action Plan for Phase 2–8

1. **Phase 2 & 5 (Frontend Tracing & Gating Fix):**
   - Add road pixel coverage threshold (at least 8% of the bottom half must be coherent road pixels).
   - Implement road confidence gating (hide path and show `"No drivable road detected"` when confidence is low or mask is fragmented).
   - Fix the stuck `"Calculating..."` bug by threading `curveInfo` back from canvas drawing or computing it in the inference loop.
   - Smooth temporal jitter using exponential moving average (EMA) on centerline coordinates.

2. **Phase 3 & 4 (Real-World Data & Robust Retraining):**
   - Incorporate real road imagery and explicit negative examples (indoor rooms, stairs, sidewalks, tile patterns).
   - Fine-tune/train with mixed precision on the RTX 4050 GPU with high weight on non-road rejection.
   - Measure true generalization on real held-out test scenes.

3. **Phase 6 (Obstacle Detection):**
   - Verify YOLOv8n ONNX class mapping, scale bounding boxes properly, and calculate road-overlap flags only on valid road regions.

4. **Phase 7 (Inference Performance):**
   - Optimize mask rendering: render 256×256 mask offscreen (65k pixels instead of 921k) and let the canvas GPU scale it with bilinear filtering.
   - Decouple object detection: run detection every 4th frame or asynchronously so road segmentation can maintain 15–30 FPS.

5. **Phase 8 (Vercel Deployment):**
   - Rebuild and redeploy to Vercel, confirming the live URL and verifying camera and accuracy improvements.
