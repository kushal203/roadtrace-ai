# RoadTrace AI — Dataset Research & Preparation Report

## 1. Overview
For RoadTrace AI, road segmentation requires drivable area classification (distinguishing ego lane, alternative lanes, and non-drivable obstacles/curbs/background), while obstacle detection requires bounding box annotations (vehicles, pedestrians, cyclists).

We evaluated two benchmark datasets:
1. **BDD100K (Berkeley DeepDrive 100K)** — Primary Dataset
2. **India Driving Dataset (IDD)** — Secondary / Evaluation Dataset

---

## 2. BDD100K Analysis (Primary)

### License & Access
- **License:** Freely permitted for non-commercial, educational, and academic research purposes. Copyright © The Regents of the University of California.
- **Registration:** Account required at `bdd-data.berkeley.edu`.
- **Cost:** ₹0 (Free tier).

### Dataset Breakdown & Sizes
| Asset | Size | Purpose |
| :--- | :--- | :--- |
| `bdd100k_drivable_maps.zip` | ~661 MB | Pre-rendered 8-bit PNG segmentation masks |
| `bdd100k_images_10k.zip` | ~2.1 GB | 10,000 720p images matching segmentation labels |
| `bdd100k_labels_release.zip` | ~15 MB | Object detection 2D bounding boxes (COCO/Scalabel format) |
| Full 100K images | ~7.1 GB | Full video keyframe set (unnecessary for initial training) |

### Class Encodings for Drivable Area
BDD100K drivable area maps encode semantic road pixels into 3 classes:
- **Class 0:** Background / Non-drivable (curbs, sidewalks, vegetation, barriers, sky)
- **Class 1:** Direct Drivable Area (ego vehicle's current travel lane / clear right-of-way)
- **Class 2:** Alternative Drivable Area (adjacent drivable lanes, oncoming lanes, or passable shoulders)

### Suitability
- Masks are already rasterized PNGs matching image basenames (`<id>_drivable_id.png`).
- Direct 1-to-1 pixel alignment with 720p images (1280×720).
- Highly relevant to real-world autonomous driving perception.

---

## 3. India Driving Dataset (IDD) Analysis (Secondary)

### License & Access
- **License:** Free for academic and scientific research. Copyright © IIIT Hyderabad / iHUB-Data.
- **Registration:** Required at `idd.insaan.iiit.ac.in`.
- **Cost:** ₹0.

### Dataset Breakdown & Sizes
- **IDD 20K Segmentation:** ~24 GB total (Part 1: 18.5 GB, Part 2: 5.5 GB).
- **IDD Lite:** < 30 MB (lightweight subset with polygon JSON annotations).

### Suitability & Challenges
- Extremely unstructured traffic conditions (two-wheelers, auto-rickshaws, pedestrians, unpaved roads).
- High diversity, but Cityscapes-style multi-level polygon JSONs require rasterization (`createLabels.py`).
- Download footprint (24 GB) is prohibitive for fast, bounded prototyping. IDD Lite can serve as a test bench for qualitative domain shift evaluation.

---

## 4. Dataset Strategy for Bounded Local Training
To respect bandwidth and RTX 4050 resource constraints:
1. **Target Subset:** 1,000 training pairs + 200 validation pairs sampled randomly with seed `42`.
2. **Resolution:** Resized to 256×256 for rapid convergence and low VRAM footprint (~2.1 GB during batch size 4).
3. **Data Preparation Script:** Provided at `scripts/prepare_dataset.py` to automatically verify masks, validate checksums, and prepare train/val subsets.
4. **Synthetic/Demo Fallback:** If offline or awaiting full archive download, `scripts/prepare_dataset.py --create-synthetic` generates a verified synthetic driving set (curving road lanes, vehicles, lane markers) that allows 100% end-to-end training and test pipeline verification without blocking.
