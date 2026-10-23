"""
Phase 1 Audit Script:
1. Loads checkpoint models/checkpoints/best_model.pth
2. Inspects architecture, layer shapes, parameter counts
3. Checks training_log.json
4. Evaluates on current validation set (synth_val)
5. Tests model response on synthetic vs real-world / non-road images (stairs, floor, white noise, black)
6. Measures exact class prediction distribution and false positives
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RoadSegmentationNet
from dataset import BDD100KDrivableDataset

def run_audit():
    print("=" * 60)
    print("ROADTRACE AI — PHASE 1 MODEL & DATASET AUDIT")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Audit Device: {device}")
    
    # 1. Inspect Checkpoint
    ckpt_path = "models/checkpoints/best_model.pth"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"\n--- Checkpoint Metadata ---")
    print(f"Epoch: {ckpt.get('epoch')}")
    print(f"Recorded best_val_iou in ckpt: {ckpt.get('best_val_iou')}")
    print(f"Config stored in ckpt: {ckpt.get('config')}")
    
    # Check parameters
    model = RoadSegmentationNet(pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # 2. Check Training Log
    log_path = "models/checkpoints/training_log.json"
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            logs = json.load(f)
        print(f"\n--- Training Log Entries ({len(logs)} epochs) ---")
        for entry in logs:
            ep = entry['epoch']
            t_loss = entry['train']['loss']
            v_loss = entry['val']['loss']
            v_iou = entry['val']['iou']
            print(f"  Epoch {ep}: Train Loss={t_loss:.4f}, Val Loss={v_loss:.4f}, Val mIoU={v_iou.get('mean_iou')}")
            print(f"           Per-class Val IoU: {v_iou}")
    
    # 3. Check Dataset Files
    train_dir = "data/bdd100k/images/10k/train"
    val_dir = "data/bdd100k/images/10k/val"
    train_files = sorted(os.listdir(train_dir))
    val_files = sorted(os.listdir(val_dir))
    
    print(f"\n--- Dataset Audit ---")
    print(f"Train directory: {train_dir} ({len(train_files)} files)")
    print(f"Sample train filenames: {train_files[:5]}")
    print(f"Val directory:   {val_dir} ({len(val_files)} files)")
    print(f"Sample val filenames:   {val_files[:5]}")
    
    # Check whether files are synthetic
    all_synth_train = all(f.startswith("synth_") for f in train_files)
    all_synth_val = all(f.startswith("synth_val_") for f in val_files)
    print(f"Are train images 100% synthetic generator output? {all_synth_train}")
    print(f"Are val images 100% synthetic generator output?   {all_synth_val}")
    
    # 4. Check masks
    mask_val_dir = "data/bdd100k/drivable_maps/labels/val"
    mask_files = sorted(os.listdir(mask_val_dir))
    sample_mask_path = os.path.join(mask_val_dir, mask_files[0])
    sample_mask = np.array(Image.open(sample_mask_path))
    unique_vals = np.unique(sample_mask)
    print(f"Sample validation mask unique pixel values: {unique_vals}")
    print(f"Sample validation mask shape: {sample_mask.shape}")
    
    # 5. Evaluate on Validation Set
    val_dataset = BDD100KDrivableDataset(
        images_dir=val_dir,
        masks_dir=mask_val_dir,
        img_size=256,
        subset_size=200,
        augment=False,
    )
    
    confusion = torch.zeros(3, 3, dtype=torch.long)
    with torch.no_grad():
        for i in range(len(val_dataset)):
            img_tensor, mask_tensor = val_dataset[i]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            logits = model(img_tensor)
            preds = logits.argmax(dim=1).squeeze(0).cpu()
            
            for c in range(3):
                p_c = (preds == c)
                t_c = (mask_tensor == c)
                confusion[c, c] += (p_c & t_c).sum().item()
                for o in range(3):
                    if o != c:
                        confusion[c, o] += (p_c & (mask_tensor == o)).sum().item()
                        
    ious = {}
    names = ['background', 'direct_drivable', 'alt_drivable']
    for c in range(3):
        tp = confusion[c, c].item()
        fp = confusion[c, :].sum().item() - tp
        fn = confusion[:, c].sum().item() - tp
        iou = tp / (tp + fp + fn + 1e-8)
        ious[names[c]] = round(iou, 4)
    ious['mean_iou'] = round(sum(ious.values()) / 3, 4)
    
    print(f"\n--- Recalculated Validation Metrics on Current Dataset ---")
    print(f"Confusion Matrix (Pred vs True):\n{confusion}")
    print(f"Per-Class IoU: {ious}")
    print(f"Drivable Area (Class 1 + Class 2) combined IoU:")
    # Combined drivable
    tp_drivable = (confusion[1, 1] + confusion[1, 2] + confusion[2, 1] + confusion[2, 2]).item()
    fp_drivable = (confusion[1, 0] + confusion[2, 0]).item()
    fn_drivable = (confusion[0, 1] + confusion[0, 2]).item()
    drivable_iou = tp_drivable / (tp_drivable + fp_drivable + fn_drivable + 1e-8)
    print(f"Combined Drivable IoU: {drivable_iou:.4f}")
    
    # 6. Test on Non-Road Patterns (Simulating Staircases, Indoor Floors, Blanks)
    print(f"\n--- Testing Model on Non-Road & Synthetic Staircase Patterns ---")
    
    # Scenario A: Staircase simulation (horizontal stepped bands with railings)
    stairs = np.zeros((256, 256, 3), dtype=np.uint8)
    for row in range(0, 256, 16):
        shade = 80 + (row % 32) * 3
        stairs[row:row+16, :] = shade
    # Add vertical lines (railings / banisters)
    for col in range(20, 240, 30):
        stairs[:, col:col+3] = 200
    
    # Scenario B: Flat indoor floor with wall (horizontal horizon, wooden/tile color)
    floor = np.zeros((256, 256, 3), dtype=np.uint8)
    floor[:120, :] = [210, 200, 190] # Wall
    floor[120:, :] = [160, 120, 80]  # Floor
    # Tile seams
    for col in range(0, 256, 40):
        floor[120:, col:col+2] = [100, 80, 50]
        
    # Scenario C: Pure uniform white wall
    white_wall = np.full((256, 256, 3), 240, dtype=np.uint8)
    
    test_cases = [
        ("Simulated Staircase", stairs),
        ("Indoor Floor & Wall", floor),
        ("Blank White Wall", white_wall),
    ]
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    os.makedirs("docs/audit_artifacts", exist_ok=True)
    
    for name, img_arr in test_cases:
        # Save test input
        clean_name = name.lower().replace(" ", "_").replace("&", "and")
        Image.fromarray(img_arr).save(f"docs/audit_artifacts/{clean_name}_input.png")
        
        # Preprocess exactly as in training / inference
        norm_img = ((img_arr / 255.0) - mean) / std
        t = torch.from_numpy(norm_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(t)
            probs = F.softmax(out, dim=1).squeeze(0).cpu().numpy()
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
            
        bg_pct = (pred == 0).mean() * 100
        direct_pct = (pred == 1).mean() * 100
        alt_pct = (pred == 2).mean() * 100
        max_prob_mean = probs.max(axis=0).mean()
        road_prob_bottom = probs[1:, 180:, :].sum(axis=0).mean() # probability of class 1 or 2 in bottom third
        
        print(f"\nTest Scenario: {name}")
        print(f"  Predicted Pixel Distribution: Background={bg_pct:.1f}%, Direct Drivable={direct_pct:.1f}%, Alt Drivable={alt_pct:.1f}%")
        print(f"  Mean Confidence: {max_prob_mean:.3f}")
        print(f"  Bottom-Third Road Probability: {road_prob_bottom:.3f}")
        
        # Save prediction visual
        mask_vis = np.zeros((256, 256, 3), dtype=np.uint8)
        mask_vis[pred == 1] = [0, 200, 100] # Green for direct
        mask_vis[pred == 2] = [0, 100, 200] # Blue for alt
        
        overlay = img_arr.copy()
        mask_active = pred > 0
        overlay[mask_active] = (img_arr[mask_active] * 0.5 + mask_vis[mask_active] * 0.5).astype(np.uint8)
        
        Image.fromarray(overlay).save(f"docs/audit_artifacts/{clean_name}_pred.png")
        Image.fromarray(mask_vis).save(f"docs/audit_artifacts/{clean_name}_mask.png")

if __name__ == '__main__':
    run_audit()
