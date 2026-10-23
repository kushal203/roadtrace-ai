"""
RoadTrace AI - BDD100K Dataset Loader for Road Segmentation

Loads BDD100K drivable area segmentation data.
Expected directory structure:
  data/bdd100k/
    images/10k/
      train/  (*.jpg)
      val/    (*.jpg)
    drivable_maps/
      labels/
        train/  (*_drivable_id.png)
        val/    (*_drivable_id.png)

Mask encoding:
  0 = Background (non-drivable)
  1 = Direct drivable area  
  2 = Alternative drivable area
"""

import os
import glob
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


def resolve_split_dirs(data_root: str, split: str) -> Tuple[str, str]:
    """Find an image/mask split in common BDD100K layouts."""
    image_candidates = [
        os.path.join(data_root, 'images', '100k', split),
        os.path.join(data_root, 'bdd100k', 'images', '100k', split),
        os.path.join(data_root, 'images', '10k', split),
        os.path.join(data_root, 'images', split),
    ]
    mask_candidates = [
        os.path.join(data_root, 'drivable_maps', 'labels', split),
        os.path.join(data_root, 'drivable_maps', split),
        os.path.join(data_root, 'labels', 'drivable', 'masks', split),
        os.path.join(data_root, 'labels', 'drivable', split),
    ]
    image_dir = next((p for p in image_candidates if os.path.isdir(p)), image_candidates[0])
    mask_dir = next((p for p in mask_candidates if os.path.isdir(p)), mask_candidates[0])
    return image_dir, mask_dir


def collect_image_mask_pairs(images_dir: str, masks_dir: str) -> Tuple[List[Tuple[str, str]], int]:
    """Match images with official BDD drivable ID masks without silently hiding misses."""
    image_files = sorted(
        p for p in glob.glob(os.path.join(images_dir, '*'))
        if os.path.splitext(p)[1].lower() in {'.jpg', '.jpeg', '.png'}
    )
    pairs: List[Tuple[str, str]] = []
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        candidates = (
            os.path.join(masks_dir, f'{basename}_drivable_id.png'),
            os.path.join(masks_dir, f'{basename}.png'),
        )
        mask_path = next((p for p in candidates if os.path.isfile(p)), None)
        if mask_path:
            pairs.append((img_path, mask_path))
    return pairs, len(image_files)


class BDD100KDrivableDataset(Dataset):
    """BDD100K Drivable Area Segmentation Dataset."""
    
    NUM_CLASSES = 3
    CLASS_NAMES = ['background', 'direct_drivable', 'alternative_drivable']
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        img_size: int = 256,
        subset_size: Optional[int] = None,
        augment: bool = False,
    ):
        """
        Args:
            images_dir: Path to image directory (e.g., data/bdd100k/images/10k/train)
            masks_dir: Path to mask directory (e.g., data/bdd100k/drivable_maps/labels/train)
            img_size: Target image size (square)
            subset_size: If set, use only this many samples
            augment: Whether to apply data augmentation
        """
        self.img_size = img_size
        self.augment = augment
        
        # Match images to masks
        self.pairs, image_count = collect_image_mask_pairs(images_dir, masks_dir)
        
        if not self.pairs:
            raise FileNotFoundError(
                f"No matching image-mask pairs found.\n"
                f"  Images dir: {images_dir} ({image_count} images)\n"
                f"  Masks dir:  {masks_dir}"
            )

        if len(self.pairs) != image_count:
            print(f"WARNING: matched {len(self.pairs)}/{image_count} images in {images_dir}; "
                  "unmatched images will be excluded")
        
        # Limit to subset
        if subset_size and subset_size < len(self.pairs):
            # Use deterministic selection for reproducibility
            rng = np.random.RandomState(42)
            indices = rng.choice(len(self.pairs), subset_size, replace=False)
            indices.sort()
            self.pairs = [self.pairs[i] for i in indices]
        
        print(f"Dataset: {len(self.pairs)} image-mask pairs "
              f"(img_size={img_size}, augment={augment})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random brightness/contrast
            if torch.rand(1).item() > 0.5:
                image = TF.adjust_brightness(image, 0.8 + torch.rand(1).item() * 0.4)
            if torch.rand(1).item() > 0.5:
                image = TF.adjust_contrast(image, 0.8 + torch.rand(1).item() * 0.4)
        
        # Convert to tensors
        image = TF.to_tensor(image)  # [3, H, W], float [0, 1]
        
        # Normalize with ImageNet stats (matching MobileNetV2 pretrained weights)
        image = TF.normalize(image, 
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        
        # Mask to long tensor
        mask = torch.from_numpy(np.array(mask)).long()
        
        values = torch.unique(mask)
        if torch.any((values < 0) | (values >= self.NUM_CLASSES)):
            raise ValueError(
                f"Invalid drivable mask values in {mask_path}: {values.tolist()}; expected only 0, 1, 2"
            )
        
        return image, mask


def create_dataloaders(
    data_root: str,
    img_size: int = 256,
    batch_size: int = 4,
    subset_size: Optional[int] = None,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_images, train_masks = resolve_split_dirs(data_root, 'train')
    val_images, val_masks = resolve_split_dirs(data_root, 'val')
    
    train_dataset = BDD100KDrivableDataset(
        images_dir=train_images,
        masks_dir=train_masks,
        img_size=img_size,
        subset_size=subset_size,
        augment=True,
    )
    
    val_dataset = BDD100KDrivableDataset(
        images_dir=val_images,
        masks_dir=val_masks,
        img_size=img_size,
        subset_size=max(100, (subset_size or 1000) // 5) if subset_size else None,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def validate_dataset(data_root: str) -> dict:
    """Validate dataset structure and report statistics."""
    report = {
        'valid': True,
        'errors': [],
        'train_images': 0,
        'train_masks': 0,
        'train_pairs': 0,
        'train_unmatched_images': 0,
        'val_images': 0,
        'val_masks': 0,
        'val_pairs': 0,
        'val_unmatched_images': 0,
    }
    
    for split in ['train', 'val']:
        img_dir, mask_dir = resolve_split_dirs(data_root, split)
        
        if not os.path.isdir(img_dir):
            report['errors'].append(f"Missing image directory: {img_dir}")
            report['valid'] = False
            continue
        
        if not os.path.isdir(mask_dir):
            report['errors'].append(f"Missing mask directory: {mask_dir}")
            report['valid'] = False
            continue
        
        pairs, image_count = collect_image_mask_pairs(img_dir, mask_dir)
        images = [p[0] for p in pairs]
        masks = [p[1] for p in pairs]
        
        report[f'{split}_images'] = image_count
        report[f'{split}_masks'] = len(masks)
        report[f'{split}_pairs'] = len(pairs)
        report[f'{split}_unmatched_images'] = image_count - len(pairs)
        
        if not pairs:
            report['errors'].append(f"No matching pairs in {split} split")
            report['valid'] = False
        # Some BDD100K frames are shipped without a drivable-area annotation;
        # the loader deliberately excludes those images from segmentation.
    
    # Check a sample mask
    if report['valid']:
        _, sample_mask_dir = resolve_split_dirs(data_root, 'train')
        sample_pairs, _ = collect_image_mask_pairs(resolve_split_dirs(data_root, 'train')[0], sample_mask_dir)
        if sample_pairs:
            mask = np.array(Image.open(sample_pairs[0][1]))
            unique_vals = np.unique(mask)
            report['mask_unique_values'] = unique_vals.tolist()
            report['mask_shape'] = mask.shape
            if not all(v in [0, 1, 2] for v in unique_vals):
                report['errors'].append(
                    f"Unexpected mask values: {unique_vals}. Expected [0, 1, 2]"
                )
    
    return report


if __name__ == '__main__':
    import sys
    
    data_root = sys.argv[1] if len(sys.argv) > 1 else 'data/bdd100k_real'
    
    print(f"Validating dataset at: {data_root}")
    report = validate_dataset(data_root)
    
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    if report['valid']:
        print("\nDataset validation passed! Creating sample batch...")
        train_loader, val_loader = create_dataloaders(
            data_root, subset_size=10, batch_size=2
        )
        images, masks = next(iter(train_loader))
        print(f"  Image batch: {images.shape}, dtype={images.dtype}")
        print(f"  Mask batch:  {masks.shape}, dtype={masks.dtype}")
        print(f"  Mask unique values: {torch.unique(masks).tolist()}")
    else:
        print("\nDataset validation failed!")
        for err in report['errors']:
            print(f"  ERROR: {err}")
