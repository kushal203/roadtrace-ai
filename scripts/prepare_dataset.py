"""
RoadTrace AI - Enhanced Dataset Preparation & Synthetic Generator Script

Generates a diversified driving perception dataset:
1. 70% Diverse Road Driving Scenes:
   - Asphalt textures, road curvature (straight, curve left, curve right)
   - Lighting variations: daylight, dusk, overcast, dynamic shadows
   - Lane markings, road shoulders, adjacent lanes
   - Class 1 (Direct Drivable / Ego), Class 2 (Alternative Drivable), Class 0 (Background)
2. 30% Hard Negative Non-Road Scenes (Ground Truth: 100% Class 0):
   - Staircases & steps (horizontal steps, riser shadows, vertical railings/balusters)
   - Indoor corridors, floors & walls (tile grids, wooden floorboards, skirting)
   - Outdoor terrain (grass lawns, gravel, sidewalks, foliage)
"""

import os
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def add_texture_noise(img: Image.Image, sigma: float = 6.0) -> Image.Image:
    """Adds subtle visual grain/texture to make surfaces realistic."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


def create_diverse_road_sample(idx: int, img_size: int = 256):
    """Generates varied roadway scene with direct and alternative drivable lanes."""
    # Sky lighting variation (bright day, dusk, overcast)
    lighting = random.choice(['day', 'overcast', 'dusk', 'bright'])
    if lighting == 'day':
        sky_color = (135, 206, 235)
        asphalt_base = (70, 75, 80)
    elif lighting == 'overcast':
        sky_color = (160, 165, 175)
        asphalt_base = (60, 62, 65)
    elif lighting == 'dusk':
        sky_color = (220, 140, 110)
        asphalt_base = (50, 52, 58)
    else:
        sky_color = (100, 180, 240)
        asphalt_base = (80, 85, 90)

    img = Image.new('RGB', (img_size, img_size), color=sky_color)
    draw = ImageDraw.Draw(img)

    horizon_y = int(img_size * (0.42 + random.uniform(-0.05, 0.05)))
    ground_color = (45, 55, 45) # roadside / terrain
    draw.rectangle([0, horizon_y, img_size, img_size], fill=ground_color)

    # Segmentation mask
    mask = Image.new('L', (img_size, img_size), color=0)
    draw_mask = ImageDraw.Draw(mask)

    # Vanishing point
    vp_x = int(img_size * (0.5 + random.uniform(-0.15, 0.15)))
    vp_y = horizon_y

    # Road curvature offset
    curve_bias = random.choice([-35, -20, -10, 0, 10, 20, 35])
    ego_left_x = int(img_size * (0.24 + random.uniform(-0.04, 0.04))) + curve_bias // 2
    ego_right_x = int(img_size * (0.76 + random.uniform(-0.04, 0.04))) + curve_bias // 2

    # Alternative lanes (Class 2)
    has_alt_left = random.random() > 0.45
    has_alt_right = random.random() > 0.45

    if has_alt_left:
        alt_left_x = max(0, ego_left_x - int(img_size * random.uniform(0.18, 0.25)))
        alt_poly = [(vp_x - 12, vp_y), (vp_x, vp_y), (ego_left_x, img_size), (alt_left_x, img_size)]
        draw_mask.polygon(alt_poly, fill=2)
        draw.polygon(alt_poly, fill=(asphalt_base[0] - 10, asphalt_base[1] - 10, asphalt_base[2] - 10))

    if has_alt_right:
        alt_right_x = min(img_size, ego_right_x + int(img_size * random.uniform(0.18, 0.25)))
        alt_poly = [(vp_x, vp_y), (vp_x + 12, vp_y), (alt_right_x, img_size), (ego_right_x, img_size)]
        draw_mask.polygon(alt_poly, fill=2)
        draw.polygon(alt_poly, fill=(asphalt_base[0] - 10, asphalt_base[1] - 10, asphalt_base[2] - 10))

    # Direct drivable ego lane (Class 1)
    ego_poly = [(vp_x - 8, vp_y), (vp_x + 8, vp_y), (ego_right_x, img_size), (ego_left_x, img_size)]
    draw_mask.polygon(ego_poly, fill=1)
    draw.polygon(ego_poly, fill=asphalt_base)

    # Road lane markings
    dash_color = (235, 235, 240) if random.random() > 0.25 else (230, 190, 50)
    for step in range(6):
        t1 = step / 6.0
        t2 = (step + 0.55) / 6.0
        y1 = int(vp_y + (img_size - vp_y) * (t1 ** 2))
        y2 = int(vp_y + (img_size - vp_y) * (t2 ** 2))
        x1_l = int(vp_x - 8 + (ego_left_x - (vp_x - 8)) * t1)
        x2_l = int(vp_x - 8 + (ego_left_x - (vp_x - 8)) * t2)
        draw.line([(x1_l, y1), (x2_l, y2)], fill=dash_color, width=max(1, int(t2 * 3.5)))

        x1_r = int(vp_x + 8 + (ego_right_x - (vp_x + 8)) * t1)
        x2_r = int(vp_x + 8 + (ego_right_x - (vp_x + 8)) * t2)
        draw.line([(x1_r, y1), (x2_r, y2)], fill=dash_color, width=max(1, int(t2 * 3.5)))

    # Cross shadows (e.g. tree or overpass shadows)
    if random.random() > 0.4:
        shadow_y = int(img_size * random.uniform(0.5, 0.85))
        shadow_h = int(img_size * random.uniform(0.04, 0.12))
        draw.rectangle([0, shadow_y, img_size, shadow_y + shadow_h], fill=(30, 32, 35))

    # Add realistic texture noise
    img = add_texture_noise(img, sigma=random.uniform(4.0, 7.0))
    return img, mask


def create_staircase_sample(idx: int, img_size: int = 256):
    """
    Hard negative sample: Staircase with treads, risers, shadows and railings.
    Ground truth mask: 100% Class 0 (Background / Non-Drivable).
    """
    img = Image.new('RGB', (img_size, img_size), color=(180, 175, 170)) # Wall / background
    draw = ImageDraw.Draw(img)

    # Stair material color (wood, concrete, carpet)
    mat_type = random.choice(['wood', 'carpet', 'concrete', 'tile'])
    if mat_type == 'wood':
        tread_color = (130, 85, 50)
        riser_color = (85, 50, 30)
    elif mat_type == 'carpet':
        tread_color = (90, 70, 95)
        riser_color = (60, 45, 65)
    elif mat_type == 'concrete':
        tread_color = (140, 140, 145)
        riser_color = (95, 95, 100)
    else:
        tread_color = (200, 195, 185)
        riser_color = (140, 135, 130)

    # Draw steps
    num_steps = random.randint(8, 14)
    step_height = img_size // num_steps

    for s in range(num_steps):
        top_y = s * step_height
        # Tread (horizontal step surface)
        draw.rectangle([0, top_y, img_size, top_y + step_height // 2], fill=tread_color)
        # Riser (vertical step face with drop shadow)
        draw.rectangle([0, top_y + step_height // 2, img_size, top_y + step_height], fill=riser_color)
        # Shadow line under nosing
        draw.line([(0, top_y + step_height // 2), (img_size, top_y + step_height // 2)], fill=(20, 20, 20), width=2)

    # Draw vertical railings/balusters
    railing_color = random.choice([(40, 40, 40), (220, 220, 225), (120, 70, 35)])
    railing_side = random.choice(['left', 'right', 'both'])
    if railing_side in ['left', 'both']:
        for col in range(15, 60, 12):
            draw.line([(col, 0), (col, img_size)], fill=railing_color, width=3)
        draw.line([(5, 0), (70, img_size)], fill=railing_color, width=6) # Handrail

    if railing_side in ['right', 'both']:
        for col in range(img_size - 60, img_size - 15, 12):
            draw.line([(col, 0), (col, img_size)], fill=railing_color, width=3)
        draw.line([(img_size - 70, 0), (img_size - 5, img_size)], fill=railing_color, width=6) # Handrail

    # Mask is 100% Class 0 (zero drivable road)
    mask = Image.new('L', (img_size, img_size), color=0)
    img = add_texture_noise(img, sigma=5.0)
    return img, mask


def create_indoor_floor_sample(idx: int, img_size: int = 256):
    """
    Hard negative sample: Indoor room with parquet floor, tiles, skirting and walls.
    Ground truth mask: 100% Class 0 (Background / Non-Drivable).
    """
    wall_color = random.choice([(215, 210, 205), (195, 205, 215), (225, 220, 210), (180, 190, 185)])
    img = Image.new('RGB', (img_size, img_size), color=wall_color)
    draw = ImageDraw.Draw(img)

    # Floor horizon
    floor_y = int(img_size * random.uniform(0.35, 0.50))
    floor_type = random.choice(['tiles', 'wood', 'carpet'])

    if floor_type == 'wood':
        floor_color = (150, 100, 60)
        draw.rectangle([0, floor_y, img_size, img_size], fill=floor_color)
        # Floorboard seams
        for x in range(0, img_size, random.randint(18, 30)):
            draw.line([(x, floor_y), (x, img_size)], fill=(100, 65, 35), width=1)
    elif floor_type == 'tiles':
        floor_color = (210, 205, 200)
        draw.rectangle([0, floor_y, img_size, img_size], fill=floor_color)
        grid_size = random.randint(25, 45)
        for x in range(0, img_size, grid_size):
            draw.line([(x, floor_y), (x, img_size)], fill=(130, 125, 120), width=2)
        for y in range(floor_y, img_size, grid_size):
            draw.line([(0, y), (img_size, y)], fill=(130, 125, 120), width=2)
    else:
        floor_color = (80, 85, 95)
        draw.rectangle([0, floor_y, img_size, img_size], fill=floor_color)

    # Skirting board
    draw.rectangle([0, floor_y - 8, img_size, floor_y], fill=(240, 240, 240))
    draw.line([(0, floor_y), (img_size, floor_y)], fill=(50, 50, 50), width=1)

    # Door or window frame
    if random.random() > 0.4:
        door_x = int(img_size * random.uniform(0.2, 0.6))
        draw.rectangle([door_x, floor_y - 100, door_x + 50, floor_y], fill=(120, 75, 40))

    mask = Image.new('L', (img_size, img_size), color=0)
    img = add_texture_noise(img, sigma=4.0)
    return img, mask


def create_outdoor_nonroad_sample(idx: int, img_size: int = 256):
    """
    Hard negative sample: Lawns, parks, gravel, building facades.
    Ground truth mask: 100% Class 0 (Background / Non-Drivable).
    """
    img = Image.new('RGB', (img_size, img_size), color=(120, 180, 230))
    draw = ImageDraw.Draw(img)

    horizon_y = int(img_size * 0.4)
    # Grass / vegetation / dirt
    ground_color = random.choice([(50, 110, 45), (120, 100, 70), (80, 95, 80)])
    draw.rectangle([0, horizon_y, img_size, img_size], fill=ground_color)

    # Trees or bushes
    for _ in range(5):
        cx = random.randint(20, img_size - 20)
        cy = random.randint(horizon_y - 20, horizon_y + 40)
        r = random.randint(15, 35)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 80, 30))

    mask = Image.new('L', (img_size, img_size), color=0)
    img = add_texture_noise(img, sigma=6.0)
    return img, mask


def prepare_balanced_dataset(data_root: str, num_train: int = 1000, num_val: int = 200):
    """
    Prepares a balanced dataset containing 70% diverse roads and 30% hard negatives
    (stairs, indoor floors, outdoor non-road surfaces).
    """
    train_img_dir = os.path.join(data_root, 'images', '10k', 'train')
    train_mask_dir = os.path.join(data_root, 'drivable_maps', 'labels', 'train')
    val_img_dir = os.path.join(data_root, 'images', '10k', 'val')
    val_mask_dir = os.path.join(data_root, 'drivable_maps', 'labels', 'val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    print(f"Generating balanced dataset: {num_train} train + {num_val} val images...")
    print("Composition: 70% diverse roads, 10% staircases, 10% indoor floors, 10% outdoor terrain.")

    # 1. Generate Training Set
    road_train_count = int(num_train * 0.70)
    stairs_train_count = int(num_train * 0.10)
    floor_train_count = int(num_train * 0.10)
    outdoor_train_count = num_train - (road_train_count + stairs_train_count + floor_train_count)

    train_generators = (
        [('road', create_diverse_road_sample)] * road_train_count +
        [('stairs', create_staircase_sample)] * stairs_train_count +
        [('floor', create_indoor_floor_sample)] * floor_train_count +
        [('outdoor', create_outdoor_nonroad_sample)] * outdoor_train_count
    )
    random.seed(42)
    random.shuffle(train_generators)

    for i, (cat, gen_fn) in enumerate(train_generators):
        img_name = f"train_{cat}_{i:05d}.jpg"
        mask_name = f"train_{cat}_{i:05d}_drivable_id.png"
        img, mask = gen_fn(i)
        img.save(os.path.join(train_img_dir, img_name), quality=92)
        mask.save(os.path.join(train_mask_dir, mask_name))
        if (i + 1) % 250 == 0 or i == num_train - 1:
            print(f"  Train: [{i+1}/{num_train}] generated")

    # 2. Generate Validation Set
    road_val_count = int(num_val * 0.70)
    stairs_val_count = int(num_val * 0.10)
    floor_val_count = int(num_val * 0.10)
    outdoor_val_count = num_val - (road_val_count + stairs_val_count + floor_val_count)

    val_generators = (
        [('road', create_diverse_road_sample)] * road_val_count +
        [('stairs', create_staircase_sample)] * stairs_val_count +
        [('floor', create_indoor_floor_sample)] * floor_val_count +
        [('outdoor', create_outdoor_nonroad_sample)] * outdoor_val_count
    )
    random.seed(999)
    random.shuffle(val_generators)

    for i, (cat, gen_fn) in enumerate(val_generators):
        img_name = f"val_{cat}_{i:04d}.jpg"
        mask_name = f"val_{cat}_{i:04d}_drivable_id.png"
        img, mask = gen_fn(i + 50000)
        img.save(os.path.join(val_img_dir, img_name), quality=92)
        mask.save(os.path.join(val_mask_dir, mask_name))
        if (i + 1) % 50 == 0 or i == num_val - 1:
            print(f"  Val: [{i+1}/{num_val}] generated")

    print("Balanced dataset preparation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/bdd100k', help='Dataset root path')
    parser.add_argument('--train_count', type=int, default=1000)
    parser.add_argument('--val_count', type=int, default=200)
    args = parser.parse_args()

    prepare_balanced_dataset(args.data_root, args.train_count, args.val_count)
