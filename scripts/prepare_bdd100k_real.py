"""Prepare real BDD100K drivable-area image/mask pairs from the official archive.

The BDD100K images archive contains polygon annotations in its train/val JSON
files. This script rasterizes those polygons into single-channel class masks
compatible with RoadTrace's loader: 0=background, 1=direct, 2=alternative.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


SPLITS = {
    "train": "bdd100k/labels/bdd100k_labels_images_train.json",
    "val": "bdd100k/labels/bdd100k_labels_images_val.json",
}


def make_mask(frame: dict, size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for label in frame.get("labels", []):
        if label.get("category") != "drivable area":
            continue
        area_type = label.get("attributes", {}).get("areaType")
        class_id = {"direct": 1, "alternative": 2}.get(area_type)
        if class_id is None:
            continue
        for polygon in label.get("poly2d", []) or []:
            vertices = polygon.get("vertices", [])
            types = polygon.get("types", "L" * len(vertices))
            if len(vertices) < 3 or not polygon.get("closed", False):
                continue
            # BDD encodes cubic Bezier curves as a run of three C vertices:
            # two control points followed by the curve endpoint.
            points = [vertices[0]]
            index = 1
            while index < len(vertices):
                if types[index] == "L":
                    points.append(vertices[index])
                    index += 1
                    continue
                if types[index:index + 3] != "CCC" or index + 2 >= len(vertices):
                    raise ValueError(f"Malformed BDD poly2d curve in frame {frame.get('name')}")
                p0 = points[-1]
                p1, p2, p3 = vertices[index:index + 3]
                for step in range(1, 13):
                    t = step / 12
                    u = 1 - t
                    points.append((
                        u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0],
                        u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1],
                    ))
                index += 3
            points = [(round(x), round(y)) for x, y in points]
            draw.polygon(points, fill=class_id)
    return mask


def prepare(archive: Path, output_root: Path, max_images: int | None = None) -> None:
    with zipfile.ZipFile(archive) as zf:
        for split, annotation_name in SPLITS.items():
            frames = json.loads(zf.read(annotation_name))
            image_dir = output_root / "bdd100k" / "images" / "100k" / split
            mask_dir = output_root / "drivable_maps" / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            if max_images is not None:
                frames = frames[:max_images]

            print(f"{split}: preparing {len(frames):,} labeled frames", flush=True)
            for index, frame in enumerate(frames, 1):
                name = frame["name"]
                image_path = image_dir / name
                mask_path = mask_dir / f"{Path(name).stem}_drivable_id.png"
                member = f"bdd100k/images/100k/{split}/{name}"

                if not image_path.is_file() or image_path.stat().st_size == 0:
                    with zf.open(member) as source, image_path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)

                with Image.open(image_path) as image:
                    size = image.size
                make_mask(frame, size).save(mask_path, compress_level=1)

                if index % 1000 == 0 or index == len(frames):
                    print(f"{split}: {index:,}/{len(frames):,}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/bdd100k_assembled.zip"))
    parser.add_argument("--output-root", type=Path, default=Path("data/bdd100k_real"))
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional per-split limit for a quick smoke run")
    args = parser.parse_args()
    prepare(args.archive, args.output_root, args.max_images)


if __name__ == "__main__":
    main()
