#!/usr/bin/env python
"""
Visualize bounding boxes defined in a COCO-style annotations JSON.

Example:
    python scripts/visualize_dataset_bboxes.py \
        --annotations /path/to/train_annotations.json \
        --images_dir /path/to/train/images \
        --file_name 3065_4419_46_37.png \
        --output /tmp/3065_4419_boxes.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def load_annotations(annotations_path: Path) -> Dict:
    with annotations_path.open("r") as f:
        data = json.load(f)

    if "images" not in data or "annotations" not in data:
        raise ValueError(
            f"Annotation file {annotations_path} is missing required keys 'images' and 'annotations'."
        )

    return data


def select_image(
    data: Dict, image_id: Optional[int] = None, file_name: Optional[str] = None
) -> Dict:
    images: List[Dict] = data["images"]

    if file_name:
        for img in images:
            if img["file_name"] == file_name:
                return img
        raise ValueError(f"Image with file_name='{file_name}' not found in annotations.")

    if image_id is not None:
        for img in images:
            if img["id"] == image_id:
                return img
        raise ValueError(f"Image with id={image_id} not found in annotations.")

    # Default: return first image
    if not images:
        raise ValueError("No images found in annotations file.")
    return images[0]


def draw_boxes(
    image_path: Path,
    bboxes: List[List[float]],
    scores: Optional[List[float]] = None,
    output_path: Optional[Path] = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_np)
    ax.axis("off")

    for idx, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        if scores is not None:
            ax.text(
                x,
                y - 4,
                f"{scores[idx]:.2f}",
                color="black",
                fontsize=8,
                bbox=dict(facecolor="lime", alpha=0.5, pad=2),
            )

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize bounding boxes from COCO-style dataset annotations."
    )
    parser.add_argument("--annotations", type=Path, required=True, help="Path to annotations JSON.")
    parser.add_argument(
        "--images_dir",
        type=Path,
        required=True,
        help="Directory containing the corresponding images.",
    )
    parser.add_argument(
        "--image_id",
        type=int,
        default=None,
        help="ID of image to visualize (matches the 'id' field in annotations).",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="File name of image to visualize (overrides --image_id if provided).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the visualization (PNG). If omitted, displays the figure.",
    )

    args = parser.parse_args()

    data = load_annotations(args.annotations)
    image_info = select_image(data, image_id=args.image_id, file_name=args.file_name)

    image_path = args.images_dir / image_info["file_name"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_id = image_info["id"]
    bboxes = [ann["bbox"] for ann in data["annotations"] if ann["image_id"] == image_id]

    if not bboxes:
        raise ValueError(f"No annotations found for image_id={image_id} ({image_path.name}).")

    draw_boxes(image_path, bboxes, output_path=args.output)


if __name__ == "__main__":
    main()

