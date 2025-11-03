"""
Comprehensive test script for DINOv2 + Mask R-CNN (518x518 setup)
Evaluates a trained model on a dataset split, computes metrics, and
optionally generates qualitative visualizations with predicted masks.

Based on the Faster R-CNN 224x224 evaluation workflow.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_maskrcnn_improved import collate_fn_maskrcnn, compute_metrics


def _find_existing_path(*candidates: Path) -> Optional[Path]:
    """Return the first existing path from candidates, or None."""

    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_split_paths(data_root: Path, split: str) -> Tuple[Path, Path, Optional[Path]]:
    """Resolve image, label, and annotation paths for a dataset split."""

    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Dataset split '{split}' not found under {data_root}")

    image_dir = _find_existing_path(
        split_root / "images",
        split_root / "Images",
        split_root / "imgs",
        split_root / "Imgs",
    )
    label_dir = _find_existing_path(
        split_root / "labels",
        split_root / "Labels",
        split_root / "masks",
        split_root / "Masks",
    )

    if image_dir is None or label_dir is None:
        missing = []
        if image_dir is None:
            missing.append("images")
        if label_dir is None:
            missing.append("labels")
        raise FileNotFoundError(
            f"Could not locate {', '.join(missing)} directory for split '{split}' under {split_root}"
        )

    annotation_candidates = [
        split_root / f"{split}_annotations.json",
        split_root / f"{split}_annotation.json",
        split_root / "annotations.json",
        split_root / "annotation.json",
    ]
    annotation_path = _find_existing_path(*annotation_candidates)

    return image_dir, label_dir, annotation_path


def visualize_test_predictions(
    images: Iterable[torch.Tensor],
    predictions: Iterable[Dict[str, torch.Tensor]],
    targets: Iterable[Dict[str, torch.Tensor]],
    output_dir: Path,
    image_names: Iterable[str],
    score_threshold: float = 0.05,
) -> None:
    """Visualize predictions, ground-truth boxes, and masks for qualitative review."""

    output_dir = Path(output_dir)
    vis_dir = output_dir / "test_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_tensor, pred, target, name) in enumerate(
        zip(images, predictions, targets, image_names)
    ):
        img = img_tensor.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        if img.max() > 1.0:
            img = np.clip(img, 0, 1)

        pred_boxes = pred["boxes"].cpu().numpy()
        pred_scores = pred["scores"].cpu().numpy()
        pred_masks = pred.get("masks")
        pred_masks = pred_masks.cpu().numpy() if pred_masks is not None else None

        target_boxes_norm = target["boxes"].cpu().numpy()
        orig_size = target.get("orig_size")
        if orig_size is not None:
            orig_h, orig_w = orig_size.cpu().numpy()
        else:
            orig_h = orig_w = 518

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

        # Ground truth boxes (converted from normalized YOLO format)
        ax1.imshow(img)
        if len(target_boxes_norm) > 0:
            gt_boxes = np.zeros_like(target_boxes_norm)
            gt_boxes[:, 0] = (target_boxes_norm[:, 0] - target_boxes_norm[:, 2] / 2) * orig_w
            gt_boxes[:, 1] = (target_boxes_norm[:, 1] - target_boxes_norm[:, 3] / 2) * orig_h
            gt_boxes[:, 2] = (target_boxes_norm[:, 0] + target_boxes_norm[:, 2] / 2) * orig_w
            gt_boxes[:, 3] = (target_boxes_norm[:, 1] + target_boxes_norm[:, 3] / 2) * orig_h

            scale_x = 518 / orig_w
            scale_y = 518 / orig_h
            gt_boxes[:, [0, 2]] *= scale_x
            gt_boxes[:, [1, 3]] *= scale_y

            ax1.set_title(
                f"Ground Truth ({len(gt_boxes)} nuclei, orig: {int(orig_w)}x{int(orig_h)})",
                fontsize=14,
            )
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green", facecolor="none"
                )
                ax1.add_patch(rect)
        else:
            ax1.set_title("Ground Truth (0 nuclei)", fontsize=14)
        ax1.axis("off")

        # Predicted boxes (filtered by threshold)
        ax2.imshow(img)
        keep = pred_scores > score_threshold
        filtered_boxes = pred_boxes[keep]
        filtered_scores = pred_scores[keep]

        ax2.set_title(
            f"Predictions (>{score_threshold:.2f}, {len(filtered_boxes)} boxes)", fontsize=14
        )
        for box, score in zip(filtered_boxes, filtered_scores):
            x1, y1, x2, y2 = box
            color = plt.cm.RdYlGn(score)
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax2.add_patch(rect)
            ax2.text(
                x1,
                y1 - 5,
                f"{score:.2f}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7),
            )
        ax2.axis("off")

        # Predicted masks
        ax3.imshow(img)
        if pred_masks is not None and len(pred_masks) > 0 and keep.any():
            pred_masks = pred_masks[keep]
            combined_mask = np.zeros((img.shape[0], img.shape[1], 3))
            colors = plt.cm.Set3(np.linspace(0, 1, len(pred_masks)))

            for mask, color in zip(pred_masks, colors):
                mask_2d = mask[0]
                combined_mask[mask_2d > 0.5] = color[:3]

            ax3.imshow(combined_mask, alpha=0.5)
            ax3.set_title(f"Masks ({len(pred_masks)} instances)", fontsize=14)
        else:
            ax3.set_title("Masks (none above threshold)", fontsize=14)
        ax3.axis("off")

        plt.tight_layout()
        image_stem = Path(name).stem if name else f"sample_{idx}"
        plt.savefig(vis_dir / f"{image_stem}_prediction.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict:
    """Load checkpoint and associated configuration metadata."""

    # Older checkpoints were saved with Python objects that require full unpickling.
    # PyTorch 2.6 defaults to `weights_only=True`, which rejects these objects.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    output_parent = checkpoint_path.parent
    config_path = output_parent / "config.json"

    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        print("Warning: config.json not found alongside checkpoint; using CLI defaults where possible.")

    return checkpoint, config


def test_model(
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path,
    split: str = "test",
    batch_size: int = 2,
    score_threshold: float = 0.05,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    visualize: bool = True,
    max_visualizations: int = 12,
) -> Dict[str, float]:
    """Run evaluation on the specified dataset split."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    checkpoint, config = load_checkpoint_metadata(checkpoint_path)

    # Instantiate model using saved configuration
    from dinov2_maskrcnn_improved import create_improved_maskrcnn

    dinov2_checkpoint = config.get("dinov2_checkpoint")
    if dinov2_checkpoint is None:
        raise RuntimeError(
            "dinov2_checkpoint not found in config.json; please specify a checkpoint trained with train_maskrcnn_improved.py"
        )

    freeze_backbone = config.get("freeze_backbone", True)

    print("\nBuilding model...")
    model = create_improved_maskrcnn(
        dinov2_checkpoint_path=dinov2_checkpoint,
        freeze_backbone=freeze_backbone,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    epoch = checkpoint.get("epoch", "unknown")
    train_loss = checkpoint.get("train_loss", "N/A")
    f1_score = checkpoint.get("f1_score", "N/A")

    print("Checkpoint info:")
    print(f"  Epoch: {epoch}")
    print(f"  Train Loss: {train_loss}")
    print(f"  F1 Score: {f1_score}")

    # Prepare dataloader
    from hover_dataset_loader import HoverNetDataset, get_transform

    try:
        image_dir, label_dir, annotation_path = _resolve_split_paths(data_root, split)
    except FileNotFoundError as exc:
        if split != "valid":
            print(f"Split '{split}' not found: {exc}. Falling back to 'valid'.")
            image_dir, label_dir, annotation_path = _resolve_split_paths(data_root, "valid")
            split = "valid"
        else:
            raise

    dataset = HoverNetDataset(
        images_dir=str(image_dir),
        labels_dir=str(label_dir),
        annotations_file=str(annotation_path) if annotation_path else None,
        transform=get_transform(train=False, target_size=518),
        train=False,
        augmentation_multiplier=1,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_maskrcnn,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    print(f"\nSplit: {split}")
    print(f"Samples: {len(dataset)}, batches: {len(dataloader)}")

    all_predictions: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []
    detection_counts: List[int] = []
    confidence_scores: List[float] = []
    mask_pixels: List[float] = []
    mask_fractions: List[float] = []

    visual_images: List[torch.Tensor] = []
    visual_predictions: List[Dict[str, torch.Tensor]] = []
    visual_targets: List[Dict[str, torch.Tensor]] = []
    visual_names: List[str] = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Testing")):
            images_on_device = [img.to(device) for img in images]
            outputs = model(images_on_device)

            outputs_cpu = [
                {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in out.items()}
                for out in outputs
            ]
            targets_cpu = [
                {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()}
                for target in targets
            ]

            all_predictions.extend(outputs_cpu)
            all_targets.extend(targets_cpu)

            for pred in outputs_cpu:
                num_dets = len(pred["boxes"])
                detection_counts.append(num_dets)
                if num_dets > 0:
                    confidence_scores.extend(pred["scores"].numpy().tolist())
                    masks_tensor = pred.get("masks")
                    if masks_tensor is not None and len(masks_tensor) > 0:
                        masks_np = masks_tensor.numpy()
                        binary_masks = (masks_np > 0.5).astype(np.float32)
                        flattened = binary_masks.reshape(len(masks_np), -1)
                        mask_pixels.extend(flattened.sum(axis=1).tolist())
                        mask_fractions.extend(flattened.mean(axis=1).tolist())

            if visualize and len(visual_images) < max_visualizations:
                for img_idx, img in enumerate(images):
                    if len(visual_images) >= max_visualizations:
                        break
                    global_idx = batch_idx * batch_size + img_idx
                    image_name = (
                        str(dataset.image_files[global_idx])
                        if hasattr(dataset, "image_files") and global_idx < len(dataset.image_files)
                        else f"image_{global_idx}"
                    )
                    visual_images.append(img.cpu())
                    visual_predictions.append(outputs_cpu[img_idx])
                    visual_targets.append(targets_cpu[img_idx])
                    visual_names.append(image_name)

    print("\nComputing metrics...")
    metrics = compute_metrics(
        all_predictions,
        all_targets,
        iou_threshold=0.5,
        score_threshold=score_threshold,
    )

    avg_detections = float(np.mean(detection_counts)) if detection_counts else 0.0
    avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    avg_mask_pixels = float(np.mean(mask_pixels)) if mask_pixels else 0.0
    avg_mask_fraction = float(np.mean(mask_fractions)) if mask_fractions else 0.0
    max_detections = int(np.max(detection_counts)) if detection_counts else 0
    min_detections = int(np.min(detection_counts)) if detection_counts else 0

    print("\n" + "=" * 70)
    print("MASK R-CNN TEST RESULTS")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    print(f"Samples: {len(dataset)}")
    print(f"Score threshold (metrics): {score_threshold}")
    print("\nMetrics (IoU=0.5):")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print("\nCounts:")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print("\nDetection statistics:")
    print(f"  Avg detections/image: {avg_detections:.1f}")
    print(f"  Min/Max detections:   {min_detections}/{max_detections}")
    print(f"  Avg confidence:       {avg_confidence:.3f}")
    if mask_pixels:
        print("\nMask statistics:")
        print(f"  Avg mask pixels:   {avg_mask_pixels:.1f}")
        print(f"  Avg mask fraction: {avg_mask_fraction:.4f}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "test_results.json"
    results = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "samples": len(dataset),
        "metrics": {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
        },
        "statistics": {
            "avg_detections": avg_detections,
            "min_detections": min_detections,
            "max_detections": max_detections,
            "avg_confidence": avg_confidence,
            "avg_mask_pixels": avg_mask_pixels,
            "avg_mask_fraction": avg_mask_fraction,
        },
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    if visualize and visual_images:
        print("\nGenerating visualizations...")
        visualize_test_predictions(
            visual_images,
            visual_predictions,
            visual_targets,
            output_dir,
            visual_names,
            score_threshold=score_threshold,
        )
        print(f"Visualizations saved to: {output_dir / 'test_visualizations'}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Mask R-CNN model on a dataset split"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset root containing split subdirectories",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test; falls back to valid if missing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store evaluation outputs (default: alongside checkpoint)",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.05,
        help="Score threshold used for reporting metrics and visualizations",
    )
    parser.add_argument("--no_visualize", action="store_true", help="Skip qualitative visualizations")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument(
        "--max_visualizations",
        type=int,
        default=12,
        help="Maximum number of images to visualize (default: 12)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = checkpoint_path.parent / f"test_results_{timestamp}"

    output_dir = Path(args.output_dir).resolve()
    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Starting Mask R-CNN evaluation...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data root:  {data_root}")
    print(f"Split:      {args.split}")
    print(f"Output dir: {output_dir}")

    metrics = test_model(
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        output_dir=output_dir,
        split=args.split,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
        num_workers=args.num_workers,
        device=device,
        visualize=not args.no_visualize,
        max_visualizations=args.max_visualizations,
    )

    print("\n✅ Evaluation complete")
    print(f"Final F1 Score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()


