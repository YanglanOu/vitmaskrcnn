"""
Improved training script for DINOv2 + Mask R-CNN
Fixed: Proper coordinate conversion from normalized YOLO to absolute Mask R-CNN format
Added: Augmentation multiplier support and mask loss tracking
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def collate_fn_maskrcnn(batch):
    """Collate function for Mask R-CNN"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


def compute_metrics(predictions, targets, iou_threshold=0.5, score_threshold=0.05):
    """
    Compute precision, recall, F1 for evaluation (detection metrics)
    Also computes segmentation metrics: mask IoU and Dice coefficient
    Updated for cropped images: boxes are normalized to cropped image size
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Segmentation metrics
    mask_ious = []
    dice_scores = []
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_masks = pred.get('masks')
        if pred_masks is not None:
            pred_masks = pred_masks.cpu().numpy()
        
        # Images are already cropped to target_size (e.g., 224x224) in dataset loader
        # Get image dimensions from orig_size (which is [crop_size, crop_size])
        if 'orig_size' in target:
            img_h, img_w = target['orig_size'].cpu().numpy()
        else:
            # Fallback: assume images are already cropped to 224x224
            img_h, img_w = 224, 224
        
        # Convert normalized YOLO boxes to absolute coordinates
        # Boxes are normalized [cx, cy, w, h] relative to the cropped image
        target_boxes_norm = target['boxes'].cpu().numpy()
        target_masks = target.get('masks')
        if target_masks is not None:
            target_masks = target_masks.cpu().numpy()
        
        if len(target_boxes_norm) > 0:
            # Convert normalized boxes directly to absolute coordinates using cropped image size
            target_boxes = np.zeros_like(target_boxes_norm)
            target_boxes[:, 0] = (target_boxes_norm[:, 0] - target_boxes_norm[:, 2] / 2) * img_w  # x1
            target_boxes[:, 1] = (target_boxes_norm[:, 1] - target_boxes_norm[:, 3] / 2) * img_h  # y1
            target_boxes[:, 2] = (target_boxes_norm[:, 0] + target_boxes_norm[:, 2] / 2) * img_w  # x2
            target_boxes[:, 3] = (target_boxes_norm[:, 1] + target_boxes_norm[:, 3] / 2) * img_h  # y2
        else:
            target_boxes = np.zeros((0, 4))
        
        # Filter predictions by score
        keep = pred_scores > score_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        if pred_masks is not None:
            pred_masks = pred_masks[keep]
        
        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            continue
        elif len(pred_boxes) == 0:
            false_negatives += len(target_boxes)
            continue
        elif len(target_boxes) == 0:
            false_positives += len(pred_boxes)
            continue
        
        # Compute IoU matrix
        from torchvision.ops import box_iou
        ious = box_iou(
            torch.tensor(pred_boxes),
            torch.tensor(target_boxes)
        ).numpy()
        
        # Match predictions to targets and compute mask metrics
        matched_targets = set()
        for i in range(len(pred_boxes)):
            if ious.shape[1] == 0:
                false_positives += 1
                continue
            
            max_iou_idx = ious[i].argmax()
            max_iou = ious[i, max_iou_idx]
            
            if max_iou >= iou_threshold and max_iou_idx not in matched_targets:
                true_positives += 1
                matched_targets.add(max_iou_idx)
                
                # Compute mask IoU and Dice for matched pairs
                if pred_masks is not None and target_masks is not None and len(target_masks) > max_iou_idx:
                    pred_mask = pred_masks[i]
                    gt_mask = target_masks[max_iou_idx]
                    
                    # Resize masks to same size if needed
                    if len(pred_mask.shape) == 3:
                        pred_mask = pred_mask[0]  # Remove channel dimension
                    if len(gt_mask.shape) == 3:
                        gt_mask = gt_mask[0]
                    
                    # Resize to match if shapes differ
                    if pred_mask.shape != gt_mask.shape:
                        zoom_factors = (gt_mask.shape[0] / pred_mask.shape[0], 
                                       gt_mask.shape[1] / pred_mask.shape[1])
                        pred_mask = zoom(pred_mask, zoom_factors, order=1)
                    
                    # Binarize masks
                    pred_binary = (pred_mask > 0.5).astype(np.float32)
                    gt_binary = (gt_mask > 0.5).astype(np.float32)
                    
                    # Compute mask IoU
                    intersection = np.logical_and(pred_binary, gt_binary).sum()
                    union = np.logical_or(pred_binary, gt_binary).sum()
                    if union > 0:
                        mask_iou = intersection / union
                        mask_ious.append(mask_iou)
                    
                    # Compute Dice coefficient
                    pred_sum = pred_binary.sum()
                    gt_sum = gt_binary.sum()
                    if pred_sum + gt_sum > 0:
                        dice = (2 * intersection) / (pred_sum + gt_sum)
                        dice_scores.append(dice)
            else:
                false_positives += 1
        
        false_negatives += len(target_boxes) - len(matched_targets)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Aggregate segmentation metrics
    avg_mask_iou = np.mean(mask_ious) if mask_ious else 0.0
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives,
        'mask_iou': avg_mask_iou,
        'dice': avg_dice,
        'num_matched_masks': len(mask_ious)
    }


def visualize_predictions_vs_targets(images, predictions, targets, epoch, output_dir, num_samples=2):
    """Visualize predictions vs ground truth for debugging - includes masks and segmentation labels"""
    output_dir = Path(output_dir)
    vis_dir = output_dir / 'debug_visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    for idx in range(min(num_samples, len(images))):
        img_tensor = images[idx].detach().cpu()
        img = img_tensor.numpy()
        
        # Denormalize if needed (Mask R-CNN doesn't normalize input)
        if img.max() <= 1.0:
            img = np.transpose(img, (1, 2, 0))
        else:
            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1)
        
        pred = predictions[idx]
        target = targets[idx]
        
        # Determine image spatial dimensions (after cropping)
        # Use actual displayed image dimensions, not orig_size
        img_h_actual, img_w_actual = img.shape[0], img.shape[1]
        img_h_actual = int(img_h_actual)
        img_w_actual = int(img_w_actual)
        
        # Create 2x2 layout: GT boxes, GT masks, Pred boxes, Pred masks
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        
        # Get original dimensions (for logging only)
        if 'orig_size' in target:
            orig_h, orig_w = target['orig_size'].cpu().numpy()
        else:
            orig_h, orig_w = img_h_actual, img_w_actual
        orig_h = int(orig_h)
        orig_w = int(orig_w)
        
        # Ground truth - boxes
        ax1.imshow(img)
        # Ensure axis limits match image dimensions for correct coordinate system
        ax1.set_xlim(-0.5, img_w_actual - 0.5)
        ax1.set_ylim(img_h_actual - 0.5, -0.5)
        target_boxes_norm = target['boxes'].cpu().numpy()
        
        # Debug: Print box values before conversion
        if len(target_boxes_norm) > 0:
            print(f"\n[DEBUG] Epoch {epoch}, Sample {idx}:")
            print(f"  Image dimensions: {img_w_actual}x{img_h_actual}")
            print(f"  Number of boxes: {len(target_boxes_norm)}")
            print(f"  Normalized boxes (first 3): {target_boxes_norm[:3]}")
            print(f"  Max normalized width: {target_boxes_norm[:,2].max():.4f}")
            print(f"  Max normalized height: {target_boxes_norm[:,3].max():.4f}")
        
        if len(target_boxes_norm) > 0:
            # Convert normalized boxes directly to absolute coordinates using actual displayed image dimensions
            gt_boxes = np.zeros_like(target_boxes_norm)
            gt_boxes[:, 0] = (target_boxes_norm[:, 0] - target_boxes_norm[:, 2] / 2) * img_w_actual
            gt_boxes[:, 1] = (target_boxes_norm[:, 1] - target_boxes_norm[:, 3] / 2) * img_h_actual
            gt_boxes[:, 2] = (target_boxes_norm[:, 0] + target_boxes_norm[:, 2] / 2) * img_w_actual
            gt_boxes[:, 3] = (target_boxes_norm[:, 1] + target_boxes_norm[:, 3] / 2) * img_h_actual
            
            # Clip to image bounds to avoid rendering artifacts
            gt_boxes[:, [0, 2]] = np.clip(gt_boxes[:, [0, 2]], 0, img_w_actual)
            gt_boxes[:, [1, 3]] = np.clip(gt_boxes[:, [1, 3]], 0, img_h_actual)
            
            # Debug: Check for suspiciously large boxes
            box_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
            box_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
            print(f"  Absolute boxes (first 3): {gt_boxes[:3]}")
            print(f"  Box widths (pixels, first 5): {box_widths[:5]}")
            print(f"  Box heights (pixels, first 5): {box_heights[:5]}")
            print(f"  Max box width: {box_widths.max():.1f} pixels ({box_widths.max()/img_w_actual*100:.1f}% of image)")
            print(f"  Max box height: {box_heights.max():.1f} pixels ({box_heights.max()/img_h_actual*100:.1f}% of image)")
            if box_widths.max() > img_w_actual * 0.6 or box_heights.max() > img_h_actual * 0.6:
                print(f"  *** WARNING: Large boxes detected! Image size: {img_w_actual}x{img_h_actual}, "
                      f"Max box: {box_widths.max():.1f}x{box_heights.max():.1f}, "
                      f"Normalized max: w={target_boxes_norm[:,2].max():.3f}, h={target_boxes_norm[:,3].max():.3f}")
            
            ax1.set_title(
                f'Ground Truth Boxes ({len(gt_boxes)} nuclei, crop: {img_w_actual}x{img_h_actual}, orig: {orig_w}x{orig_h})',
                fontsize=14
            )
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         linewidth=2, edgecolor='green', facecolor='none')
                ax1.add_patch(rect)
        else:
            ax1.set_title('Ground Truth Boxes (0 nuclei)', fontsize=14)
        
        ax1.axis('off')
        
        # Ground truth - masks (segmentation labels)
        ax2.imshow(img)
        if 'masks' in target and len(target['masks']) > 0:
            gt_masks = target['masks'].cpu().numpy()
            
            # Resize masks from original size to current image size
            gt_masks_tensor = torch.tensor(gt_masks, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
            gt_masks_resized = F.interpolate(
                gt_masks_tensor,
                size=(img_h_actual, img_w_actual),
                mode='bilinear',
                align_corners=False
            ).squeeze(1).numpy()  # Remove channel dimension and convert to numpy
            
            # Combine all masks with different colors
            combined_gt_mask = np.zeros((img_h_actual, img_w_actual, 3))
            colors = plt.cm.Set3(np.linspace(0, 1, len(gt_masks_resized)))
            
            for i, mask in enumerate(gt_masks_resized):
                combined_gt_mask[mask > 0.5] = colors[i][:3]
            
            ax2.imshow(combined_gt_mask, alpha=0.5)
            ax2.set_title(f'Ground Truth Masks ({len(gt_masks_resized)} masks)', fontsize=14)
        else:
            ax2.set_title('No ground truth masks', fontsize=14)
        ax2.axis('off')
        
        # Predictions - boxes
        ax3.imshow(img)
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        keep = pred_scores > 0.05
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        
        ax3.set_title(f'Predicted Boxes ({len(pred_boxes)} boxes, avg conf: {pred_scores.mean():.3f})', fontsize=14)
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box
            color = plt.cm.RdYlGn(score)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=1, edgecolor=color, facecolor='none', alpha=0.7)
            ax3.add_patch(rect)
        ax3.axis('off')
        
        # Predictions - masks
        ax4.imshow(img)
        if 'masks' in pred and len(pred['masks']) > 0:
            pred_masks = pred['masks'].cpu().numpy()
            pred_masks = pred_masks[keep]  # Apply same filtering as boxes
            
            # Combine all masks with different colors
            # Use HSV color space to generate many distinct colors
            combined_pred_mask = np.zeros((img_h_actual, img_w_actual, 3))
            num_masks = len(pred_masks)
            
            for i, mask in enumerate(pred_masks):
                mask = mask[0] if len(mask.shape) == 3 else mask  # Remove channel dimension if present
                
                # Generate distinct colors using HSV: vary hue, keep saturation and value high
                hue = (i / num_masks) % 1.0  # Cycle through hues
                saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
                value = 0.9 + (i % 2) * 0.1  # Vary brightness slightly
                
                # Convert HSV to RGB
                from matplotlib.colors import hsv_to_rgb
                hsv_color = np.array([[[hue, saturation, value]]])
                rgb_color = hsv_to_rgb(hsv_color)[0, 0]
                
                combined_pred_mask[mask > 0.5] = rgb_color
            
            ax4.imshow(combined_pred_mask, alpha=0.5)
            ax4.set_title(f'Predicted Masks ({len(pred_masks)} masks)', fontsize=14)
        else:
            ax4.set_title('No masks predicted', fontsize=14)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'epoch_{epoch:03d}_sample_{idx}.png', dpi=100, bbox_inches='tight')
        plt.close()


def train_one_epoch(model, train_loader, optimizer, device, epoch, writer, scaler=None):
    """Train for one epoch with mixed precision support"""
    model.train()
    total_loss = 0
    loss_components = {'classifier': 0, 'box_reg': 0, 'objectness': 0, 'rpn_box': 0, 'mask': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += losses.item()
        
        # Accumulate loss components
        for key in loss_components.keys():
            loss_key = f'loss_{key}'
            if loss_key in loss_dict:
                loss_components[key] += loss_dict[loss_key].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'cls': f'{loss_dict.get("loss_classifier", 0):.3f}',
            'box': f'{loss_dict.get("loss_box_reg", 0):.3f}',
            'mask': f'{loss_dict.get("loss_mask", 0):.3f}'
        })
        
        # Log to tensorboard
        if batch_idx % 10 == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/total_loss', losses.item(), step)
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{k}', v.item(), step)
    
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    # Log average loss components
    for key, value in loss_components.items():
        writer.add_scalar(f'train/avg_{key}', value / len(train_loader), epoch)
    
    return avg_loss


@torch.no_grad()
def evaluate(model, val_loader, device, epoch, writer, output_dir=None):
    """Comprehensive evaluation with metrics and visualization"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_images = []
    detection_counts = []
    confidence_scores = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    for images, targets in pbar:
        images_on_device = [img.to(device) for img in images]
        
        # Get predictions
        predictions = model(images_on_device)
        
        # Store for visualization (first 2 batches only)
        if len(all_images) < 8:
            all_images.extend(images)
        
        for pred in predictions:
            num_dets = len(pred['boxes'])
            detection_counts.append(num_dets)
            if num_dets > 0:
                confidence_scores.extend(pred['scores'].cpu().numpy().tolist())
        
        all_predictions.extend(predictions)
        all_targets.extend(targets)
    
    # Visualize predictions vs targets
    if output_dir and len(all_images) > 0:
        visualize_predictions_vs_targets(
            all_images[:4], 
            all_predictions[:4], 
            all_targets[:4], 
            epoch, 
            output_dir,
            num_samples=4
        )
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets, iou_threshold=0.5, score_threshold=0.05)
    
    # Statistics
    avg_detections = np.mean(detection_counts) if detection_counts else 0
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    # Log metrics
    writer.add_scalar('val/avg_detections', avg_detections, epoch)
    writer.add_scalar('val/avg_confidence', avg_confidence, epoch)
    writer.add_scalar('val/precision', metrics['precision'], epoch)
    writer.add_scalar('val/recall', metrics['recall'], epoch)
    writer.add_scalar('val/f1', metrics['f1'], epoch)
    
    print(f"\nValidation Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  TP/FP/FN: {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    print(f"  Avg Detections: {avg_detections:.1f}")
    print(f"  Avg Confidence: {avg_confidence:.3f}")
    
    return metrics['f1'], avg_detections


def main():
    parser = argparse.ArgumentParser(description='Improved Mask R-CNN Training')
    
    # Data paths
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dinov2_checkpoint', type=str, required=True)
    parser.add_argument('--resume_from', type=str, default=None)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_backbone', action='store_true')
    
    # Cropping options
    parser.add_argument('--crop_size', type=int, default=224,
                       help='Crop size (default: 224)')
    parser.add_argument('--max_crops_per_image', type=int, default=4,
                       help='Number of random crops per training image')
    parser.add_argument('--collapse_categories', action='store_true',
                       help='Treat all dataset categories as a single class')
    
    # Advanced options
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--augmentation_multiplier', type=int, default=1,
                       help='Multiply training dataset size with augmentations (e.g., 3 = 3x more batches per epoch)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs_maskrcnn_improved')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mixed precision: {args.use_amp}")
    print(f"Augmentation multiplier: {args.augmentation_multiplier}x")
    
    # Create dataloaders
    print("Loading datasets...")
    from maskrcnn_crop_dataset_loader import create_dataloaders
    
    train_loader, val_loader = create_dataloaders(
        train_images_dir=os.path.join(args.data_root, "train/images"),
        train_labels_dir=os.path.join(args.data_root, "train/labels"),
        val_images_dir=os.path.join(args.data_root, "valid/images"),
        val_labels_dir=os.path.join(args.data_root, "valid/labels"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        max_crops_per_image=args.max_crops_per_image,
        train_annotations_file=os.path.join(args.data_root, "train/train_annotations.json") if os.path.exists(os.path.join(args.data_root, "train/train_annotations.json")) else None,
        val_annotations_file=os.path.join(args.data_root, "valid/val_annotations.json") if os.path.exists(os.path.join(args.data_root, "valid/val_annotations.json")) else None,
        collapse_categories=args.collapse_categories,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Crop size: {args.crop_size}x{args.crop_size}")
    print(f"Crops per image: {args.max_crops_per_image}")
    
    # Create model
    print("\nBuilding model...")
    from dinov2_maskrcnn_improved import create_improved_maskrcnn
    
    model = create_improved_maskrcnn(
        dinov2_checkpoint_path=args.dinov2_checkpoint,
        freeze_backbone=args.freeze_backbone,
        target_size=args.crop_size
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    if not args.freeze_backbone:
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': head_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler() if args.use_amp else None
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        print(f"\nResuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch} to {args.num_epochs}")
    print(f"Output directory: {output_dir}\n")
    
    best_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer, scaler)
        
        # Validate
        f1_score, avg_dets = evaluate(model, val_loader, device, epoch, writer, output_dir=output_dir)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', current_lr, epoch)
        
        # Early stopping and checkpoint logic
        is_best = f1_score > best_f1
        if is_best:
            best_f1 = f1_score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Prepare checkpoint payload once per epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'f1_score': f1_score,
            'best_f1': best_f1,
        }

        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch}")

        # Always update best model when improved
        if is_best:
            best_path = output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (F1: {best_f1:.4f})")

        # Save final checkpoint at the end of training
        if epoch == args.num_epochs - 1:
            final_checkpoint_path = output_dir / 'final_checkpoint.pth'
            torch.save(checkpoint, final_checkpoint_path)
            print("✓ Saved final checkpoint")
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  F1 Score: {f1_score:.4f} {'(BEST)' if is_best else ''}")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Best F1: {best_f1:.4f} (epoch {best_epoch})")
        print(f"  Patience: {epochs_without_improvement}/{args.early_stopping_patience}\n")
        
        # Early stopping
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
            break
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best model saved at: {output_dir / 'best_model.pth'}")
    print(f"Best F1 score: {best_f1:.4f}")


if __name__ == "__main__":
    main()
