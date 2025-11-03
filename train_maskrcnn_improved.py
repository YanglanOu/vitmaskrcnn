"""
Improved training script for DINOv2 + Mask R-CNN
Fixed: Proper coordinate conversion from normalized YOLO to absolute Mask R-CNN format
Added: Augmentation multiplier support and mask loss tracking
"""
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime
import numpy as np
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
    Compute precision, recall, F1 for evaluation
    Fixed: Properly converts normalized YOLO boxes using original image dimensions
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        # Get original dimensions
        if 'orig_size' in target:
            orig_h, orig_w = target['orig_size'].cpu().numpy()
        else:
            orig_h, orig_w = 518, 518
        
        # Convert normalized YOLO boxes to absolute coordinates at 518x518
        target_boxes_norm = target['boxes'].cpu().numpy()
        
        if len(target_boxes_norm) > 0:
            # Step 1: Convert normalized to absolute in original image space
            target_boxes_abs = np.zeros_like(target_boxes_norm)
            target_boxes_abs[:, 0] = (target_boxes_norm[:, 0] - target_boxes_norm[:, 2] / 2) * orig_w  # x1
            target_boxes_abs[:, 1] = (target_boxes_norm[:, 1] - target_boxes_norm[:, 3] / 2) * orig_h  # y1
            target_boxes_abs[:, 2] = (target_boxes_norm[:, 0] + target_boxes_norm[:, 2] / 2) * orig_w  # x2
            target_boxes_abs[:, 3] = (target_boxes_norm[:, 1] + target_boxes_norm[:, 3] / 2) * orig_h  # y2
            
            # Step 2: Scale to 518x518
            scale_x = 518 / orig_w
            scale_y = 518 / orig_h
            target_boxes = target_boxes_abs.copy()
            target_boxes[:, [0, 2]] *= scale_x
            target_boxes[:, [1, 3]] *= scale_y
        else:
            target_boxes = np.zeros((0, 4))
        
        # Filter predictions by score
        keep = pred_scores > score_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        
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
        
        # Match predictions to targets
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
            else:
                false_positives += 1
        
        false_negatives += len(target_boxes) - len(matched_targets)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }


def visualize_predictions_vs_targets(images, predictions, targets, epoch, output_dir, num_samples=2):
    """Visualize predictions vs ground truth for debugging - includes masks"""
    output_dir = Path(output_dir)
    vis_dir = output_dir / 'debug_visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    for idx in range(min(num_samples, len(images))):
        img = images[idx].cpu().numpy()
        
        # Denormalize if needed (Mask R-CNN doesn't normalize input)
        if img.max() <= 1.0:
            img = np.transpose(img, (1, 2, 0))
        else:
            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1)
        
        pred = predictions[idx]
        target = targets[idx]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        
        # Get original dimensions
        if 'orig_size' in target:
            orig_h, orig_w = target['orig_size'].cpu().numpy()
        else:
            orig_h, orig_w = 518, 518
        
        # Ground truth - convert normalized boxes to 518x518 space
        ax1.imshow(img)
        target_boxes_norm = target['boxes'].cpu().numpy()
        
        if len(target_boxes_norm) > 0:
            # Convert to absolute at original size then scale to 518x518
            gt_boxes = np.zeros_like(target_boxes_norm)
            gt_boxes[:, 0] = (target_boxes_norm[:, 0] - target_boxes_norm[:, 2] / 2) * orig_w
            gt_boxes[:, 1] = (target_boxes_norm[:, 1] - target_boxes_norm[:, 3] / 2) * orig_h
            gt_boxes[:, 2] = (target_boxes_norm[:, 0] + target_boxes_norm[:, 2] / 2) * orig_w
            gt_boxes[:, 3] = (target_boxes_norm[:, 1] + target_boxes_norm[:, 3] / 2) * orig_h
            
            scale_x = 518 / orig_w
            scale_y = 518 / orig_h
            gt_boxes[:, [0, 2]] *= scale_x
            gt_boxes[:, [1, 3]] *= scale_y
            
            ax1.set_title(f'Ground Truth ({len(gt_boxes)} nuclei, orig: {orig_w}x{orig_h})', fontsize=14)
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         linewidth=2, edgecolor='green', facecolor='none')
                ax1.add_patch(rect)
        else:
            ax1.set_title('Ground Truth (0 nuclei)', fontsize=14)
        
        ax1.axis('off')
        
        # Predictions - boxes
        ax2.imshow(img)
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        keep = pred_scores > 0.05
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        
        ax2.set_title(f'Predictions ({len(pred_boxes)} boxes, avg conf: {pred_scores.mean():.3f})', fontsize=14)
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box
            color = plt.cm.RdYlGn(score)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=1, edgecolor=color, facecolor='none', alpha=0.7)
            ax2.add_patch(rect)
        ax2.axis('off')
        
        # Predictions - masks
        ax3.imshow(img)
        if 'masks' in pred and len(pred['masks']) > 0:
            pred_masks = pred['masks'].cpu().numpy()
            pred_masks = pred_masks[keep]  # Apply same filtering as boxes
            
            # Combine all masks with different colors
            combined_mask = np.zeros((img.shape[0], img.shape[1], 3))
            colors = plt.cm.Set3(np.linspace(0, 1, len(pred_masks)))
            
            for i, mask in enumerate(pred_masks):
                mask = mask[0]  # Remove channel dimension
                combined_mask[mask > 0.5] = colors[i][:3]
            
            ax3.imshow(combined_mask, alpha=0.5)
            ax3.set_title(f'Masks ({len(pred_masks)} masks)', fontsize=14)
        else:
            ax3.set_title('No masks predicted', fontsize=14)
        ax3.axis('off')
        
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
    from hover_dataset_loader import HoverNetDataset, get_transform
    
    train_dataset = HoverNetDataset(
        images_dir=os.path.join(args.data_root, "train/images"),
        labels_dir=os.path.join(args.data_root, "train/labels"),
        annotations_file=os.path.join(args.data_root, "train/train_annotations.json"),
        transform=get_transform(train=True, target_size=518),
        train=True,
        augmentation_multiplier=args.augmentation_multiplier
    )
    
    val_dataset = HoverNetDataset(
        images_dir=os.path.join(args.data_root, "valid/images"),
        labels_dir=os.path.join(args.data_root, "valid/labels"),
        annotations_file=os.path.join(args.data_root, "valid/val_annotations.json"),
        transform=get_transform(train=False, target_size=518),
        train=False,
        augmentation_multiplier=1
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_maskrcnn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_maskrcnn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    print(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
    print(f"Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
    
    # Create model
    print("\nBuilding model...")
    from dinov2_maskrcnn_improved import create_improved_maskrcnn
    
    model = create_improved_maskrcnn(
        dinov2_checkpoint_path=args.dinov2_checkpoint,
        freeze_backbone=args.freeze_backbone
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
