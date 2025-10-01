"""
Mask R-CNN inference script for testing the complete pipeline
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from PIL import Image
import torchvision.transforms as T


def load_model(checkpoint_path, dinov2_checkpoint_path, device='cuda'):
    """Load trained Mask R-CNN model"""
    from dinov2_maskrcnn_improved import create_improved_maskrcnn
    
    # Create model
    model = create_improved_maskrcnn(
        dinov2_checkpoint_path=dinov2_checkpoint_path,
        freeze_backbone=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, target_size=518):
    """Preprocess image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size  # (width, height)
    
    # Transform
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor()
    ])
    
    image_tensor = transform(image)
    
    return image_tensor, orig_size


def postprocess_predictions(predictions, orig_size, score_threshold=0.5):
    """Postprocess predictions and scale back to original image size"""
    pred = predictions[0]  # Single image
    
    # Filter by score
    keep = pred['scores'] > score_threshold
    boxes = pred['boxes'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    masks = pred['masks'][keep].cpu().numpy()
    
    # Scale boxes back to original image size
    orig_w, orig_h = orig_size
    scale_x = orig_w / 518
    scale_y = orig_h / 518
    
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    
    # Scale masks back to original image size
    if len(masks) > 0:
        masks = F.interpolate(
            torch.tensor(masks),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).numpy()
    
    return boxes, scores, masks


def visualize_predictions(image_path, boxes, scores, masks, output_path=None):
    """Visualize predictions on image"""
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Original image
    ax1.imshow(image_array)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Image with boxes
    ax2.imshow(image_array)
    ax2.set_title(f'Detections ({len(boxes)} objects)')
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        color = plt.cm.RdYlGn(score)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1-5, f'{score:.2f}', color=color, fontsize=8, weight='bold')
    
    ax2.axis('off')
    
    # Image with masks
    ax3.imshow(image_array)
    ax3.set_title(f'Masks ({len(masks)} masks)')
    
    if len(masks) > 0:
        # Combine all masks with different colors
        combined_mask = np.zeros((image_array.shape[0], image_array.shape[1], 3))
        colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
        
        for i, mask in enumerate(masks):
            mask = mask[0]  # Remove channel dimension
            combined_mask[mask > 0.5] = colors[i][:3]
        
        ax3.imshow(combined_mask, alpha=0.5)
    
    ax3.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN Inference')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dinov2_checkpoint', type=str, required=True, help='Path to DINOv2 checkpoint')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Score threshold for detections')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint_path, args.dinov2_checkpoint, args.device)
    print("Model loaded successfully!")
    
    # Preprocess image
    print("Preprocessing image...")
    image_tensor, orig_size = preprocess_image(args.image_path)
    image_tensor = image_tensor.unsqueeze(0).to(args.device)  # Add batch dimension
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model([image_tensor])
    
    # Postprocess predictions
    print("Postprocessing predictions...")
    boxes, scores, masks = postprocess_predictions(predictions, orig_size, args.score_threshold)
    
    # Print results
    print(f"\nDetection Results:")
    print(f"  Found {len(boxes)} objects")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Average score: {scores.mean():.3f}")
    
    if len(boxes) > 0:
        print(f"\nTop detections:")
        for i, (box, score) in enumerate(zip(boxes[:5], scores[:5])):
            x1, y1, x2, y2 = box
            print(f"  {i+1}. Score: {score:.3f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Visualize results
    print("Creating visualization...")
    visualize_predictions(args.image_path, boxes, scores, masks, args.output_path)
    
    print("Inference completed!")


if __name__ == "__main__":
    main()
