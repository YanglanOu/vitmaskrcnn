"""
Improved DINOv2 + Mask R-CNN with proper coordinate conversion
Optimized for cropped images (e.g., 224x224) instead of resizing
Fixed: Handles variable-sized images correctly
Added: Mask prediction head for instance segmentation
"""
import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F
from collections import OrderedDict
import warnings


class DINOv2BackboneSimple(nn.Module):
    """DINOv2 backbone for Mask R-CNN - single scale"""
    
    def __init__(self, dinov2_model, freeze=True):
        super().__init__()
        self.backbone = dinov2_model
        self.freeze = freeze
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # DINOv2 ViT-G output dimension
        embed_dim = 1536
        
        # Single projection layer
        self.proj = nn.Conv2d(embed_dim, 256, 1)
        
        # Required attribute for Mask R-CNN
        self.out_channels = 256
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] image tensor
        Returns:
            features: OrderedDict with single feature map
        """
        B, C, H, W = x.shape
        
        # Ensure dimensions are divisible by 14
        target_h = ((H + 6) // 14) * 14
        target_w = ((W + 6) // 14) * 14
        
        if H != target_h or W != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        with torch.set_grad_enabled(not self.freeze):
            features = self.backbone.forward_features(x)
            patch_tokens = features['x_norm_patchtokens']
            
            # Reshape to spatial grid
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
            
            # Upsample to 64x64 for better resolution
            patch_tokens = F.interpolate(
                patch_tokens,
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            )
        
        # Project to 256 channels
        features_out = self.proj(patch_tokens)
        
        # Mask R-CNN expects OrderedDict
        return OrderedDict([('0', features_out)])


class ImprovedDINOv2MaskRCNN(nn.Module):
    """
    Mask R-CNN with DINOv2 backbone
    Optimized for cropped images (e.g., 224x224) instead of resizing
    Properly handles variable-sized images with correct coordinate conversion
    Includes mask prediction head for instance segmentation
    """
    def __init__(self, dinov2_model, num_classes=2, freeze_backbone=True, target_size=224):
        super().__init__()
        
        self.target_size = target_size
        
        # Create simple backbone
        backbone = DINOv2BackboneSimple(dinov2_model, freeze=freeze_backbone)
        
        # Anchor generator with custom sizes for nuclei
        # Adjust anchor sizes based on target_size if needed
        anchor_generator = AnchorGenerator(
            sizes=((4, 6, 8, 10, 12, 16, 20, 24, 32),),
            aspect_ratios=((0.5, 0.75, 1.0, 1.25, 1.5, 2.0),)
        )
        
        # RoI pooling
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Mask RoI pooling
        mask_roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=14,
            sampling_ratio=2
        )
        
        # Create Mask R-CNN with default transform
        # Images are already cropped to target_size in dataset loader
        # So we just set min_size/max_size to match (images won't be resized)
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_roi_pooler,
            min_size=target_size,
            max_size=target_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            # RPN parameters
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=1000,
            rpn_post_nms_top_n_test=500,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.5,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            # Box parameters
            box_score_thresh=0.01,
            box_nms_thresh=0.4,
            box_detections_per_img=500,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=256,
            box_positive_fraction=0.25,
        )
    
    def forward(self, images, targets=None):
        """
        Forward pass with proper coordinate conversion
        Images are already cropped to target_size in dataset loader
        Boxes are normalized [cx, cy, w, h] relative to the cropped image
        """
        # Convert to list format
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                image_list = [img for img in images]
            else:
                image_list = [images]
        else:
            image_list = images
        
        # Training mode
        if self.training and targets is not None:
            rcnn_targets = []
            valid_indices = []
            
            for i, target in enumerate(targets):
                # Images are already cropped to target_size in dataset loader
                # Boxes are normalized [cx, cy, w, h] relative to the cropped image
                img_h, img_w = image_list[i].shape[-2:]  # Should be target_size x target_size
                
                boxes = target['boxes']  # Normalized YOLO format [cx, cy, w, h] relative to cropped image
                
                if len(boxes) == 0:
                    continue
                
                # Convert normalized YOLO boxes to absolute XYXY coordinates
                # Boxes are already normalized to the cropped image, so use current image size directly
                boxes_xyxy = torch.zeros_like(boxes)
                boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w  # x1 = (cx - w/2) * width
                boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h  # y1 = (cy - h/2) * height
                boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w  # x2 = (cx + w/2) * width
                boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h  # y2 = (cy + h/2) * height
                
                # Clamp to image boundaries
                boxes_xyxy[:, 0].clamp_(min=0, max=img_w)
                boxes_xyxy[:, 1].clamp_(min=0, max=img_h)
                boxes_xyxy[:, 2].clamp_(min=0, max=img_w)
                boxes_xyxy[:, 3].clamp_(min=0, max=img_h)
                
                # Filter invalid boxes
                valid_boxes = (
                    (boxes_xyxy[:, 2] > boxes_xyxy[:, 0] + 1.0) &
                    (boxes_xyxy[:, 3] > boxes_xyxy[:, 1] + 1.0)
                )
                
                if valid_boxes.sum() == 0:
                    continue
                
                boxes_xyxy = boxes_xyxy[valid_boxes]
                labels = torch.ones(len(boxes_xyxy), dtype=torch.int64, device=boxes.device)
                
                # Use real masks if available, otherwise create dummy masks
                if 'masks' in target and len(target['masks']) > 0:
                    # Masks are already cropped to img_h x img_w in dataset loader
                    real_masks = target['masks'][valid_boxes]  # Filter masks by valid boxes
                    # Masks should already be the right size (target_size x target_size)
                    masks = real_masks
                else:
                    # Fallback to dummy masks
                    masks = self._create_dummy_masks(boxes_xyxy, img_h, img_w)
                
                rcnn_targets.append({
                    'boxes': boxes_xyxy,
                    'labels': labels,
                    'masks': masks
                })
                valid_indices.append(i)
            
            if len(rcnn_targets) == 0:
                # Return zero losses if no valid targets
                # Create losses connected to model parameters to allow backpropagation
                device = image_list[0].device
                # Get a trainable parameter from the model to connect the loss to the graph
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                if len(trainable_params) > 0:
                    dummy_param = trainable_params[0]
                    # Create separate zero losses for each component (all connected to the graph)
                    zero_loss = (dummy_param.sum() * 0.0)
                    zero_loss_classifier = zero_loss.clone()
                    zero_loss_box_reg = zero_loss.clone()
                    zero_loss_objectness = zero_loss.clone()
                    zero_loss_rpn_box_reg = zero_loss.clone()
                    zero_loss_mask = zero_loss.clone()
                else:
                    # Fallback: create zero tensors on device (shouldn't happen if model has trainable params)
                    zero_loss_classifier = torch.tensor(0.0, device=device, requires_grad=False)
                    zero_loss_box_reg = torch.tensor(0.0, device=device, requires_grad=False)
                    zero_loss_objectness = torch.tensor(0.0, device=device, requires_grad=False)
                    zero_loss_rpn_box_reg = torch.tensor(0.0, device=device, requires_grad=False)
                    zero_loss_mask = torch.tensor(0.0, device=device, requires_grad=False)
                warnings.warn(f"No valid targets found in batch of {len(images)} images. Returning zero losses.")
                return {
                    'loss_classifier': zero_loss_classifier,
                    'loss_box_reg': zero_loss_box_reg,
                    'loss_objectness': zero_loss_objectness,
                    'loss_rpn_box_reg': zero_loss_rpn_box_reg,
                    'loss_mask': zero_loss_mask
                }
            
            # Use only images with valid targets
            valid_images = [image_list[i] for i in valid_indices]
            return self.model(valid_images, rcnn_targets)
        else:
            # Inference mode
            return self.model(image_list)
    
    def _create_dummy_masks(self, boxes, img_h, img_w):
        """
        Create dummy circular masks for training
        In practice, you would load actual mask annotations
        """
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = box.int()
            mask = torch.zeros((img_h, img_w), dtype=torch.uint8, device=box.device)
            
            # Create circular mask
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = min((x2 - x1), (y2 - y1)) // 2
            
            # Simple circular mask
            y, x = torch.meshgrid(torch.arange(img_h, device=box.device), 
                                 torch.arange(img_w, device=box.device), indexing='ij')
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2).float()
            masks.append(mask)
        
        return torch.stack(masks) if masks else torch.zeros((0, img_h, img_w), dtype=torch.float32, device=boxes.device)


def create_improved_maskrcnn(dinov2_checkpoint_path, freeze_backbone=True, target_size=224):
    """Create improved Mask R-CNN model with DINOv2 backbone"""
    from dinov2_detr import load_dinov2_from_checkpoint
    
    dinov2 = load_dinov2_from_checkpoint(dinov2_checkpoint_path)
    
    model = ImprovedDINOv2MaskRCNN(
        dinov2_model=dinov2,
        num_classes=2,
        freeze_backbone=freeze_backbone,
        target_size=target_size
    )
    
    return model


if __name__ == "__main__":
    dinov2_checkpoint = "/rodata/dlmp_path/han/data/m328672/dinov2-main/dgx_vitg14_patch37M/training_287499/teacher_checkpoint.pth"
    
    model = create_improved_maskrcnn(
        dinov2_checkpoint_path=dinov2_checkpoint,
        freeze_backbone=True
    )
    
    print("Model created successfully!")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    dummy_input = [torch.randn(3, 224, 224).to(device)]
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nTest inference:")
    print(f"Detections: {len(output[0]['boxes'])}")
    if len(output[0]['boxes']) > 0:
        print(f"Score range: [{output[0]['scores'].min():.3f}, {output[0]['scores'].max():.3f}]")
        print(f"Masks shape: {output[0]['masks'].shape}")
