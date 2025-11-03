"""
HoverNet dataset loader for Mask R-CNN training
Handles .mat files with instance segmentation masks
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import scipy.io
import json
from torchvision.ops import box_convert


class HoverNetDataset(Dataset):
    """
    Dataset loader for HoverNet format with .mat mask files
    """
    
    def __init__(
        self, 
        images_dir: str,
        labels_dir: str,
        annotations_file: str = None,
        transform=None,
        target_size: int = 518,
        max_objects: int = 200,
        train: bool = True,
        augmentation_multiplier: int = 1
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.target_size = target_size
        self.max_objects = max_objects
        self.train = train
        self.augmentation_multiplier = augmentation_multiplier if train else 1
        
        # YOLO-style augmentations
        self.augmentation = YOLOStyleAugmentation(train=train)
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.png')) + 
                                 list(self.images_dir.glob('*.jpg')) +
                                 list(self.images_dir.glob('*.tif')))
        
        # Load annotations if provided
        self.annotations = {}
        if annotations_file and Path(annotations_file).exists():
            with open(annotations_file, 'r') as f:
                data = json.load(f)
                for img_info in data['images']:
                    self.annotations[img_info['file_name']] = {
                        'width': img_info['width'],
                        'height': img_info['height']
                    }
        
        if train and augmentation_multiplier > 1:
            print(f"Found {len(self.image_files)} images in {images_dir}")
            print(f"With {augmentation_multiplier}x augmentation → {len(self.image_files) * augmentation_multiplier} effective samples")
        else:
            print(f"Found {len(self.image_files)} images in {images_dir}")
        
    def __len__(self):
        return len(self.image_files) * self.augmentation_multiplier
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Map augmented index back to original image index
        original_idx = idx % len(self.image_files)
        
        # Load image
        img_path = self.image_files[original_idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get original dimensions
        orig_w, orig_h = image.size
        
        # Load corresponding mask file - try multiple formats
        mask_path = None
        inst_map = None
        
        # Try .mat format first
        mat_path = self.labels_dir / (img_path.stem + '.mat')
        if mat_path.exists():
            mat_data = scipy.io.loadmat(str(mat_path))
            inst_map = mat_data['inst_map']  # Instance segmentation map
            mask_path = mat_path
        else:
            # Try PNG format with .png_label.png pattern (filename.png_label.png)
            png_path = self.labels_dir / (img_path.stem + '.png_label.png')
            if png_path.exists():
                # Load PNG instance map or binary mask
                png_data = np.array(Image.open(png_path))
                # If binary (only 0 and 255), convert to instance map using connected components
                if len(np.unique(png_data)) <= 2:
                    from scipy import ndimage
                    binary = png_data > 0
                    inst_map, _ = ndimage.label(binary)
                else:
                    inst_map = png_data
                mask_path = png_path
            else:
                # Try PNG format with _label suffix (filename_label.png)
                png_path2 = self.labels_dir / (img_path.stem + '_label.png')
                if png_path2.exists():
                    png_data = np.array(Image.open(png_path2))
                    # If binary, convert to instance map
                    if len(np.unique(png_data)) <= 2:
                        from scipy import ndimage
                        binary = png_data > 0
                        inst_map, _ = ndimage.label(binary)
                    else:
                        inst_map = png_data
                    mask_path = png_path2
                else:
                    # Try PNG format without suffix
                    png_path3 = self.labels_dir / (img_path.stem + '.png')
                    if png_path3.exists():
                        png_data = np.array(Image.open(png_path3))
                        # If binary, convert to instance map
                        if len(np.unique(png_data)) <= 2:
                            from scipy import ndimage
                            binary = png_data > 0
                            inst_map, _ = ndimage.label(binary)
                        else:
                            inst_map = png_data
                        mask_path = png_path3
        
        boxes = []
        labels = []
        masks = []
        
        if inst_map is not None:
            # Extract individual instance masks and bounding boxes
            unique_instances = np.unique(inst_map)
            unique_instances = unique_instances[unique_instances > 0]  # Remove background (0)
            
            for inst_id in unique_instances:
                # Create binary mask for this instance
                mask = (inst_map == inst_id).astype(np.uint8)
                
                # Find bounding box
                rows, cols = np.where(mask)
                if len(rows) == 0:
                    continue
                
                y1, y2 = rows.min(), rows.max()
                x1, x2 = cols.min(), cols.max()
                
                # Convert to normalized coordinates (YOLO format: center_x, center_y, width, height)
                center_x = (x1 + x2) / 2.0 / orig_w
                center_y = (y1 + y2) / 2.0 / orig_h
                width = (x2 - x1) / orig_w
                height = (y2 - y1) / orig_h
                
                # Skip very small objects
                if width < 0.01 or height < 0.01:
                    continue
                
                boxes.append([center_x, center_y, width, height])
                labels.append(1)  # All instances are class 1 (nucleus)
                masks.append(mask)
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, orig_h, orig_w), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.stack(masks), dtype=torch.float32)
        
        # Apply YOLO-style augmentations (handles boxes and masks correctly)
        image, boxes, masks = self.augmentation(image, boxes, masks)
        
        # Apply transforms to image (resize + to tensor)
        if self.transform is not None:
            image = self.transform(image)
        
        # Create target dictionary
        target = {
            'boxes': boxes,  # Normalized coordinates [0, 1]
            'labels': labels,
            'masks': masks,  # Binary masks
            'image_id': torch.tensor([original_idx]),
            'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.float32)
        }
        
        return image, target
    
    def get_image_name(self, idx: int) -> str:
        return self.image_files[idx].name


def collate_fn(batch):
    """Custom collate function - returns lists for Mask R-CNN"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


class YOLOStyleAugmentation:
    """
    YOLO-style augmentations that properly handle bounding boxes and masks
    Applied BEFORE resize to maintain accuracy
    """
    def __init__(self, train: bool = True):
        self.train = train
    
    def __call__(self, image, boxes, masks):
        """
        Args:
            image: PIL Image
            boxes: tensor of [cx, cy, w, h] normalized coordinates
            masks: tensor of [N, H, W] binary masks
        Returns:
            augmented image, boxes, and masks
        """
        if not self.train or len(boxes) == 0:
            return image, boxes, masks
        
        import random
        
        # Horizontal flip
        if random.random() < 0.5:
            image = T.functional.hflip(image)
            boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip center_x
            masks = torch.flip(masks, dims=[2])  # Flip masks horizontally
        
        # Vertical flip
        if random.random() < 0.5:
            image = T.functional.vflip(image)
            boxes[:, 1] = 1.0 - boxes[:, 1]  # Flip center_y
            masks = torch.flip(masks, dims=[1])  # Flip masks vertically
        
        # 90-degree rotations (only multiples of 90 to keep boxes axis-aligned)
        rotation = random.choice([0, 90, 180, 270])
        if rotation == 90:
            image = T.functional.rotate(image, 90, expand=False)
            # Rotate boxes: (cx, cy) -> (1-cy, cx), swap width/height
            new_boxes = boxes.clone()
            new_boxes[:, 0] = 1.0 - boxes[:, 1]  # new cx = 1 - old cy
            new_boxes[:, 1] = boxes[:, 0]         # new cy = old cx
            new_boxes[:, 2] = boxes[:, 3]         # new w = old h
            new_boxes[:, 3] = boxes[:, 2]         # new h = old w
            boxes = new_boxes
            # Rotate masks
            masks = torch.rot90(masks, k=1, dims=[1, 2])
        elif rotation == 180:
            image = T.functional.rotate(image, 180, expand=False)
            boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip center_x
            boxes[:, 1] = 1.0 - boxes[:, 1]  # Flip center_y
            # Rotate masks
            masks = torch.rot90(masks, k=2, dims=[1, 2])
        elif rotation == 270:
            image = T.functional.rotate(image, 270, expand=False)
            # Rotate boxes: (cx, cy) -> (cy, 1-cx), swap width/height
            new_boxes = boxes.clone()
            new_boxes[:, 0] = boxes[:, 1]         # new cx = old cy
            new_boxes[:, 1] = 1.0 - boxes[:, 0]   # new cy = 1 - old cx
            new_boxes[:, 2] = boxes[:, 3]         # new w = old h
            new_boxes[:, 3] = boxes[:, 2]         # new h = old w
            boxes = new_boxes
            # Rotate masks
            masks = torch.rot90(masks, k=3, dims=[1, 2])
        
        # Color jitter (doesn't affect boxes or masks)
        if random.random() < 0.8:
            image = T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )(image)
        
        # Gaussian blur (doesn't affect boxes or masks)
        if random.random() < 0.1:
            image = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(image)
        
        return image, boxes, masks


def get_transform(train: bool = True, target_size: int = 518):
    """
    Get image transforms
    NOTE: Mask R-CNN does its own normalization, so we don't normalize here
    """
    transforms = []
    
    # Only resize and convert to tensor
    transforms.append(T.Resize((target_size, target_size)))
    transforms.append(T.ToTensor())
    
    return T.Compose(transforms)


def create_dataloaders(
    train_images_dir: str,
    train_labels_dir: str,
    val_images_dir: str,
    val_labels_dir: str,
    train_annotations_file: str = None,
    val_annotations_file: str = None,
    batch_size: int = 4,
    num_workers: int = 4,
    target_size: int = 518,
    augmentation_multiplier: int = 1
):
    """Create train and validation dataloaders"""
    
    train_dataset = HoverNetDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        annotations_file=train_annotations_file,
        transform=get_transform(train=True, target_size=target_size),
        target_size=target_size,
        train=True,
        augmentation_multiplier=augmentation_multiplier
    )
    
    val_dataset = HoverNetDataset(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        annotations_file=val_annotations_file,
        transform=get_transform(train=False, target_size=target_size),
        target_size=target_size,
        train=False,
        augmentation_multiplier=1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    base_path = "/rodata/dlmp_path/han/data/hovernet_dataset/hoverdata"
    
    train_images = os.path.join(base_path, "train/Images")
    train_labels = os.path.join(base_path, "train/Labels")
    train_annotations = os.path.join(base_path, "train/train_annotations.json")
    
    val_images = os.path.join(base_path, "val/Images")
    val_labels = os.path.join(base_path, "val/Labels")
    val_annotations = os.path.join(base_path, "val/val_annotations.json")
    
    train_loader, val_loader = create_dataloaders(
        train_images_dir=train_images,
        train_labels_dir=train_labels,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        train_annotations_file=train_annotations,
        val_annotations_file=val_annotations,
        batch_size=2,
        num_workers=2,
        target_size=518
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading a batch
    images, targets = next(iter(train_loader))
    print(f"\nBatch info:")
    print(f"Number of samples: {len(images)}")
    
    for i, target in enumerate(targets):
        print(f"\nSample {i}:")
        print(f"  Image shape: {images[i].shape}")
        print(f"  Original size (H, W): {target['orig_size']}")
        print(f"  Number of nuclei: {len(target['boxes'])}")
        print(f"  Masks shape: {target['masks'].shape}")
        if len(target['boxes']) > 0:
            print(f"  First box (normalized cx, cy, w, h): {target['boxes'][0]}")
            print(f"  First mask shape: {target['masks'][0].shape}")
            print(f"  First mask unique values: {torch.unique(target['masks'][0])}")
