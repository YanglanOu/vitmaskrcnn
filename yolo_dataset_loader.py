import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class YOLOv5toDETRDataset(Dataset):
    """
    Dataset loader for YOLOv5 format to object detection training
    CRITICAL: Preserves original image dimensions for proper coordinate conversion
    """
    
    def __init__(
        self, 
        images_dir: str,
        labels_dir: str,
        transform=None,
        target_size: int = 518,
        max_objects: int = 200,
        train: bool = True,
        augmentation_multiplier: int = 1  # NEW: multiply dataset size with augmentations
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
        self.image_files = sorted(list(self.images_dir.glob('*.tif')) + 
                                 list(self.images_dir.glob('*.tiff')) +
                                 list(self.images_dir.glob('*.png')) +
                                 list(self.images_dir.glob('*.jpg')))
        
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
        
        # CRITICAL: Store original dimensions BEFORE any transforms
        orig_w, orig_h = image.size
        
        # Load corresponding label file
        label_path = self.labels_dir / (img_path.stem + '.txt')
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # YOLOv5 format: normalized [0, 1] (cx, cy, w, h)
                        boxes.append([cx, cy, w, h])
                        labels.append(class_id)
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        # Apply YOLO-style augmentations (handles boxes correctly)
        image, boxes = self.augmentation(image, boxes)
        
        # Apply transforms to image (resize + to tensor)
        if self.transform is not None:
            image = self.transform(image)
        
        # Create target dictionary with ORIGINAL dimensions
        target = {
            'boxes': boxes,  # Normalized coordinates [0, 1] (potentially augmented)
            'labels': labels,
            'image_id': torch.tensor([original_idx]),  # Use original index
            'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.float32)  # CRITICAL!
        }
        
        return image, target
    
    def get_image_name(self, idx: int) -> str:
        return self.image_files[idx].name


def collate_fn(batch):
    """Custom collate function - returns lists for Faster R-CNN"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


class YOLOStyleAugmentation:
    """
    YOLO-style augmentations that properly handle bounding boxes
    Applied BEFORE resize to maintain box accuracy
    """
    def __init__(self, train: bool = True):
        self.train = train
    
    def __call__(self, image, boxes):
        """
        Args:
            image: PIL Image
            boxes: tensor of [cx, cy, w, h] normalized coordinates
        Returns:
            augmented image and boxes
        """
        if not self.train or len(boxes) == 0:
            return image, boxes
        
        import random
        
        # Horizontal flip
        if random.random() < 0.5:
            image = T.functional.hflip(image)
            boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip center_x
        
        # Vertical flip
        if random.random() < 0.5:
            image = T.functional.vflip(image)
            boxes[:, 1] = 1.0 - boxes[:, 1]  # Flip center_y
        
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
        elif rotation == 180:
            image = T.functional.rotate(image, 180, expand=False)
            boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip center_x
            boxes[:, 1] = 1.0 - boxes[:, 1]  # Flip center_y
        elif rotation == 270:
            image = T.functional.rotate(image, 270, expand=False)
            # Rotate boxes: (cx, cy) -> (cy, 1-cx), swap width/height
            new_boxes = boxes.clone()
            new_boxes[:, 0] = boxes[:, 1]         # new cx = old cy
            new_boxes[:, 1] = 1.0 - boxes[:, 0]   # new cy = 1 - old cx
            new_boxes[:, 2] = boxes[:, 3]         # new w = old h
            new_boxes[:, 3] = boxes[:, 2]         # new h = old w
            boxes = new_boxes
        
        # Color jitter (doesn't affect boxes)
        if random.random() < 0.8:
            image = T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )(image)
        
        # Gaussian blur (doesn't affect boxes)
        if random.random() < 0.1:
            image = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(image)
        
        return image, boxes


def get_transform(train: bool = True, target_size: int = 896):
    """
    Get image transforms
    NOTE: Faster R-CNN does its own normalization, so we don't normalize here
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
    batch_size: int = 4,
    num_workers: int = 4,
    target_size: int = 518,
    augmentation_multiplier: int = 1  # NEW: multiply training data
):
    """Create train and validation dataloaders"""
    
    train_dataset = YOLOv5toDETRDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        transform=get_transform(train=True, target_size=target_size),
        target_size=target_size,
        train=True,
        augmentation_multiplier=augmentation_multiplier  # Apply multiplier
    )
    
    val_dataset = YOLOv5toDETRDataset(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        transform=get_transform(train=False, target_size=target_size),
        target_size=target_size,
        train=False,
        augmentation_multiplier=1  # Never multiply validation data
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
    base_path = "/dgx1data/skunkworks/pathology/bloodbytes/m080982a/DETR/YOLOv5 dataset"
    
    train_images = os.path.join(base_path, "train/images")
    train_labels = os.path.join(base_path, "train/labels")
    val_images = os.path.join(base_path, "valid/images")
    val_labels = os.path.join(base_path, "valid/labels")
    
    train_loader, val_loader = create_dataloaders(
        train_images_dir=train_images,
        train_labels_dir=train_labels,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        batch_size=4,
        num_workers=4,
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
        if len(target['boxes']) > 0:
            print(f"  First box (normalized cx, cy, w, h): {target['boxes'][0]}")