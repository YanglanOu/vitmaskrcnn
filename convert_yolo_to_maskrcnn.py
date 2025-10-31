"""
Convert YOLO format dataset to Mask R-CNN format
- Reads YOLO .txt files (class_id, center_x, center_y, width, height)
- Creates instance segmentation maps (.mat files) from bounding boxes
- Generates annotation JSON files for Mask R-CNN loader
"""
import os
import json
import numpy as np
import scipy.io
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse


def yolo_to_instance_map(label_file, image_width, image_height):
    """
    Convert YOLO format bounding boxes to instance segmentation map
    
    Args:
        label_file: Path to YOLO .txt file
        image_width: Original image width
        image_height: Original image height
    
    Returns:
        inst_map: numpy array of shape (H, W) with instance IDs
    """
    inst_map = np.zeros((image_height, image_width), dtype=np.uint32)
    
    if not os.path.exists(label_file):
        return inst_map
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    instance_id = 1
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        center_x = float(parts[1])
        center_y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Convert normalized YOLO coordinates to absolute pixel coordinates
        # YOLO: center_x, center_y, width, height (all normalized 0-1)
        # Convert to absolute: x1, y1, x2, y2
        abs_center_x = center_x * image_width
        abs_center_y = center_y * image_height
        abs_width = width * image_width
        abs_height = height * image_height
        
        x1 = int(abs_center_x - abs_width / 2)
        y1 = int(abs_center_y - abs_height / 2)
        x2 = int(abs_center_x + abs_width / 2)
        y2 = int(abs_center_y + abs_height / 2)
        
        # Clamp to image boundaries
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))
        
        # Create rectangular mask for this instance
        if x2 > x1 and y2 > y1:
            inst_map[y1:y2, x1:x2] = instance_id
            instance_id += 1
    
    return inst_map


def convert_split(images_dir, labels_dir, output_dir, split_name):
    """
    Convert a single split (train/valid/test) from YOLO to Mask R-CNN format
    
    Args:
        images_dir: Directory containing YOLO images
        labels_dir: Directory containing YOLO .txt label files
        output_dir: Output directory for converted dataset
        split_name: Name of the split (train/valid/test)
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_images_dir = output_dir / split_name / "images"
    output_labels_dir = output_dir / split_name / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    image_files = sorted(set(image_files))
    
    print(f"\nConverting {split_name} split: {len(image_files)} images")
    
    annotations = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "nucleus", "supercategory": "cell"}]
    }
    
    annotation_id = 1
    
    for img_path in tqdm(image_files, desc=f"Converting {split_name}"):
        # Load image to get dimensions
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue
        
        # Copy image to output directory
        output_img_path = output_images_dir / img_path.name
        if not output_img_path.exists():
            import shutil
            shutil.copy2(img_path, output_img_path)
        
        # Convert YOLO label to instance map
        label_file = labels_dir / (img_path.stem + '.txt')
        inst_map = yolo_to_instance_map(label_file, img_width, img_height)
        
        # Save instance map as .mat file
        mat_file_path = output_labels_dir / (img_path.stem + '.mat')
        scipy.io.savemat(str(mat_file_path), {'inst_map': inst_map})
        
        # Add to annotations JSON
        annotations["images"].append({
            "id": len(annotations["images"]) + 1,
            "file_name": img_path.name,
            "width": img_width,
            "height": img_height
        })
        
        # Count instances
        unique_instances = np.unique(inst_map)
        unique_instances = unique_instances[unique_instances > 0]  # Remove background
        
        for inst_id in unique_instances:
            mask = (inst_map == inst_id)
            rows, cols = np.where(mask)
            
            if len(rows) > 0:
                y1, y2 = rows.min(), rows.max()
                x1, x2 = cols.min(), cols.max()
                
                # YOLO format for storage (normalized)
                center_x = ((x1 + x2) / 2.0) / img_width
                center_y = ((y1 + y2) / 2.0) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": len(annotations["images"]),
                    "category_id": 1,
                    "bbox": [center_x, center_y, width, height],  # YOLO format (normalized)
                    "area": int(np.sum(mask))
                })
                annotation_id += 1
    
    # Save annotations JSON
    annotations_file = output_dir / split_name / f"{split_name}_annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Converted {split_name}: {len(image_files)} images, {len(annotations['annotations'])} annotations")
    print(f"  Saved to: {output_dir / split_name}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO dataset to Mask R-CNN format')
    parser.add_argument('--yolo_dataset', type=str, default='./yolo_dataset',
                       help='Path to YOLO dataset directory')
    parser.add_argument('--output_dir', type=str, default='./maskrcnn_dataset',
                       help='Output directory for Mask R-CNN format dataset')
    
    args = parser.parse_args()
    
    yolo_dataset = Path(args.yolo_dataset)
    output_dir = Path(args.output_dir)
    
    if not yolo_dataset.exists():
        raise ValueError(f"YOLO dataset directory not found: {yolo_dataset}")
    
    print(f"Converting YOLO dataset from: {yolo_dataset}")
    print(f"Output directory: {output_dir}")
    
    # Convert train split
    train_images = yolo_dataset / "train" / "images"
    train_labels = yolo_dataset / "train" / "labels"
    if train_images.exists() and train_labels.exists():
        convert_split(train_images, train_labels, output_dir, "train")
    else:
        print(f"Warning: Train split not found at {train_images} or {train_labels}")
    
    # Convert valid/val split
    valid_images = yolo_dataset / "valid" / "images"
    valid_labels = yolo_dataset / "valid" / "labels"
    if valid_images.exists() and valid_labels.exists():
        convert_split(valid_images, valid_labels, output_dir, "valid")
    else:
        print(f"Warning: Valid split not found at {valid_images} or {valid_labels}")
    
    # Convert test split (if exists)
    test_images = yolo_dataset / "test" / "images"
    test_labels = yolo_dataset / "test" / "labels"
    if test_images.exists() and test_labels.exists():
        convert_split(test_images, test_labels, output_dir, "test")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Dataset structure:")
    print(f"    {output_dir}/")
    print(f"      train/")
    print(f"        images/")
    print(f"        labels/ (.*.mat)")
    print(f"        train_annotations.json")
    print(f"      valid/")
    print(f"        images/")
    print(f"        labels/ (.*.mat)")
    print(f"        valid_annotations.json")


if __name__ == "__main__":
    main()

