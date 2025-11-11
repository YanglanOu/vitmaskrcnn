"""
Mask R-CNN dataset loader with cropping instead of resizing
Supports both YOLO format (.txt) and HoverNet format (.mat/.png masks)
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import torchvision.transforms as T

# Enable loading truncated images (some images may be corrupted)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import random
import scipy.io
from collections import defaultdict
import json
import mmap
import struct

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover
    mask_utils = None


class MaskRCNNCropDataset(Dataset):
    """
    Dataset loader that crops images to target_size instead of resizing
    Uses random crops during training, center crop during validation
    Supports both YOLO format (boxes only) and HoverNet format (masks)
    """
    
    def __init__(
        self, 
        images_dir: str,
        labels_dir: str,
        crop_size: int = 224,
        train: bool = True,
        max_crops_per_image: int = 4,  # For training: generate multiple crops per image
        min_nuclei_per_crop: int = 1,  # Minimum nuclei required to keep a crop
        annotations_file: str = None,  # Optional: for getting original image dimensions
        collapse_categories: bool = False,
        lazy_load_annotations: bool = None,  # Auto-detect based on file size (None = auto, True/False = override)
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.crop_size = crop_size
        self.train = train
        self.max_crops_per_image = max_crops_per_image if train else 1
        self.min_nuclei_per_crop = min_nuclei_per_crop
        self.collapse_categories = collapse_categories
        self.category_id_to_label: Dict[int, int] = {}
        self.coco_image_by_name: Dict[str, Dict] = {}
        self.coco_annotations_by_image: Dict[int, List[Dict]] = defaultdict(list)
        
        # For lazy loading: store file handle and build index
        self.annotations_file_path = Path(annotations_file) if annotations_file else None
        self.annotations_file_handle = None
        self.annotations_index: Dict[int, Tuple[int, int]] = {}  # image_id -> (start_pos, end_pos) in file
        self.image_id_by_name: Dict[str, int] = {}  # file_name -> image_id
        self.images_meta: Dict[int, Dict] = {}  # image_id -> image metadata
        self.category_ids: set = set()
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.tif')) + 
                                 list(self.images_dir.glob('*.tiff')) +
                                 list(self.images_dir.glob('*.png')) +
                                 list(self.images_dir.glob('*.jpg')))
        
        # Load annotations if provided
        self.annotations = {}
        self._annotations_loaded = False
        if annotations_file and Path(annotations_file).exists():
            file_size_gb = Path(annotations_file).stat().st_size / 1e9
            file_size_mb = file_size_gb * 1024
            
            # Auto-detect: use lazy loading only for files > 1GB
            if lazy_load_annotations is None:
                use_lazy_loading = file_size_gb > 1.0
                if use_lazy_loading:
                    print(f"Auto-detected: File is {file_size_gb:.2f} GB ({file_size_mb:.0f} MB) - using lazy loading")
                else:
                    print(f"Auto-detected: File is {file_size_mb:.0f} MB ({file_size_gb:.3f} GB) - loading directly (lazy loading not needed)")
            else:
                use_lazy_loading = lazy_load_annotations
                if use_lazy_loading:
                    print(f"Lazy loading enabled by user: File is {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)")
                else:
                    print(f"Lazy loading disabled by user: File is {file_size_mb:.0f} MB ({file_size_gb:.3f} GB)")
            
            if file_size_gb > 1.0:  # Warn if file is larger than 1GB
                print(f"WARNING: Large annotations file ({file_size_gb:.2f} GB) - consider: 1) Reducing num_workers to 0-1, 2) Using .mat/.png masks instead")
            
            if use_lazy_loading:
                print(f"Deferred loading: Annotations will be loaded on first access")
                # Don't load yet - will load on first __getitem__ call
                self._annotations_file = annotations_file
                self.lazy_load_annotations = True
            else:
                print(f"Loading annotations into memory...")
                self._load_annotations_optimized(annotations_file)
                self._annotations_loaded = True
                self.lazy_load_annotations = False
        else:
            self.lazy_load_annotations = False
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
        if train and max_crops_per_image > 1:
            print(f"Generating up to {max_crops_per_image} crops per image → ~{len(self.image_files) * max_crops_per_image} training samples")
    
    def _build_annotations_index(self, annotations_file: str):
        """Build an index for lazy loading annotations - only loads metadata, not full annotations"""
        import json
        print("Building annotations index (this may take a while for large files)...")
        
        # For very large files, we'll use a streaming approach to build an index
        # First, try to load just the images metadata (usually small)
        try:
            with open(annotations_file, 'r') as f:
                # Read file in chunks to find the images section
                # For COCO format, images come before annotations
                data_start = f.read(1000)  # Read first 1KB to check format
                f.seek(0)
                
                # Use ijson for streaming if available, otherwise fall back to full load with memory optimization
                try:
                    import ijson
                    print("Using ijson for streaming JSON parsing...")
                    
                    # Parse images
                    with open(annotations_file, 'rb') as f:
                        parser = ijson.parse(f)
                        current_key = None
                        image_list = []
                        ann_list = []
                        
                        for prefix, event, value in parser:
                            if prefix == 'images.item' and event == 'start_map':
                                current_image = {}
                            elif prefix.startswith('images.item.') and event == 'string' or event == 'number':
                                key = prefix.split('.')[-1]
                                current_image[key] = value
                            elif prefix == 'images.item' and event == 'end_map':
                                image_list.append(current_image)
                                self.image_id_by_name[current_image['file_name']] = current_image['id']
                                self.images_meta[current_image['id']] = current_image
                                self.annotations[current_image['file_name']] = {
                                    'width': current_image['width'],
                                    'height': current_image['height']
                                }
                                self.coco_image_by_name[current_image['file_name']] = current_image
                            
                            elif prefix.startswith('annotations.item.category_id') and event == 'number':
                                self.category_ids.add(value)
                        
                        # For annotations, we'll load them on-demand
                        # Store that we need to parse annotations lazily
                        self._has_ijson = True
                        
                except ImportError:
                    print("ijson not available, using optimized full load with memory cleanup...")
                    self._has_ijson = False
                    # Fall back to optimized full load
                    self._load_annotations_optimized(annotations_file)
                    return
                
        except Exception as e:
            print(f"Warning: Could not build lazy index, falling back to full load: {e}")
            self._load_annotations_optimized(annotations_file)
        
        # Build category mapping
        if self.category_ids:
            if self.collapse_categories:
                self.category_id_to_label = {cat_id: 1 for cat_id in self.category_ids}
            else:
                self.category_id_to_label = {
                    cat_id: idx + 1 for idx, cat_id in enumerate(sorted(self.category_ids))
                }
        
        print(f"Index built: {len(self.images_meta)} images, {len(self.category_ids)} categories")
        print("Annotations will be loaded on-demand per image to save memory")
    
    def _load_annotations_optimized(self, annotations_file: str):
        """Optimized full load that minimizes memory overhead"""
        import json
        import gc
        
        print("Loading annotations with memory optimization...")
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        images_meta = data.get('images', [])
        annotations_meta = data.get('annotations', [])
        
        # Load image metadata (small)
        for img_info in images_meta:
            self.annotations[img_info['file_name']] = {
                'width': img_info['width'],
                'height': img_info['height']
            }
            self.coco_image_by_name[img_info['file_name']] = img_info
            self.image_id_by_name[img_info['file_name']] = img_info['id']
            self.images_meta[img_info['id']] = img_info
        
        # Load annotations (large - this is the memory issue)
        # Group by image_id for faster lookup
        for ann in annotations_meta:
            image_id = ann.get('image_id')
            if image_id is not None:
                self.coco_annotations_by_image[image_id].append(ann)
                self.category_ids.add(ann.get('category_id', 1))
        
        # Build category mapping
        if self.category_ids:
            if self.collapse_categories:
                self.category_id_to_label = {cat_id: 1 for cat_id in self.category_ids}
            else:
                self.category_id_to_label = {
                    cat_id: idx + 1 for idx, cat_id in enumerate(sorted(self.category_ids))
                }
        
        # Clear large lists from memory
        del images_meta
        del annotations_meta
        gc.collect()
        
        print(f"Loaded {len(self.coco_image_by_name)} images, {sum(len(anns) for anns in self.coco_annotations_by_image.values())} annotations")
    
    def _load_annotations_full(self, annotations_file: str):
        """Original full load method (for backward compatibility)"""
        import json
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            images_meta = data.get('images', [])
            annotations_meta = data.get('annotations', [])

            for img_info in images_meta:
                self.annotations[img_info['file_name']] = {
                    'width': img_info['width'],
                    'height': img_info['height']
                }
                self.coco_image_by_name[img_info['file_name']] = img_info
                self.image_id_by_name[img_info['file_name']] = img_info['id']
                self.images_meta[img_info['id']] = img_info

            for ann in annotations_meta:
                image_id = ann.get('image_id')
                if image_id is not None:
                    self.coco_annotations_by_image[image_id].append(ann)
                    self.category_ids.add(ann.get('category_id', 1))

            if annotations_meta:
                if self.collapse_categories:
                    self.category_id_to_label = {cat_id: 1 for cat_id in self.category_ids}
                else:
                    self.category_id_to_label = {
                        cat_id: idx + 1 for idx, cat_id in enumerate(sorted(self.category_ids))
                    }
    
    def _load_annotations_for_image(self, img_path: Path) -> List[Dict]:
        """Load annotations for a specific image on-demand"""
        # Deferred loading: load annotations on first access if using lazy loading
        if self.lazy_load_annotations and not self._annotations_loaded and hasattr(self, '_annotations_file'):
            print("Loading annotations on first access (deferred loading to save memory at init)...")
            self._load_annotations_optimized(self._annotations_file)
            self._annotations_loaded = True
        
        # Return annotations for this image
        image_info = self.coco_image_by_name.get(img_path.name)
        if image_info:
            image_id = image_info['id']
            return self.coco_annotations_by_image.get(image_id, [])
        return []
    
    def __len__(self):
        return len(self.image_files) * self.max_crops_per_image
    
    def _get_random_crop_coords(self, img_w, img_h):
        """Get random crop coordinates ensuring we stay within bounds"""
        if img_w <= self.crop_size and img_h <= self.crop_size:
            # Image smaller than crop, pad later
            return 0, 0
        
        max_x = max(0, img_w - self.crop_size)
        max_y = max(0, img_h - self.crop_size)
        
        if self.train:
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)
        else:
            # Center crop for validation
            crop_x = max_x // 2
            crop_y = max_y // 2
        
        return crop_x, crop_y
    
    def _decode_coco_annotation(self, ann: Dict, orig_h: int, orig_w: int):
        """Decode a COCO annotation into normalized box and optional mask."""
        x, y, w, h = ann['bbox']
        cx = (x + w / 2.0) / orig_w
        cy = (y + h / 2.0) / orig_h
        box_w = w / orig_w
        box_h = h / orig_h

        if self.category_id_to_label:
            label = self.category_id_to_label.get(ann['category_id'], 1)
        else:
            label = ann.get('category_id', 1)

        mask = None
        segmentation = ann.get('segmentation')
        if not segmentation:
            # No segmentation data
            return [cx, cy, box_w, box_h], label, None
        if mask_utils is None:
            # mask_utils not available
            return [cx, cy, box_w, box_h], label, None
        
        try:
            if isinstance(segmentation, dict):
                # Check if RLE counts is a list (uncompressed) or string (compressed)
                counts = segmentation.get('counts', '')
                rle_size = segmentation.get('size', [])
                
                # Convert uncompressed RLE (list) to binary mask
                if isinstance(counts, list):
                    # Uncompressed RLE format: list of run lengths
                    # COCO RLE uses column-major (Fortran) ordering, not row-major
                    # Runs alternate between background (0) and foreground (1), starting with background
                    # Format: [bg_run1, fg_run1, bg_run2, fg_run2, ...]
                    if len(rle_size) == 2:
                        rle_h, rle_w = rle_size
                        # Use pycocotools decode if available (handles both compressed and uncompressed)
                        # For uncompressed RLE, pycocotools.decode() should handle it correctly
                        if mask_utils is not None:
                            try:
                                # Create RLE dict in format pycocotools expects
                                rle_dict = {
                                    'counts': counts,
                                    'size': [rle_h, rle_w]
                                }
                                # pycocotools.decode() handles uncompressed RLE (list format) correctly
                                mask = mask_utils.decode(rle_dict)
                            except Exception as e:
                                # If decode fails, fall back to manual decoding with correct column-major ordering
                                print(f"[DEBUG _decode_coco_annotation] pycocotools.decode() failed for uncompressed RLE: {e}")
                                # Fallback: manual decoding with correct column-major ordering
                                mask_flat = np.zeros(rle_h * rle_w, dtype=np.uint8)
                                pos = 0
                                for i, length in enumerate(counts):
                                    if i % 2 == 1:  # Odd index = foreground run (1)
                                        end = min(pos + length, rle_h * rle_w)
                                        mask_flat[pos:end] = 1
                                    # Move position forward (both bg and fg runs advance position)
                                    pos += length
                                    if pos >= rle_h * rle_w:
                                        break
                                # Reshape to 2D using Fortran (column-major) order for COCO RLE
                                # This is critical: COCO RLE uses column-major ordering, not row-major
                                mask = mask_flat.reshape(rle_h, rle_w, order='F')
                        else:
                            # Fallback: manual decoding with correct column-major ordering
                            mask_flat = np.zeros(rle_h * rle_w, dtype=np.uint8)
                            pos = 0
                            for i, length in enumerate(counts):
                                if i % 2 == 1:  # Odd index = foreground run (1)
                                    end = min(pos + length, rle_h * rle_w)
                                    mask_flat[pos:end] = 1
                                # Move position forward (both bg and fg runs advance position)
                                pos += length
                                if pos >= rle_h * rle_w:
                                    break
                            # Reshape to 2D using Fortran (column-major) order for COCO RLE
                            mask = mask_flat.reshape(rle_h, rle_w, order='F')
                        
                        # Resize if needed to match original image size
                        if rle_h != orig_h or rle_w != orig_w:
                            from PIL import Image as PILImage
                            mask_pil = PILImage.fromarray(mask)
                            mask_pil = mask_pil.resize((orig_w, orig_h), PILImage.NEAREST)
                            mask = np.array(mask_pil)
                    else:
                        mask = None
                else:
                    # Compressed RLE (string format) - decode directly
                    if len(rle_size) == 2:
                        rle_h, rle_w = rle_size
                        if rle_h != orig_h or rle_w != orig_w:
                            # RLE is for different size, need to decode and resize
                            mask = mask_utils.decode(segmentation)
                            # Resize mask to match original image size
                            from PIL import Image as PILImage
                            mask_pil = PILImage.fromarray(mask)
                            mask_pil = mask_pil.resize((orig_w, orig_h), PILImage.NEAREST)
                            mask = np.array(mask_pil)
                        else:
                            mask = mask_utils.decode(segmentation)
                    else:
                        mask = mask_utils.decode(segmentation)
            else:
                rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                mask = mask_utils.decode(rles)

            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = np.asarray(mask, dtype=np.uint8)
            
            # Verify mask is not empty
            mask_sum = mask.sum()
            if mask_sum == 0:
                # Mask is empty - this shouldn't happen for valid annotations
                return [cx, cy, box_w, box_h], label, None
        except Exception as e:
            # Log the error for debugging
            print(f"[DEBUG _decode_coco_annotation] Failed to decode mask for annotation {ann.get('id', 'unknown')}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            mask = None

        return [cx, cy, box_w, box_h], label, mask

    def _load_masks(self, img_path, orig_h, orig_w):
        """Load masks from .mat or .png files"""
        masks = []
        boxes = []
        labels = []
        
        # Load COCO annotations for this image (lazy loaded if enabled)
        coco_ann_list = self._load_annotations_for_image(img_path)
        image_info = self.coco_image_by_name.get(img_path.name)
        
        # Try .mat format first (HoverNet format)
        mat_path = self.labels_dir / (img_path.stem + '.mat')
        if mat_path.exists():
            mat_data = scipy.io.loadmat(str(mat_path))
            inst_map = mat_data['inst_map']  # Instance segmentation map
            
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
        else:
            # Try PNG format with various suffixes
            for suffix in ['.png_label.png', '_label.png', '.png']:
                png_path = self.labels_dir / (img_path.stem + suffix)
                if png_path.exists():
                    png_data = np.array(Image.open(png_path))
                    # If binary (only 0 and 255), convert to instance map using connected components
                    if len(np.unique(png_data)) <= 2:
                        from scipy import ndimage
                        binary = png_data > 0
                        inst_map, _ = ndimage.label(binary)
                    else:
                        inst_map = png_data
                    
                    # Extract instances
                    unique_instances = np.unique(inst_map)
                    unique_instances = unique_instances[unique_instances > 0]
                    
                    for inst_id in unique_instances:
                        mask = (inst_map == inst_id).astype(np.uint8)
                        rows, cols = np.where(mask)
                        if len(rows) == 0:
                            continue
                        
                        y1, y2 = rows.min(), rows.max()
                        x1, x2 = cols.min(), cols.max()
                        
                        center_x = (x1 + x2) / 2.0 / orig_w
                        center_y = (y1 + y2) / 2.0 / orig_h
                        width = (x2 - x1) / orig_w
                        height = (y2 - y1) / orig_h
                        
                        if width < 0.01 or height < 0.01:
                            continue
                        
                        boxes.append([center_x, center_y, width, height])
                        labels.append(1)
                        masks.append(mask)
                    break
        
        # If no masks found, try YOLO format (.txt)
        if len(masks) == 0:
            label_path = self.labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            boxes.append([cx, cy, w, h])
                            if self.collapse_categories:
                                labels.append(1)
                            else:
                                labels.append(class_id)
                            # Create dummy mask (will be created from box later)
                            masks.append(None)

        if coco_ann_list:
            # Always prefer COCO annotations when available - they have detailed per-instance annotations
            # PNG labels are coarse and only have a few instance IDs
            # When collapse_categories is True, always use COCO if available (we want full annotation set)
            # Otherwise, use COCO if PNG has way fewer boxes
            if self.collapse_categories:
                # Always use COCO when collapse_categories is enabled and COCO annotations exist
                should_use_coco = True
            else:
                should_use_coco = (
                    len(boxes) == 0 or 
                    len(boxes) < len(coco_ann_list) // 2
                )
            # Debug prints disabled to reduce log verbosity
            # Uncomment below if debugging is needed:
            # if self.train:
            #     print(f"[DEBUG _load_masks] Image: {img_path.name}, PNG boxes: {len(boxes)}, COCO annotations: {len(coco_ann_list)}, "
            #           f"should_use_coco: {should_use_coco}, collapse_categories: {self.collapse_categories}")
            if should_use_coco:
                # Check if mask_utils is available
                if mask_utils is None:
                    if self.train:
                        print(f"[DEBUG _load_masks] WARNING: mask_utils (pycocotools) is not available! Cannot decode RLE masks.")
                        print(f"[DEBUG _load_masks] Install with: pip install pycocotools")
                
                # Use COCO annotations instead of PNG/mask-based boxes
                boxes = []
                labels = []
                masks = []
                masks_decoded = 0
                masks_failed = 0
                for ann in coco_ann_list:
                    box_norm, label, mask = self._decode_coco_annotation(ann, orig_h, orig_w)
                    boxes.append(box_norm)
                    labels.append(label)
                    masks.append(mask)
                    if mask is not None:
                        masks_decoded += 1
                    else:
                        masks_failed += 1
                if masks_failed > 0 and self.train:
                    print(f"[DEBUG _load_masks] COCO masks: {masks_decoded} decoded, {masks_failed} failed (will create from boxes)")
                    if mask_utils is None:
                        print(f"[DEBUG _load_masks] Reason: mask_utils is None - install pycocotools")
        else:
            # Debug: Check why COCO annotations weren't found
            if self.coco_image_by_name or self.coco_annotations_by_image:
                if self.train:  # Only print during training
                    print(f"[DEBUG] COCO annotations not found for {img_path.name}")

        return boxes, labels, masks
    
    def _crop_and_adjust_boxes_masks(self, image, boxes_norm, labels, masks_list, orig_w, orig_h, crop_x, crop_y):
        """Crop image and adjust boxes and masks to new coordinate system"""
        # Convert normalized boxes to absolute coordinates
        if len(boxes_norm) > 0:
            boxes_abs = boxes_norm.clone()
            # YOLO format: [cx, cy, w, h] normalized
            cx_abs = boxes_abs[:, 0] * orig_w
            cy_abs = boxes_abs[:, 1] * orig_h
            w_abs = boxes_abs[:, 2] * orig_w
            h_abs = boxes_abs[:, 3] * orig_h
            
            # Convert to [x1, y1, x2, y2]
            x1 = cx_abs - w_abs / 2
            y1 = cy_abs - h_abs / 2
            x2 = cx_abs + w_abs / 2
            y2 = cy_abs + h_abs / 2
            
            # Adjust for crop offset
            x1_crop = x1 - crop_x
            y1_crop = y1 - crop_y
            x2_crop = x2 - crop_x
            y2_crop = y2 - crop_y
            
            # Calculate actual crop dimensions (may be smaller than crop_size if image is small)
            actual_crop_w = min(self.crop_size, orig_w - crop_x)
            actual_crop_h = min(self.crop_size, orig_h - crop_y)
            
            # Filter boxes that overlap with crop region (use actual crop dimensions, not crop_size)
            keep = (x2_crop > 0) & (x1_crop < actual_crop_w) & \
                   (y2_crop > 0) & (y1_crop < actual_crop_h)
            
            if keep.sum() == 0:
                return None, torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64), torch.zeros((0, self.crop_size, self.crop_size), dtype=torch.float32)
            
            # Clip to actual crop boundaries (not crop_size, to avoid boxes in padding)
            x1_crop = torch.clamp(x1_crop[keep], 0, actual_crop_w)
            y1_crop = torch.clamp(y1_crop[keep], 0, actual_crop_h)
            x2_crop = torch.clamp(x2_crop[keep], 0, actual_crop_w)
            y2_crop = torch.clamp(y2_crop[keep], 0, actual_crop_h)
            
            # Filter out boxes that became too small after clipping
            box_widths = x2_crop - x1_crop
            box_heights = y2_crop - y1_crop
            valid = (box_widths > 2) & (box_heights > 2)
            
            if valid.sum() == 0:
                return None, torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64), torch.zeros((0, self.crop_size, self.crop_size), dtype=torch.float32)
            
            x1_crop = x1_crop[valid]
            y1_crop = y1_crop[valid]
            x2_crop = x2_crop[valid]
            y2_crop = y2_crop[valid]
            labels_crop = labels[keep][valid]
            
            # Crop and adjust masks
            masks_cropped = []
            if masks_list:
                # Get indices that passed both keep and valid filters
                keep_indices = torch.where(keep)[0]
                valid_indices = keep_indices[valid]
                
                masks_none_count = 0
                masks_valid_count = 0
                for valid_idx in valid_indices:
                    orig_idx = valid_idx.item()
                    mask = masks_list[orig_idx] if orig_idx < len(masks_list) else None
                    
                    if mask is None:
                        masks_none_count += 1
                        # Create dummy mask from box
                        mask_crop = torch.zeros((self.crop_size, self.crop_size), dtype=torch.float32)
                        # Find the box index in the filtered set
                        box_idx = torch.where(valid_indices == valid_idx)[0]
                        if len(box_idx) > 0:
                            box_idx = box_idx[0].item()
                            if box_idx < len(x1_crop):
                                y1_int = max(0, int(y1_crop[box_idx]))
                                y2_int = min(self.crop_size, int(y2_crop[box_idx]))
                                x1_int = max(0, int(x1_crop[box_idx]))
                                x2_int = min(self.crop_size, int(x2_crop[box_idx]))
                                mask_crop[y1_int:y2_int, x1_int:x2_int] = 1.0
                        masks_cropped.append(mask_crop)
                    else:
                        masks_valid_count += 1
                        # Crop mask
                        mask_np = mask.astype(np.float32)
                        # Crop the mask array
                        mask_crop_np = mask_np[crop_y:crop_y+self.crop_size, crop_x:crop_x+self.crop_size]
                        # Pad if necessary
                        if mask_crop_np.shape[0] < self.crop_size or mask_crop_np.shape[1] < self.crop_size:
                            pad_h = max(0, self.crop_size - mask_crop_np.shape[0])
                            pad_w = max(0, self.crop_size - mask_crop_np.shape[1])
                            mask_crop_np = np.pad(mask_crop_np, ((0, pad_h), (0, pad_w)), mode='constant')
                        # Crop to exact size if larger
                        mask_crop_np = mask_crop_np[:self.crop_size, :self.crop_size]
                        masks_cropped.append(torch.tensor(mask_crop_np, dtype=torch.float32))
                
                # Debug: Log mask status (only during training, reduce output)
                if masks_none_count > 0 and self.train:
                    print(f"[DEBUG _crop_and_adjust_boxes_masks] Masks: {masks_valid_count} valid, {masks_none_count} None (creating rectangles)")
            
            if len(masks_cropped) == 0:
                masks_cropped = torch.zeros((0, self.crop_size, self.crop_size), dtype=torch.float32)
            else:
                masks_cropped = torch.stack(masks_cropped)
            
            # Convert back to normalized [cx, cy, w, h] format
            boxes_cropped = torch.stack([
                (x1_crop + x2_crop) / 2 / self.crop_size,  # cx normalized
                (y1_crop + y2_crop) / 2 / self.crop_size,  # cy normalized
                (x2_crop - x1_crop) / self.crop_size,       # w normalized
                (y2_crop - y1_crop) / self.crop_size        # h normalized
            ], dim=1)
        else:
            return None, torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64), torch.zeros((0, self.crop_size, self.crop_size), dtype=torch.float32)
        
        # Perform the crop on image
        image_crop = T.functional.crop(
            image, 
            top=crop_y, 
            left=crop_x, 
            height=min(self.crop_size, orig_h - crop_y),
            width=min(self.crop_size, orig_w - crop_x)
        )
        
        # Store actual content dimensions before padding
        actual_crop_w = image_crop.width
        actual_crop_h = image_crop.height
        
        # Pad if necessary (for images smaller than crop_size)
        if image_crop.size != (self.crop_size, self.crop_size):
            image_crop = T.functional.pad(
                image_crop,
                padding=[0, 0, self.crop_size - image_crop.width, self.crop_size - image_crop.height],
                fill=0
            )
            
            # Filter out boxes that fall in the padding area
            if len(boxes_cropped) > 0:
                # Boxes are in normalized coordinates [cx, cy, w, h] relative to crop_size
                # Convert to absolute to check against actual content size
                boxes_abs = boxes_cropped.clone()
                cx_abs = boxes_abs[:, 0] * self.crop_size
                cy_abs = boxes_abs[:, 1] * self.crop_size
                w_abs = boxes_abs[:, 2] * self.crop_size
                h_abs = boxes_abs[:, 3] * self.crop_size
                
                # Get box boundaries
                x1_abs = cx_abs - w_abs / 2
                y1_abs = cy_abs - h_abs / 2
                x2_abs = cx_abs + w_abs / 2
                y2_abs = cy_abs + h_abs / 2
                
                # Keep only boxes that overlap with actual content (not entirely in padding)
                # A box is valid if any part of it is within the actual content area
                keep_in_content = (x2_abs > 0) & (x1_abs < actual_crop_w) & \
                                 (y2_abs > 0) & (y1_abs < actual_crop_h)
                
                if keep_in_content.sum() < len(boxes_cropped):
                    # Filter boxes and masks
                    boxes_cropped = boxes_cropped[keep_in_content]
                    labels_crop = labels_crop[keep_in_content]
                    if masks_cropped.shape[0] > 0:
                        masks_cropped = masks_cropped[keep_in_content]
                    else:
                        masks_cropped = torch.zeros((0, self.crop_size, self.crop_size), dtype=torch.float32)
        
        return image_crop, boxes_cropped, labels_crop, masks_cropped
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Map to original image index
        original_idx = idx % len(self.image_files)
        
        # Load image with error handling for corrupted files
        img_path = self.image_files[original_idx]
        
        # Check if file exists and has content
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        file_size = os.path.getsize(img_path)
        if file_size == 0:
            # If file is empty, try next image
            print(f"WARNING: Empty image file {img_path}, trying next image...")
            original_idx = (original_idx + 1) % len(self.image_files)
            img_path = self.image_files[original_idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            # Verify image is valid (has dimensions)
            if orig_w <= 0 or orig_h <= 0:
                raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")
        except (OSError, IOError, struct.error, ValueError) as e:
            # If image is corrupted, try next image
            original_img_path = img_path
            print(f"WARNING: Corrupted/invalid image file {original_img_path}: {e}, trying next image...")
            original_idx = (original_idx + 1) % len(self.image_files)
            img_path = self.image_files[original_idx]
            try:
                image = Image.open(img_path).convert('RGB')
                orig_w, orig_h = image.size
                if orig_w <= 0 or orig_h <= 0:
                    raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")
            except (OSError, IOError, struct.error, ValueError) as e2:
                # If still corrupted, raise error
                raise OSError(f"Failed to load image after retry. Original: {original_img_path}, Retry: {img_path}, Error: {e2}")
        
        # Load masks/boxes
        boxes, labels, masks_list = self._load_masks(img_path, orig_h, orig_w)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks_list = []
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            # Convert masks to numpy arrays if they aren't already
            if masks_list and masks_list[0] is not None:
                masks_list = [np.array(m) if isinstance(m, np.ndarray) else m for m in masks_list]
        
        # Try multiple random crops until we get one with enough nuclei
        max_attempts = 10 if self.train else 1
        
        for attempt in range(max_attempts):
            crop_x, crop_y = self._get_random_crop_coords(orig_w, orig_h)
            image_crop, boxes_crop, labels_crop, masks_crop = self._crop_and_adjust_boxes_masks(
                image, boxes, labels, masks_list, orig_w, orig_h, crop_x, crop_y
            )
            
            if image_crop is not None and len(boxes_crop) >= self.min_nuclei_per_crop:
                break
            
            # If last attempt and still no valid crop, return empty
            if attempt == max_attempts - 1:
                # Just take center crop even if empty
                crop_x = max(0, (orig_w - self.crop_size) // 2)
                crop_y = max(0, (orig_h - self.crop_size) // 2)
                image_crop, boxes_crop, labels_crop, masks_crop = self._crop_and_adjust_boxes_masks(
                    image, boxes, labels, masks_list, orig_w, orig_h, crop_x, crop_y
                )
                if image_crop is None:
                    # Last resort: crop top-left
                    image_crop = T.functional.crop(image, 0, 0, 
                                                   min(self.crop_size, orig_h),
                                                   min(self.crop_size, orig_w))
                    if image_crop.size != (self.crop_size, self.crop_size):
                        image_crop = T.functional.pad(
                            image_crop,
                            [0, 0, self.crop_size - image_crop.width, self.crop_size - image_crop.height],
                            fill=0
                        )
                    boxes_crop = torch.zeros((0, 4), dtype=torch.float32)
                    labels_crop = torch.zeros((0,), dtype=torch.int64)
                    masks_crop = torch.zeros((0, self.crop_size, self.crop_size), dtype=torch.float32)
        
        # Apply color augmentations (doesn't affect boxes)
        if self.train:
            if random.random() < 0.8:
                image_crop = T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                )(image_crop)
            
            # Horizontal flip
            if random.random() < 0.5:
                image_crop = T.functional.hflip(image_crop)
                if len(boxes_crop) > 0:
                    boxes_crop[:, 0] = 1.0 - boxes_crop[:, 0]
                if masks_crop.shape[0] > 0:
                    masks_crop = torch.flip(masks_crop, dims=[2])
            
            # Vertical flip
            if random.random() < 0.5:
                image_crop = T.functional.vflip(image_crop)
                if len(boxes_crop) > 0:
                    boxes_crop[:, 1] = 1.0 - boxes_crop[:, 1]
                if masks_crop.shape[0] > 0:
                    masks_crop = torch.flip(masks_crop, dims=[1])
        
        # Convert to tensor
        image_tensor = T.ToTensor()(image_crop)
        
        # Create target
        target = {
            'boxes': boxes_crop,
            'labels': labels_crop,
            'masks': masks_crop,
            'image_id': torch.tensor([original_idx]),
            'orig_size': torch.tensor([self.crop_size, self.crop_size], dtype=torch.float32)
        }
        
        return image_tensor, target


def collate_fn(batch):
    """Custom collate function"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


def create_dataloaders(
    train_images_dir: str,
    train_labels_dir: str,
    val_images_dir: str,
    val_labels_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    crop_size: int = 224,
    max_crops_per_image: int = 4,
    train_annotations_file: str = None,
    val_annotations_file: str = None,
    collapse_categories: bool = False,
    lazy_load_annotations: bool = None,  # Auto-detect based on file size (None = auto, True/False = override)
):
    """Create train and validation dataloaders"""
    
    train_dataset = MaskRCNNCropDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        crop_size=crop_size,
        train=True,
        max_crops_per_image=max_crops_per_image,
        min_nuclei_per_crop=1,
        annotations_file=train_annotations_file,
        collapse_categories=collapse_categories,
        lazy_load_annotations=lazy_load_annotations,
    )
    
    val_dataset = MaskRCNNCropDataset(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        crop_size=crop_size,
        train=False,
        max_crops_per_image=1,
        min_nuclei_per_crop=0,  # Allow empty crops in validation
        annotations_file=val_annotations_file,
        collapse_categories=collapse_categories,
        lazy_load_annotations=lazy_load_annotations,
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
    base_path = "/path/to/your/dataset"
    
    train_images = os.path.join(base_path, "train/images")
    train_labels = os.path.join(base_path, "train/labels")
    val_images = os.path.join(base_path, "valid/images")
    val_labels = os.path.join(base_path, "valid/labels")
    
    train_loader, val_loader = create_dataloaders(
        train_images_dir=train_images,
        train_labels_dir=train_labels,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        batch_size=8,
        num_workers=4,
        crop_size=224,
        max_crops_per_image=4
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
        print(f"  Number of nuclei: {len(target['boxes'])}")
        print(f"  Masks shape: {target['masks'].shape}")
        if len(target['boxes']) > 0:
            print(f"  First box (normalized cx, cy, w, h): {target['boxes'][0]}")

