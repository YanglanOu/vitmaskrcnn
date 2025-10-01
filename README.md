# ViT Mask R-CNN Training Pipeline

This project implements a Mask R-CNN training pipeline using DINOv2 as the backbone, adapted from the ViT Faster R-CNN project.

## Project Structure

```
vitmaskrcnn/
├── dinov2_maskrcnn_improved.py    # Mask R-CNN model with DINOv2 backbone
├── train_maskrcnn_improved.py     # Training script
├── maskrcnn_inference.py          # Inference script
├── test_model.py                  # Model testing script
├── yolo_dataset_loader.py         # Dataset loader (copied from vitfasterrcnn)
├── dinov2_detr.py                 # DINOv2 loading utilities (copied from vitfasterrcnn)
├── creat_model.py                 # Model creation script
└── README.md                      # This file
```

## Key Features

- **DINOv2 Backbone**: Uses pre-trained DINOv2 ViT-G as the feature extractor
- **Mask R-CNN Head**: Includes both bounding box detection and instance segmentation
- **YOLO Dataset Support**: Loads YOLO format annotations with proper coordinate conversion
- **Mixed Precision Training**: Supports automatic mixed precision for faster training
- **Comprehensive Evaluation**: Includes precision, recall, F1 score, and visualization
- **Augmentation Support**: YOLO-style augmentations with configurable multiplier

## Model Architecture

The model consists of:
1. **DINOv2 Backbone**: Extracts patch features from input images
2. **Feature Projection**: Projects DINOv2 features to 256 channels
3. **RPN (Region Proposal Network)**: Generates object proposals
4. **RoI Heads**: 
   - **Box Head**: Predicts bounding boxes and class labels
   - **Mask Head**: Predicts instance segmentation masks

## Installation

1. Activate the UNI conda environment:
```bash
conda activate UNI
```

2. Ensure you have the required dependencies (PyTorch, torchvision, etc.)

## Usage

### 1. Test Model Creation

First, verify that the model can be created and runs correctly:

```bash
python test_model.py
```

This will test:
- Model creation with DINOv2 checkpoint loading
- Forward pass in inference mode
- Forward pass in training mode with loss computation

### 2. Training

Train the model on your dataset:

#### For HoverNet Format (with real masks):
```bash
python train_maskrcnn_improved.py \
    --data_root /rodata/dlmp_path/han/data/hovernet_dataset/hoverdata \
    --dinov2_checkpoint /path/to/dinov2/checkpoint.pth \
    --batch_size 4 \
    --num_epochs 50 \
    --lr 5e-4 \
    --use_amp \
    --augmentation_multiplier 3 \
    --output_dir ./outputs_maskrcnn_improved
```

#### For YOLO Format (with dummy masks):
```bash
# Modify train_maskrcnn_improved.py to use yolo_dataset_loader instead of hover_dataset_loader
python train_maskrcnn_improved.py \
    --data_root /path/to/yolo/dataset \
    --dinov2_checkpoint /path/to/dinov2/checkpoint.pth \
    --batch_size 4 \
    --num_epochs 50 \
    --lr 5e-4 \
    --use_amp \
    --output_dir ./outputs_maskrcnn_improved
```

#### Training Arguments

- `--data_root`: Path to dataset root (HoverNet: `train/Images`, `train/Labels`; YOLO: `images/train`, `labels/train`)
- `--dinov2_checkpoint`: Path to DINOv2 checkpoint file
- `--batch_size`: Batch size for training (default: 4)
- `--num_epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 5e-4)
- `--use_amp`: Enable automatic mixed precision training
- `--augmentation_multiplier`: Multiply training dataset size with augmentations (default: 1)
- `--freeze_backbone`: Freeze DINOv2 backbone parameters
- `--output_dir`: Output directory for checkpoints and logs

### 3. Inference

Run inference on a single image:

```bash
python maskrcnn_inference.py \
    --image_path /path/to/image.jpg \
    --checkpoint_path /path/to/best_model.pth \
    --dinov2_checkpoint /path/to/dinov2/checkpoint.pth \
    --output_path /path/to/output.png \
    --score_threshold 0.5
```

#### Inference Arguments

- `--image_path`: Path to input image
- `--checkpoint_path`: Path to trained model checkpoint
- `--dinov2_checkpoint`: Path to DINOv2 checkpoint file
- `--output_path`: Path to save visualization (optional)
- `--score_threshold`: Score threshold for detections (default: 0.5)

## Dataset Format

The project supports two dataset formats:

### 1. HoverNet Format (Recommended)
For datasets with instance segmentation masks:

```
dataset/
├── train/
│   ├── Images/
│   │   ├── image1.png
│   │   └── image2.png
│   ├── Labels/
│   │   ├── image1.mat
│   │   └── image2.mat
│   └── train_annotations.json
└── val/
    ├── Images/
    ├── Labels/
    └── val_annotations.json
```

- **Images**: PNG/JPG/TIF files
- **Labels**: `.mat` files containing `inst_map` (instance segmentation map)
- **Annotations**: JSON files with image metadata (optional)

### 2. YOLO Format (Legacy)
For datasets with only bounding boxes:

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image1.jpg
│       └── image2.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image1.txt
        └── image2.txt
```

Each YOLO label file contains one line per object:
```
class_id center_x center_y width height
```

All coordinates are normalized to [0, 1].

## Model Outputs

### Training Mode
Returns a dictionary of losses:
- `loss_classifier`: Classification loss
- `loss_box_reg`: Bounding box regression loss
- `loss_mask`: Mask prediction loss
- `loss_objectness`: RPN objectness loss
- `loss_rpn_box_reg`: RPN box regression loss

### Inference Mode
Returns a list of dictionaries (one per image) containing:
- `boxes`: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
- `labels`: Class labels [N]
- `scores`: Confidence scores [N]
- `masks`: Instance masks [N, 1, H, W]

## Key Differences from Faster R-CNN

1. **Mask Head**: Added mask prediction head for instance segmentation
2. **Mask Loss**: Includes mask loss in training
3. **Mask Visualization**: Enhanced visualization includes mask overlays
4. **Real Masks**: Uses actual instance segmentation masks from HoverNet dataset
5. **HoverNet Support**: Native support for .mat files with instance maps

## Performance

- **Total Parameters**: ~1.15B (mostly from DINOv2 backbone)
- **Trainable Parameters**: ~17.6M (when backbone is frozen)
- **Memory Usage**: ~8-12GB GPU memory for batch size 4

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **DINOv2 Checkpoint Not Found**: Ensure the checkpoint path is correct
3. **Dataset Loading Issues**: Check that image and label paths are correct
4. **Import Errors**: Make sure you're in the UNI conda environment

## Future Improvements

- [ ] Support for real mask annotations instead of dummy masks
- [ ] Multi-scale training
- [ ] Advanced augmentation strategies
- [ ] Model ensemble support
- [ ] Export to ONNX/TensorRT for deployment
