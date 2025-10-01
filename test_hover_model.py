"""
Test script to verify Mask R-CNN model with HoverNet dataset
"""
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_hover_dataset():
    """Test HoverNet dataset loading"""
    print("Testing HoverNet dataset loading...")
    
    try:
        from hover_dataset_loader import HoverNetDataset, get_transform
        
        # Test dataset loading
        dataset = HoverNetDataset(
            images_dir="/rodata/dlmp_path/han/data/hovernet_dataset/hoverdata/train/Images",
            labels_dir="/rodata/dlmp_path/han/data/hovernet_dataset/hoverdata/train/Labels",
            annotations_file="/rodata/dlmp_path/han/data/hovernet_dataset/hoverdata/train/train_annotations.json",
            transform=get_transform(train=True, target_size=518),
            train=True
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a sample
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Target keys: {list(target.keys())}")
        print(f"Number of nuclei: {len(target['boxes'])}")
        print(f"Masks shape: {target['masks'].shape}")
        print(f"Boxes shape: {target['boxes'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False


def test_model_with_real_masks():
    """Test Mask R-CNN model with real masks"""
    print("\nTesting Mask R-CNN model with real masks...")
    
    try:
        from dinov2_maskrcnn_improved import create_improved_maskrcnn
        from hover_dataset_loader import HoverNetDataset, get_transform
        
        # Create model
        dinov2_checkpoint = "/rodata/dlmp_path/han/data/m328672/dinov2-main/dgx_vitg14_patch37M/training_287499/teacher_checkpoint.pth"
        model = create_improved_maskrcnn(
            dinov2_checkpoint_path=dinov2_checkpoint,
            freeze_backbone=True
        )
        
        # Create dataset
        dataset = HoverNetDataset(
            images_dir="/rodata/dlmp_path/han/data/hovernet_dataset/hoverdata/train/Images",
            labels_dir="/rodata/dlmp_path/han/data/hovernet_dataset/hoverdata/train/Labels",
            annotations_file="/rodata/dlmp_path/han/data/hovernet_dataset/hoverdata/train/train_annotations.json",
            transform=get_transform(train=True, target_size=518),
            train=True
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.train()
        
        # Get a sample
        image, target = dataset[0]
        image = image.to(device)  # Keep as 3D tensor [C, H, W]
        target = [{k: v.to(device) for k, v in target.items()}]
        
        print(f"Input image shape: {image.shape}")
        print(f"Target masks shape: {target[0]['masks'].shape}")
        print(f"Target boxes shape: {target[0]['boxes'].shape}")
        
        # Forward pass in training mode
        loss_dict = model([image], target)
        
        print("✅ Training mode with real masks successful!")
        print(f"Loss keys: {list(loss_dict.keys())}")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training with real masks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_mode():
    """Test inference mode"""
    print("\nTesting inference mode...")
    
    try:
        from dinov2_maskrcnn_improved import create_improved_maskrcnn
        
        # Create model
        dinov2_checkpoint = "/rodata/dlmp_path/han/data/m328672/dinov2-main/dgx_vitg14_patch37M/training_287499/teacher_checkpoint.pth"
        model = create_improved_maskrcnn(
            dinov2_checkpoint_path=dinov2_checkpoint,
            freeze_backbone=True
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = [torch.randn(3, 518, 518).to(device)]
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✅ Inference mode successful!")
        print(f"Output keys: {list(output[0].keys())}")
        print(f"Number of detections: {len(output[0]['boxes'])}")
        if len(output[0]['boxes']) > 0:
            print(f"Score range: [{output[0]['scores'].min():.3f}, {output[0]['scores'].max():.3f}]")
            print(f"Masks shape: {output[0]['masks'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference mode failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Mask R-CNN with HoverNet Dataset Test")
    print("=" * 60)
    
    # Test dataset loading
    if not test_hover_dataset():
        print("❌ Dataset test failed, exiting...")
        return
    
    # Test model with real masks
    if not test_model_with_real_masks():
        print("❌ Model training test failed, exiting...")
        return
    
    # Test inference mode
    if not test_inference_mode():
        print("❌ Inference test failed, exiting...")
        return
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Model is ready for training with real masks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
