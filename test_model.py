"""
Test script to verify Mask R-CNN model creation and forward pass
"""
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_creation():
    """Test if the model can be created successfully"""
    print("Testing Mask R-CNN model creation...")
    
    try:
        from dinov2_maskrcnn_improved import create_improved_maskrcnn
        
        # Use the same checkpoint path as in the original project
        dinov2_checkpoint = "/rodata/dlmp_path/han/data/m328672/dinov2-main/dgx_vitg14_patch37M/training_287499/teacher_checkpoint.pth"
        
        model = create_improved_maskrcnn(
            dinov2_checkpoint_path=dinov2_checkpoint,
            freeze_backbone=True
        )
        
        print("✅ Model created successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("\nTesting forward pass...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = [torch.randn(3, 518, 518).to(device)]
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✅ Forward pass successful!")
        print(f"Output keys: {list(output[0].keys())}")
        print(f"Number of detections: {len(output[0]['boxes'])}")
        
        if len(output[0]['boxes']) > 0:
            print(f"Score range: [{output[0]['scores'].min():.3f}, {output[0]['scores'].max():.3f}]")
            print(f"Masks shape: {output[0]['masks'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def test_training_mode(model):
    """Test training mode with dummy targets"""
    print("\nTesting training mode...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.train()
        
        # Create dummy input and targets
        dummy_input = [torch.randn(3, 518, 518).to(device)]
        dummy_targets = [{
            'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]], dtype=torch.float32).to(device),  # Normalized YOLO format
            'labels': torch.tensor([1], dtype=torch.int64).to(device),
            'orig_size': torch.tensor([518, 518], dtype=torch.float32).to(device)
        }]
        
        # Forward pass in training mode
        loss_dict = model(dummy_input, dummy_targets)
        
        print("✅ Training mode successful!")
        print(f"Loss keys: {list(loss_dict.keys())}")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training mode failed: {e}")
        return False


def main():
    print("=" * 50)
    print("Mask R-CNN Model Test")
    print("=" * 50)
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        print("❌ Model creation failed, exiting...")
        return
    
    # Test forward pass
    if not test_forward_pass(model):
        print("❌ Forward pass failed, exiting...")
        return
    
    # Test training mode
    if not test_training_mode(model):
        print("❌ Training mode failed, exiting...")
        return
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Model is ready for training.")
    print("=" * 50)


if __name__ == "__main__":
    main()
