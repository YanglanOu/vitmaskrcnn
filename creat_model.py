import timm
import torch

# Method 1: Load pretrained model directly from timm
# This loads both architecture and pretrained weights
# model = timm.create_model('vit_giant_patch14_dinov2.lvd142m', pretrained=True)

# Method 2: Load architecture only, then load custom checkpoint
# First, create model without pretrained weights
model = timm.create_model('vit_giant_patch14_dinov2.lvd142m', pretrained=False)

# Then load your custom checkpoint
checkpoint = torch.load('/rodata/dlmp_path/han/data/m328672/dinov2-main/dgx_vitg14_patch37M/training_287499/teacher_checkpoint.pth', map_location='cpu')

# Load state dict (adjust based on your checkpoint structure)
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
elif 'teacher' in checkpoint:
    model.load_state_dict(checkpoint['teacher'])
else:
    model.load_state_dict(checkpoint)

# Set to evaluation mode if you're doing inference
model.eval()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
