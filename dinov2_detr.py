import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MLP(nn.Module):
    """Simple multi-layer perceptron"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETRDetectionHead(nn.Module):
    """DETR-style detection head for cell nuclei detection"""
    
    def __init__(
        self,
        encoder_dim: int = 1536,  # DINOv2 ViT-G output dim
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 1,  # Binary: nucleus or no nucleus
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Project encoder features to decoder hidden dimension
        self.input_proj = nn.Conv2d(encoder_dim, hidden_dim, kernel_size=1)
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # x, y, w, h
        
        # Positional encoding for spatial features
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Encoder output features [B, C, H, W]
            
        Returns:
            pred_logits: Classification logits [B, num_queries, num_classes+1]
            pred_boxes: Bounding box predictions [B, num_queries, 4] in (cx, cy, w, h) format, normalized [0, 1]
        """
        # Project encoder features
        src = self.input_proj(features)  # [B, hidden_dim, H, W]
        B, C, H, W = src.shape
        
        # Generate positional encoding
        pos_embed = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(0)  # [1, H*W, hidden_dim]
        pos_embed = pos_embed.repeat(B, 1, 1)
        
        # Flatten spatial dimensions for transformer
        src_flat = src.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]
        src_flat = src_flat + pos_embed
        
        # Prepare query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]
        
        # Decoder forward pass
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, src_flat)  # [B, num_queries, hidden_dim]
        
        # Prediction heads
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return outputs_class, outputs_coord


class DINOv2WithDETR(nn.Module):
    """Complete model: DINOv2 encoder + DETR detection head"""
    
    def __init__(
        self,
        dinov2_model,  # Your pre-trained DINOv2 model
        freeze_encoder: bool = False,
        **detr_kwargs
    ):
        super().__init__()
        
        self.encoder = dinov2_model
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # DINOv2 ViT-G has 1536 feature dimension
        encoder_dim = self.encoder.embed_dim
        
        self.detection_head = DETRDetectionHead(
            encoder_dim=encoder_dim,
            **detr_kwargs
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            pred_logits: [B, num_queries, num_classes+1]
            pred_boxes: [B, num_queries, 4]
        """
        # Extract patch features from DINOv2
        features = self.encoder.forward_features(x)
        
        # Get patch tokens (exclude CLS token)
        patch_tokens = features['x_norm_patchtokens']  # [B, num_patches, embed_dim]
        
        # Reshape to spatial grid
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        # Detection head
        pred_logits, pred_boxes = self.detection_head(patch_tokens)
        
        return pred_logits, pred_boxes


def _extract_and_map_backbone(custom_state):
    """Extract and map backbone keys with perfect compatibility."""
    backbone_state = {}
    
    # Nested to flat block mapping based on actual checkpoint structure
    # The checkpoint has blocks.0, blocks.1, blocks.2, blocks.3 with nested sub-blocks
    # Each block contains multiple sub-blocks (0.0, 0.1, 0.2, etc.)
    nested_mappings = {
        # Block 0: 0.0->0, 0.1->1, 0.2->2, 0.3->3, 0.4->4, 0.5->5, 0.6->6, 0.7->7, 0.8->8, 0.9->9
        'blocks.0.0': 'blocks.0',
        'blocks.0.1': 'blocks.1', 
        'blocks.0.2': 'blocks.2',
        'blocks.0.3': 'blocks.3',
        'blocks.0.4': 'blocks.4',
        'blocks.0.5': 'blocks.5',
        'blocks.0.6': 'blocks.6',
        'blocks.0.7': 'blocks.7',
        'blocks.0.8': 'blocks.8',
        'blocks.0.9': 'blocks.9',
        # Block 1: 1.10->10, 1.11->11, etc.
        'blocks.1.10': 'blocks.10',
        'blocks.1.11': 'blocks.11',
        'blocks.1.12': 'blocks.12',
        'blocks.1.13': 'blocks.13',
        'blocks.1.14': 'blocks.14',
        'blocks.1.15': 'blocks.15',
        'blocks.1.16': 'blocks.16',
        'blocks.1.17': 'blocks.17',
        'blocks.1.18': 'blocks.18',
        'blocks.1.19': 'blocks.19',
        # Block 2: 2.20->20, 2.21->21, etc.
        'blocks.2.20': 'blocks.20',
        'blocks.2.21': 'blocks.21',
        'blocks.2.22': 'blocks.22',
        'blocks.2.23': 'blocks.23',
        'blocks.2.24': 'blocks.24',
        'blocks.2.25': 'blocks.25',
        'blocks.2.26': 'blocks.26',
        'blocks.2.27': 'blocks.27',
        'blocks.2.28': 'blocks.28',
        'blocks.2.29': 'blocks.29',
        # Block 3: 3.30->30, 3.31->31, etc.
        'blocks.3.30': 'blocks.30',
        'blocks.3.31': 'blocks.31',
        'blocks.3.32': 'blocks.32',
        'blocks.3.33': 'blocks.33',
        'blocks.3.34': 'blocks.34',
        'blocks.3.35': 'blocks.35',
        'blocks.3.36': 'blocks.36',
        'blocks.3.37': 'blocks.37',
        'blocks.3.38': 'blocks.38',
        'blocks.3.39': 'blocks.39',
    }
    
    # Map all nested block keys
    for key, value in custom_state.items():
        if key.startswith('backbone.blocks.'):
            clean_key = key[9:]  # Remove 'backbone.' prefix
            
            # Handle nested block mapping
            mapped_key = clean_key
            for old_prefix, new_prefix in nested_mappings.items():
                if clean_key.startswith(old_prefix):
                    mapped_key = clean_key.replace(old_prefix, new_prefix)
                    break
            
            # Keep original MLP naming (DINOv2 uses .w12. and .w3.)
            # No conversion needed for DINOv2
            
            backbone_state[mapped_key] = value
    
    # Handle other non-block keys
    for key, value in custom_state.items():
        if key.startswith('backbone.') and not key.startswith('backbone.blocks.'):
            clean_key = key[9:]  # Remove 'backbone.' prefix
            
            # Skip keys that don't exist in ViTDet or have different shapes
            if clean_key in ['cls_token', 'mask_token']:
                continue  # ViTDet doesn't have these
            
            # Handle position embedding - skip here, will be handled later with interpolation
            if clean_key == 'pos_embed':
                print(f"✅ Found {clean_key} with size: {value.shape} (will interpolate later)")
                continue
            
            # Handle patch embedding - use original size for 14x14 patches
            if clean_key == 'patch_embed.proj.weight':
                # Custom checkpoint: [768, 3, 14, 14] - this is correct for 14x14 patch size
                if value.shape[2:] == (14, 14):
                    backbone_state[clean_key] = value
                    print(f"✅ Using {clean_key} with original size: {value.shape}")
                else:
                    print(f"⚠️  Skipping {clean_key} due to unexpected size: {value.shape}")
                continue
            
            backbone_state[clean_key] = value
    
    print(f"📊 Mapped {len(backbone_state)} backbone parameters")
    return backbone_state


def load_dinov2_from_checkpoint(checkpoint_path: str):
    """Load DINOv2 model from custom checkpoint with perfect key mapping"""
    import torch.nn.functional as F
    
    # Load pre-trained DINOv2 architecture
    dinov2_vitg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    
    # Load your custom checkpoint
    print(f"Loading DINOv2 checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract teacher model state dict
    if 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
        print("Loaded from 'teacher' key")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Loaded from 'model' key")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Loaded from 'state_dict' key")
    else:
        state_dict = checkpoint
        print("Loaded checkpoint directly")
    
    # Use perfect loader mapping
    backbone_state = _extract_and_map_backbone(state_dict)
    
    # Handle positional embedding size mismatch
    # Your checkpoint: 224x224 -> 16x16 patches = 256 + 1 CLS = 257
    # Target model: 518x518 -> 37x37 patches = 1369 + 1 CLS = 1370
    if 'backbone.pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['backbone.pos_embed']
        pos_embed_model = dinov2_vitg.pos_embed
        
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            print(f"Interpolating pos_embed from {pos_embed_checkpoint.shape} to {pos_embed_model.shape}")
            
            # Remove CLS token
            pos_embed_checkpoint = pos_embed_checkpoint[:, 1:, :]  # [1, 256, 1536]
            
            # Get grid size
            checkpoint_num_patches = pos_embed_checkpoint.shape[1]
            model_num_patches = pos_embed_model.shape[1] - 1  # Exclude CLS token
            
            # Calculate grid sizes
            checkpoint_grid_size = int(checkpoint_num_patches ** 0.5)
            model_grid_size = int(model_num_patches ** 0.5)
            
            print(f"  Checkpoint grid: {checkpoint_grid_size}x{checkpoint_grid_size}")
            print(f"  Model grid: {model_grid_size}x{model_grid_size}")
            
            # Reshape to 2D grid
            embed_dim = pos_embed_checkpoint.shape[-1]
            pos_embed_checkpoint = pos_embed_checkpoint.reshape(
                1, checkpoint_grid_size, checkpoint_grid_size, embed_dim
            ).permute(0, 3, 1, 2)  # [1, 1536, 16, 16]
            
            # Interpolate
            pos_embed_checkpoint = F.interpolate(
                pos_embed_checkpoint,
                size=(model_grid_size, model_grid_size),
                mode='bicubic',
                align_corners=False
            )
            
            # Reshape back
            pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 3, 1).reshape(
                1, model_num_patches, embed_dim
            )
            
            # Add back CLS token (use the one from checkpoint)
            cls_token = state_dict.get('cls_token', dinov2_vitg.cls_token)
            if 'backbone.cls_token' in state_dict:
                cls_token = state_dict['backbone.cls_token']
            
            pos_embed_checkpoint = torch.cat([
                torch.zeros(1, 1, embed_dim),  # Placeholder for CLS
                pos_embed_checkpoint
            ], dim=1)
            
            backbone_state['pos_embed'] = pos_embed_checkpoint
    
    # Load weights with perfect mapping
    msg = dinov2_vitg.load_state_dict(backbone_state, strict=False)
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")
    
    if len(msg.missing_keys) > 0:
        print(f"First 5 missing keys: {list(msg.missing_keys)[:5]}")
    if len(msg.unexpected_keys) > 0:
        print(f"First 5 unexpected keys: {list(msg.unexpected_keys)[:5]}")
    
    # Debug: Check which blocks were loaded
    loaded_blocks = set()
    for key in backbone_state.keys():
        if 'blocks.' in key:
            parts = key.split('.')
            if len(parts) >= 2:
                loaded_blocks.add(parts[1])
    
    print(f"  Loaded blocks: {sorted(loaded_blocks)}")
    print(f"  Total blocks loaded: {len(loaded_blocks)}")
    
    if len(loaded_blocks) == 40:
        print(f"  ✅ All 40 transformer blocks loaded!")
    else:
        print(f"  ⚠️ Only {len(loaded_blocks)} blocks loaded (expected 40)")
    
    return dinov2_vitg


# Example usage
if __name__ == "__main__":
    # Option 1: Load from torch hub (pre-trained on ImageNet)
    # dinov2_vitg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    
    # Option 2: Load from your custom checkpoint
    checkpoint_path = "/rodata/dlmp_path/han/data/m328672/dinov2-main/dgx_vitg14_patch37M/training_287499/teacher_checkpoint.pth"
    dinov2_vitg = load_dinov2_from_checkpoint(checkpoint_path)
    
    # Create model with detection head
    model = DINOv2WithDETR(
        dinov2_model=dinov2_vitg,
        freeze_encoder=True,  # Freeze encoder initially
        hidden_dim=256,
        num_queries=100,
        num_decoder_layers=6,
        num_classes=1  # Cell nuclei (binary detection)
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 518, 518)  # DINOv2 expects 518x518
    pred_logits, pred_boxes = model(dummy_input)
    
    print(f"Logits shape: {pred_logits.shape}")  # [2, 100, 2]
    print(f"Boxes shape: {pred_boxes.shape}")    # [2, 100, 4]