# Mask R-CNN Parameters Comparison

## Comparison: vitmaskrcnn vs vitfasterrcnn

This document compares Mask R-CNN related parameters between:
- **vitmaskrcnn (224)**: `/home/m341664/yanglanou/projects/vitmaskrcnn/dinov2_maskrcnn_improved.py` (224×224 cropped)
- **vitmaskrcnn (518)**: `/home/m341664/yanglanou/projects/vitmaskrcnn/dinov2_maskrcnn_improved_resize518.py` (518×518 resized)
- **vitfasterrcnn**: `/home/m341664/yanglanou/projects/vitfasterrcnn/dinov2_fasterrcnn_improved.py` (518×518 resized)

---

## 1. Anchor Generator Parameters

| Parameter | vitmaskrcnn (224) | vitmaskrcnn (518) | vitfasterrcnn | Difference |
|-----------|-------------------|-------------------|--------------|------------|
| **Anchor Sizes** | `(4, 6, 8, 10, 12, 16, 20, 24, 32)` | `(4, 6, 8, 10, 12, 16, 20, 24, 32)` | `(8, 12, 16, 20, 24, 30, 36, 42, 48, 60)` | **Mask R-CNN uses smaller anchors** (4-32) vs Faster R-CNN (8-60) |
| **Aspect Ratios** | `(0.5, 0.75, 1.0, 1.25, 1.5, 2.0)` | `(0.5, 0.75, 1.0, 1.25, 1.5, 2.0)` | `(0.5, 0.75, 1.0, 1.25, 1.5)` | **Mask R-CNN has 6 ratios** (includes 2.0), Faster R-CNN has 5 ratios |
| **Number of Anchors** | 9 sizes × 6 ratios = **54 anchors** | 9 sizes × 6 ratios = **54 anchors** | 10 sizes × 5 ratios = **50 anchors** | Mask R-CNN generates more anchor variations |

**Analysis**: 
- Both Mask R-CNN variants use the same anchor configuration (smaller anchors for nuclei detection)
- Faster R-CNN uses larger anchors (for 518×518 images)
- Mask R-CNN uses more aspect ratios including 2.0

---

## 2. RPN (Region Proposal Network) Parameters

| Parameter | vitmaskrcnn (224) | vitmaskrcnn (518) | vitfasterrcnn | Difference |
|-----------|-------------------|-------------------|--------------|------------|
| `rpn_pre_nms_top_n_train` | **4000** | **4000** | **4000** | ✅ All same |
| `rpn_pre_nms_top_n_test` | **3000** | **3000** | **3000** | ✅ All same |
| `rpn_post_nms_top_n_train` | **2000** | **2000** | **3000** | ⚠️ **Mask R-CNN: 2000, Faster R-CNN: 3000** |
| `rpn_post_nms_top_n_test` | **1500** | **1500** | **2000** | ⚠️ **Mask R-CNN: 1500, Faster R-CNN: 2000** |
| `rpn_nms_thresh` | **0.7** | **0.7** | **0.7** | ✅ All same |
| `rpn_fg_iou_thresh` | **0.5** | **0.5** | **0.5** | ✅ All same |
| `rpn_bg_iou_thresh` | **0.3** | **0.3** | **0.3** | ✅ All same |
| `rpn_batch_size_per_image` | **256** | **256** | **256** | ✅ All same |
| `rpn_positive_fraction` | **0.5** | **0.5** | **0.5** | ✅ All same |

**Analysis**: Both Mask R-CNN variants keep fewer proposals after NMS (2000 vs 3000 in training, 1500 vs 2000 in testing) compared to Faster R-CNN.

---

## 3. Box/Detection Head Parameters

| Parameter | vitmaskrcnn (224) | vitmaskrcnn (518) | vitfasterrcnn | Difference |
|-----------|-------------------|-------------------|--------------|------------|
| `box_score_thresh` | **0.01** | **0.01** | **0.01** | ✅ All same |
| `box_nms_thresh` | **0.3** | **0.3** | **0.4** | ⚠️ **Mask R-CNN: 0.3 (stricter), Faster R-CNN: 0.4 (more lenient)** |
| `box_detections_per_img` | **500** | **500** | **1000** | ⚠️ **Mask R-CNN: 500, Faster R-CNN: 1000** |
| `box_fg_iou_thresh` | **0.5** | **0.5** | **0.5** | ✅ All same |
| `box_bg_iou_thresh` | **0.5** | **0.5** | **0.5** | ✅ All same |
| `box_batch_size_per_image` | **256** | **256** | **256** | ✅ All same |
| `box_positive_fraction` | **0.25** | **0.25** | **0.25** | ✅ All same |

**Analysis**: 
- Both Mask R-CNN variants use stricter NMS (0.3 vs 0.4) to reduce overlapping detections
- Both Mask R-CNN variants allow fewer detections per image (500 vs 1000)

---

## 4. Image Size Parameters

| Parameter | vitmaskrcnn (224) | vitmaskrcnn (518) | vitfasterrcnn | Difference |
|-----------|-------------------|-------------------|--------------|------------|
| `min_size` | **target_size** (default: 224) | **896** | **896** | ⚠️ **224 variant: 224, 518 variant: 896 (same as Faster R-CNN)** |
| `max_size` | **target_size** (default: 224) | **1024** | **1024** | ⚠️ **224 variant: 224, 518 variant: 1024 (same as Faster R-CNN)** |
| **Image Processing** | Cropped to fixed size | Resized to 518×518 | Resized to 518×518 | **224 variant uses cropping, 518 variant matches Faster R-CNN**

**Analysis**: 
- Mask R-CNN (224) works with small cropped images (224×224)
- Mask R-CNN (518) and Faster R-CNN both work with larger resized images (518×518, with min/max size 896-1024)

---

## 5. ROI Pooling Parameters

| Parameter | vitmaskrcnn (224) | vitmaskrcnn (518) | vitfasterrcnn | Difference |
|-----------|-------------------|-------------------|--------------|------------|
| **Box ROI Pooler** | | | | |
| - `featmap_names` | `['0']` | `['0']` | `['0']` | ✅ All same |
| - `output_size` | **7** | **7** | **7** | ✅ All same |
| - `sampling_ratio` | **2** | **2** | **2** | ✅ All same |
| **Mask ROI Pooler** | | | | |
| - `featmap_names` | `['0']` | `['0']` | N/A | ⚠️ **Mask R-CNN only** |
| - `output_size` | **14** | **14** | N/A | ⚠️ **Mask R-CNN only** |
| - `sampling_ratio` | **2** | **2** | N/A | ⚠️ **Mask R-CNN only** |

**Analysis**: 
- All use the same box ROI pooling (7×7 output)
- Both Mask R-CNN variants have an additional mask ROI pooler (14×14 output for mask prediction)

---

## 6. Backbone Parameters

| Parameter | vitmaskrcnn (224) | vitmaskrcnn (518) | vitfasterrcnn | Difference |
|-----------|-------------------|-------------------|--------------|------------|
| **Backbone Architecture** | `DINOv2BackboneSimple` | `DINOv2BackboneSimple` | `DINOv2BackboneSimple` | ✅ All same |
| **Embedding Dimension** | **1536** (ViT-G) | **1536** (ViT-G) | **1536** (ViT-G) | ✅ All same |
| **Projection Channels** | **256** | **256** | **256** | ✅ All same |
| **Feature Map Upsampling** | 64×64 | 64×64 | 64×64 | ✅ All same |
| **Freeze Backbone** | Configurable | Configurable | Configurable | ✅ All same |

**Analysis**: All three variants use identical backbone architectures.

---

## 7. Model-Specific Parameters

### Mask R-CNN Only (not in Faster R-CNN):
- **Mask ROI Pooler**: 14×14 output size for mask prediction
- **Mask Head**: Additional mask prediction branch
- **Mask Loss**: `loss_mask` component in training

### Faster R-CNN Only:
- No mask-related components

---

## Summary of Key Differences

### 1. **Anchor Sizes**
- **Mask R-CNN (both variants)**: Smaller anchors (4-32) - same for both 224 and 518 variants
- **Faster R-CNN**: Larger anchors (8-60) for larger resized images

### 2. **Aspect Ratios**
- **Mask R-CNN (both variants)**: 6 ratios (includes 2.0) - same for both variants
- **Faster R-CNN**: 5 ratios

### 3. **RPN Post-NMS Proposals**
- **Mask R-CNN (both variants)**: Fewer proposals (2000 train, 1500 test) - same for both variants
- **Faster R-CNN**: More proposals (3000 train, 2000 test)

### 4. **Box NMS Threshold**
- **Mask R-CNN (both variants)**: Stricter (0.3) - same for both variants
- **Faster R-CNN**: More lenient (0.4)

### 5. **Detections Per Image**
- **Mask R-CNN (both variants)**: 500 - same for both variants
- **Faster R-CNN**: 1000

### 6. **Image Size**
- **Mask R-CNN (224)**: 224×224 (cropped, min/max_size=224)
- **Mask R-CNN (518)**: 518×518 (resized, min/max_size=896/1024) - **matches Faster R-CNN**
- **Faster R-CNN**: 518×518 (resized, min/max_size=896/1024)

### 7. **Mask Components**
- **Mask R-CNN (both variants)**: Has mask ROI pooler and mask prediction head
- **Faster R-CNN**: No mask components

### Key Insight
**Both Mask R-CNN variants use identical detection parameters** (anchors, RPN, box head), regardless of image size. The only difference is the image preprocessing strategy (cropping vs resizing) and the min/max_size parameters. The 518 variant's image size parameters match Faster R-CNN, but all other detection parameters remain optimized for smaller objects.

---

## Recommendations

1. **Consider aligning anchor sizes**: Mask R-CNN uses smaller anchors (4-32) even for 518×518 images, while Faster R-CNN uses larger anchors (8-60). If using 518×518 images, consider using larger anchors for Mask R-CNN.

2. **Consider aligning RPN post-NMS counts**: Mask R-CNN keeps fewer proposals (2000/1500) compared to Faster R-CNN (3000/2000). For 518×518 images, consider increasing Mask R-CNN's post-NMS counts.

3. **Consider aligning box NMS threshold**: Mask R-CNN uses stricter NMS (0.3) vs Faster R-CNN (0.4). Consider matching if you want similar detection behavior.

4. **Consider aligning detections per image**: Mask R-CNN allows 500 detections vs Faster R-CNN's 1000. For 518×518 images with more objects, consider increasing this.

5. **Note**: The Mask R-CNN (518) variant uses the same detection parameters as the 224 variant, despite processing larger images. This may be suboptimal - consider scaling parameters with image size.

---

## Files Compared

- **vitmaskrcnn (224)**: `dinov2_maskrcnn_improved.py` (lines 97-147)
- **vitmaskrcnn (518)**: `dinov2_maskrcnn_improved_resize518.py` (lines 92-140)
- **vitfasterrcnn**: `dinov2_fasterrcnn_improved.py` (lines 88-129)

