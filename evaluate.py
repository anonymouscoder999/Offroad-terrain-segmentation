import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH   = 'best_model.pth'
VAL_IMG_DIR  = r'C:\Users\aditi\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Color_Images'
VAL_MASK_DIR = r'C:\Users\aditi\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Segmentation'
IMAGE_SIZE   = 256
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
print("Loading model...")
checkpoint   = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes  = checkpoint['num_classes']
class_values = checkpoint['class_values']
val_to_idx   = {v: i for i, v in enumerate(class_values)}

model = smp.Unet(
    encoder_name    = "mobilenet_v2",
    encoder_weights = None,
    in_channels     = 3,
    classes         = num_classes,
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print(f"Model loaded! Classes: {num_classes} | Values: {class_values}\n")

# ─────────────────────────────────────────────
# Transform
# ─────────────────────────────────────────────
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
images = sorted([f for f in os.listdir(VAL_IMG_DIR)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Tracking metrics
total_correct = 0
total_pixels  = 0
iou_per_class = [[] for _ in range(num_classes)]

print(f"Evaluating on {len(images)} validation images...\n")

for fname in tqdm(images):
    # Load image
    img = cv2.imread(os.path.join(VAL_IMG_DIR, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load mask
    mask_path = os.path.join(VAL_MASK_DIR, fname)
    if not os.path.exists(mask_path):
        base = os.path.splitext(fname)[0]
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(VAL_MASK_DIR, base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    # Map mask values to class indices
    mapped = np.zeros_like(mask, dtype=np.int64)
    for val, idx in val_to_idx.items():
        mapped[mask == val] = idx

    # Preprocess image
    aug    = transform(image=img)
    tensor = aug['image'].unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        pred   = output.argmax(dim=1).squeeze().cpu().numpy()

    # Pixel Accuracy
    total_correct += (pred == mapped).sum()
    total_pixels  += mapped.size

    # IoU per class
    for cls in range(num_classes):
        pred_cls   = (pred == cls)
        target_cls = (mapped == cls)
        intersection = (pred_cls & target_cls).sum()
        union        = (pred_cls | target_cls).sum()
        if union > 0:
            iou_per_class[cls].append(intersection / union)

# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────
pixel_accuracy = 100 * total_correct / total_pixels
mean_iou       = np.mean([np.mean(iou) for iou in iou_per_class if len(iou) > 0])

CLASS_NAMES = ['Sky/Background', 'Terrain', 'Vegetation', 'Rocks/Obstacles', 'Water', 'Other']

print("\n" + "="*50)
print("        EVALUATION RESULTS")
print("="*50)
print(f"  Pixel Accuracy : {pixel_accuracy:.2f}%")
print(f"  Mean IoU (mIoU): {mean_iou*100:.2f}%")
print("-"*50)
print("  Per-Class IoU:")
for i, iou_list in enumerate(iou_per_class):
    if iou_list:
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
        print(f"    {name:<20}: {np.mean(iou_list)*100:.2f}%")
print("="*50)