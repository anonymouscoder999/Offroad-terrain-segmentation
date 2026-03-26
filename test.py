import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH    = 'best_model.pth'
TEST_IMG_DIR  = r'C:\Users\aditi\Downloads\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images'
OUTPUT_DIR    = 'test_predictions'
IMAGE_SIZE    = 512
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
print("Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes  = checkpoint['num_classes']
class_values = checkpoint['class_values']

model = smp.Unet(
    encoder_name    = "mobilenet_v2",
    encoder_weights = None,
    in_channels     = 3,
    classes         = num_classes,
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print(f"Model loaded! Classes: {num_classes}, Best val_loss: {checkpoint['val_loss']:.4f}")

# ─────────────────────────────────────────────
# Color map for visualization
# ─────────────────────────────────────────────
COLORS = [
    [0,   0,   0  ],  # Class 0 — Background/Sky (black)
    [128, 64,  128],  # Class 1 — Terrain (purple)
    [0,   200, 0  ],  # Class 2 — Vegetation (green)
    [255, 165, 0  ],  # Class 3 — Rocks/Obstacles (orange)
    [0,   0,   255],  # Class 4 — Water (blue)
    [255, 255, 0  ],  # Class 5+ — Other (yellow)
]
LABELS = ['Sky/Background', 'Terrain', 'Vegetation', 'Rocks/Obstacles', 'Water', 'Other']

def class_to_color(pred_mask, num_classes):
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color = COLORS[cls] if cls < len(COLORS) else [200, 200, 200]
        color_mask[pred_mask == cls] = color
    return color_mask

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# Run Inference
# ─────────────────────────────────────────────
test_images = [f for f in os.listdir(TEST_IMG_DIR)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"\nRunning inference on {len(test_images)} images...\n")

for fname in test_images:
    img_path = os.path.join(TEST_IMG_DIR, fname)
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_img.shape[:2]

    # Preprocess
    aug = transform(image=orig_img)
    tensor = aug['image'].unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()

    # Resize back to original size
    pred_resized = cv2.resize(pred.astype(np.uint8),
                              (orig_w, orig_h),
                              interpolation=cv2.INTER_NEAREST)

    # Colorize
    color_pred = class_to_color(pred_resized, num_classes)

    # Overlay on original image (50% blend)
    overlay = cv2.addWeighted(orig_img, 0.6, color_pred, 0.4, 0)

    # Save and display
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig_img);    axes[0].set_title('Original Image');    axes[0].axis('off')
    axes[1].imshow(color_pred);  axes[1].set_title('Segmentation Mask'); axes[1].axis('off')
    axes[2].imshow(overlay);     axes[2].set_title('Overlay');           axes[2].axis('off')

    # Legend
    patches = [mpatches.Patch(color=np.array(COLORS[i])/255,
                               label=LABELS[i]) for i in range(min(num_classes, len(COLORS)))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle(fname, fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, fname.replace('.jpg', '_pred.png').replace('.jpeg', '_pred.png'))
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

print(f"\nAll predictions saved to: {OUTPUT_DIR}/")
