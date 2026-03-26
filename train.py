import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION — Edit these paths if needed
# ─────────────────────────────────────────────
BASE_DIR = r'C:\Users\aditi\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset'

TRAIN_IMG_DIR  = os.path.join(BASE_DIR, 'train', 'Color_Images')
TRAIN_MASK_DIR = os.path.join(BASE_DIR, 'train', 'Segmentation')
VAL_IMG_DIR    = os.path.join(BASE_DIR, 'val',   'Color_Images')
VAL_MASK_DIR   = os.path.join(BASE_DIR, 'val',   'Segmentation')

# Training hyperparameters
IMAGE_SIZE  = 256        # Reduced for faster training
BATCH_SIZE  = 8          # Increased for efficiency
NUM_EPOCHS  = 5          # Reduced for faster training
LR          = 1e-4
NUM_CLASSES = None       # Auto-detected from masks
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH   = 'best_model.pth'

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# STEP 1: Auto-detect number of classes
# ─────────────────────────────────────────────
def detect_classes(mask_dir, sample_count=20):
    unique_vals = set()
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for fname in files[:sample_count]:
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_vals.update(np.unique(mask).tolist())
    print(f"Unique pixel values found in masks: {sorted(unique_vals)}")
    return len(unique_vals), sorted(unique_vals)

num_classes, class_values = detect_classes(TRAIN_MASK_DIR)
print(f"Detected {num_classes} classes: {class_values}")

# Map raw pixel values to class indices 0,1,2,...
val_to_idx = {v: i for i, v in enumerate(class_values)}

# ─────────────────────────────────────────────
# STEP 2: Dataset
# ─────────────────────────────────────────────
class OffroadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]

        img  = cv2.imread(os.path.join(self.img_dir, fname))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Try matching mask filename
        mask_path = os.path.join(self.mask_dir, fname)
        if not os.path.exists(mask_path):
            # Try without extension
            base = os.path.splitext(fname)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.mask_dir, base + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Map pixel values to class indices
        mapped = np.zeros_like(mask, dtype=np.int64)
        for val, idx_cls in val_to_idx.items():
            mapped[mask == val] = idx_cls

        if self.transform:
            augmented = self.transform(image=img, mask=mapped)
            img  = augmented['image']
            mask = augmented['mask'].long()
        else:
            img  = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            mask = torch.tensor(mapped).long()

        return img, mask

# ─────────────────────────────────────────────
# STEP 3: Augmentations
# ─────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# STEP 4: DataLoaders
# ─────────────────────────────────────────────
train_dataset = OffroadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
val_dataset   = OffroadDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ─────────────────────────────────────────────
# STEP 5: Model — UNet with ResNet34 backbone
# ─────────────────────────────────────────────
model = smp.Unet(
    encoder_name    = "mobilenet_v2",  # Lighter and faster than resnet34
    encoder_weights = "imagenet",
    in_channels     = 3,
    classes         = num_classes,
)
model = model.to(DEVICE)

# ─────────────────────────────────────────────
# STEP 6: Loss, Optimizer, Scheduler
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# ─────────────────────────────────────────────
# STEP 7: Training Loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validating"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == masks).sum().item()
            total   += masks.numel()
    acc = 100 * correct / total
    return total_loss / len(loader), acc

# ─────────────────────────────────────────────
# STEP 8: Run Training
# ─────────────────────────────────────────────
train_losses, val_losses = [], []
best_val_loss = float('inf')

print("\n Starting Training...\n")
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'num_classes': num_classes,
            'class_values': class_values,
        }, SAVE_PATH)
        print(f"  ✅ Best model saved! (val_loss={val_loss:.4f})")

# ─────────────────────────────────────────────
# STEP 9: Plot Loss Curves
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()
print("\nTraining complete! Model saved to:", SAVE_PATH)
print("Loss curve saved to: loss_curve.png")
