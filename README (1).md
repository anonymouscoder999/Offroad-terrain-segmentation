# NexusPrime — Offroad Terrain Semantic Segmentation
### GHR 2.0 Hackathon Submission

---

## Project Overview
A deep learning model for semantic segmentation of offroad terrain images. Built using UNet with MobileNetV2 encoder, trained on synthetic game-engine rendered desert terrain data.

---

## Results
| Metric | Score |
|--------|-------|
| Pixel Accuracy | 84.58% |
| Mean IoU (mIoU) | 46.04% |
| Validation Loss | 0.4049 |
| Training Samples | 2,857 images |
| Classes Detected | 6 terrain types |

---

## Classes
| Class | Terrain Type |
|-------|-------------|
| 0 | Sky / Background |
| 1 | Ground Terrain |
| 2 | Vegetation / Shrubs |
| 3 | Rocks / Obstacles |
| 4 | Water / Flat Surfaces |
| 5 | Other / Mixed Terrain |

---

## Model Architecture
- **Architecture:** UNet
- **Encoder:** MobileNetV2 (pretrained on ImageNet)
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** AdamW (lr=1e-4)
- **Input Size:** 256x256
- **Epochs:** 5

---

## Project Structure
```
NexusPrime/
  train.py         # Training pipeline
  test.py          # Inference + visualization
  evaluate.py      # IoU and accuracy evaluation
  loss_curve.png   # Training loss graph
  README.md        # This file
```

---

## How to Run

### Install dependencies
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy matplotlib tqdm segmentation-models-pytorch albumentations==1.3.1
```

### Train
```
python train.py
```

### Test
```
python test.py
```

### Evaluate (IoU + Accuracy)
```
python evaluate.py
```

---

## Dataset
- **Source:** Offroad Segmentation Training Dataset (GHR 2.0)
- **Train:** 2,857 images with segmentation masks
- **Val:** 317 images with segmentation masks
- **Format:** Color images (RGB) + Grayscale segmentation masks

##Weights are available on request.

---

## Team
**Team Name:** NexusPrime  
**Hackathon:** GHR 2.0 — February 2026
