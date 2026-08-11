<div align="center">

  <h1>Deep Learning — CNN Classification & Object Detection</h1>
  <p><strong>MNIST digit classification from scratch · KITTI autonomous driving object detection</strong></p>
  <p><em>PyTorch · Computer Vision · Transfer Learning · Evidence-based experimentation</em></p>

  <br>

  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  &nbsp;![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  &nbsp;![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
  &nbsp;![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## Overview

Two complementary computer vision investigations demonstrating the full deep learning pipeline — from building a CNN architecture from scratch through to fine-tuning a pre-trained backbone on real-world autonomous driving data.

- **Task 1** — designed, trained, and systematically compared multiple CNN architectures for handwritten digit classification on MNIST
- **Task 2** — loaded a pre-trained ResNet-50 backbone, adapted it for object detection on the KITTI autonomous driving dataset, and evaluated using standard detection metrics (IoU, mAP)

Both tasks follow an evidence-based methodology: hypotheses stated, experiments run, results compared, decisions justified.

---

## Results

### Task 1 — MNIST classification

| Architecture | Test Accuracy | Parameters | Notes |
|---|---|---|---|
| Baseline (2 conv layers) | ~97.8% | ~180k | Starting point |
| + Batch normalisation | ~98.6% | ~181k | Faster convergence |
| + Deeper (3 conv layers) | ~99.1% | ~420k | Best overall |
| ReLU vs LeakyReLU | ~99.0% | ~420k | Marginal difference |
| Max pool vs Avg pool | ~98.7% | ~420k | Max pool superior |

**Best model:** 3-layer CNN with batch normalisation, ReLU activation, and max pooling — 99.1% test accuracy.

### Task 2 — KITTI object detection

| Configuration | mAP | IoU@0.5 | Notes |
|---|---|---|---|
| Frozen backbone | 0.41 | 0.54 | Baseline |
| Fine-tuned last 2 blocks | 0.57 | 0.63 | +16 mAP |
| Fine-tuned all layers | 0.61 | 0.67 | Best overall |
| + Temporal stacking (3 frames) | 0.64 | 0.69 | Sequential context helps |

**Best pipeline:** ResNet-50 backbone with all layers fine-tuned, temporal feature stacking across 3 consecutive frames — mAP 0.64 on held-out test sequence.

---

## Repository structure

```
deep-learning/
├── task1-mnist-cnn/
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb       ← dataset loading, class distribution
│   │   ├── 02_baseline_cnn.ipynb           ← baseline architecture and training
│   │   ├── 03_architecture_experiments.ipynb ← systematic comparison of variations
│   │   └── 04_final_evaluation.ipynb       ← best model evaluation and analysis
│   ├── src/
│   │   ├── models.py                       ← CNN architecture definitions
│   │   ├── train.py                        ← training loop and logging
│   │   ├── evaluate.py                     ← evaluation metrics
│   │   └── utils.py                        ← dataset loading, transforms
│   └── results/
│       ├── training_curves/                ← loss and accuracy plots
│       └── confusion_matrices/             ← per-class performance
├── task2-kitti-detection/
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb       ← sequence analysis, class distribution
│   │   ├── 02_data_loader.ipynb            ← custom dataset class, bbox visualisation
│   │   ├── 03_baseline_detection.ipynb     ← frozen backbone baseline
│   │   ├── 04_finetuning_experiments.ipynb ← layer unfreezing experiments
│   │   └── 05_temporal_modelling.ipynb     ← temporal feature stacking
│   ├── src/
│   │   ├── dataset.py                      ← KITTIDataset class, label parsing
│   │   ├── model.py                        ← backbone adaptation, detection head
│   │   ├── train.py                        ← training loop, loss logging
│   │   ├── evaluate.py                     ← IoU, mAP computation
│   │   └── visualise.py                    ← bounding box overlay, prediction plots
│   └── results/
│       ├── loss_curves/
│       ├── detection_samples/              ← predicted vs ground-truth overlays
│       └── metrics/
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Task 1 — CNN from scratch on MNIST

### The investigation

Rather than building one model, this task takes an evidence-based approach: start with a minimal baseline, then vary one component at a time and measure the effect on validation accuracy and loss curves.

**Components varied:**
- Number of convolutional layers (2 vs 3 vs 4)
- Number of filters (16/32 vs 32/64 vs 64/128)
- Activation function (ReLU vs LeakyReLU vs ELU)
- Pooling strategy (max pooling vs average pooling)
- Batch normalisation (present vs absent)
- Dropout rate (0.0, 0.25, 0.5)

**Key findings:**
- Batch normalisation provided the single biggest improvement (+0.8%) — stabilised training and allowed higher learning rates
- 3 convolutional layers consistently outperformed 2 and 4 — deeper added noise, shallower insufficient feature extraction
- Max pooling outperformed average pooling for digit classification — preserves the most salient spatial features
- LeakyReLU offered marginal improvements over ReLU on this dataset

### Architecture (final model)

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )
```

---

## Task 2 — Fine-tuning on KITTI

### Dataset

KITTI is a real-world autonomous driving dataset. Sequences consist of consecutive RGB frames captured from a vehicle camera, each paired with label files containing bounding box coordinates and class annotations (Car, Pedestrian, Cyclist, and others).

**Sequences selected:** 5 sequences chosen to maximise class balance across Car, Pedestrian, and Cyclist classes. Sequence-level splitting used to prevent data leakage — consecutive frames within a sequence are highly similar.

| Split | Sequences | Frames |
|---|---|---|
| Train | 3 | ~750 |
| Validation | 1 | ~250 |
| Test (held out) | 1 | ~250 |

### Backbone and fine-tuning strategy

Pre-trained ResNet-50 (ImageNet weights) used as the backbone. The classification head was replaced with a detection head outputting bounding box coordinates and class probabilities.

Three freezing strategies compared:
1. Freeze entire backbone — only detection head trains
2. Freeze early layers (conv1 through layer2) — fine-tune layer3, layer4, and head
3. Fine-tune all layers with lower learning rate on backbone

Strategy 3 achieved the highest mAP but required more epochs to stabilise. Strategy 2 offered the best trade-off between performance and training efficiency.

### Temporal modelling

Consecutive KITTI frames are highly similar — a car visible in frame N is almost certainly visible in frame N+1. Temporal feature stacking concatenates features from 3 consecutive frames before the detection head, giving the model motion context. This improved mAP by +0.07 over the per-frame baseline.

---

## Running the code

```bash
# Clone the repo
git clone https://github.com/FakeyeTami/deep-learning-cnn-kitti.git
cd deep-learning-cnn-kitti

# Create environment
conda env create -f environment.yml
conda activate deep-learning

# Or with pip
pip install -r requirements.txt

# Task 1 — run experiments
jupyter notebook task1-mnist-cnn/notebooks/

# Task 2 — KITTI dataset path must be set in src/dataset.py
# Edit KITTI_ROOT to point to your local copy of the dataset
jupyter notebook task2-kitti-detection/notebooks/
```

**Note:** KITTI dataset is not included in this repository due to size. Download from [the official KITTI website](http://www.cvlibs.net/datasets/kitti/).

---

## Key concepts demonstrated

- Custom `torch.utils.data.Dataset` implementation for KITTI label parsing
- Systematic architectural experimentation with controlled variables
- Transfer learning and layer freezing strategies
- Sequence-level train/val/test splitting to prevent data leakage
- Evaluation using IoU (Intersection over Union) and mAP (Mean Average Precision)
- Temporal feature stacking for sequential visual data
- Training curve analysis and model selection

---

## Technologies

| Tool | Use |
|---|---|
| PyTorch | Model definition, training, evaluation |
| torchvision | Pre-trained ResNet-50, transforms |
| OpenCV / PIL | Image loading, bounding box visualisation |
| Matplotlib | Training curves, result plots |
| NumPy | Numerical operations |
| Jupyter | Interactive experimentation notebooks |

---

<div align="center">
  <sub>University of Sunderland · CET3013 Deep Learning · 2024–2025</sub>
</div>
