# Maize Nitrogen Deficiency Classification

COMP9444 project 008: classify maize RGB images into three nitrogen-treatment
levels: `N0`, `N75`, and `NFull`.

## Final results

The held-out test set contains 180 images (60 per class). Plot IDs are disjoint
between training, validation, and test sets.

| Model | Test accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Simple CNN | 60.56% | 58.38% | 60.56% | 57.84% |
| ResNet18 | **63.89%** | **61.56%** | **63.89%** | **62.17%** |

The main remaining error is the intermediate `N75` class. ResNet18 correctly
classified 20/60 `N75` images; 19 were predicted as `N0` and 21 as `NFull`.

## Project layout

```text
project/
├── Images/                 # 1,200 extracted JPG images
├── archives/               # Original dataset archive
├── data/
│   └── split.csv           # Leakage-free train/val/test manifest
├── models/                 # Best checkpoints and training histories
│   ├── simple_cnn/
│   ├── resnet18_stage1/
│   └── resnet18/
├── references/             # Original PDF, notebook, and dependencies
├── results/
│   ├── simple_cnn/         # Metrics and per-image predictions
│   ├── resnet18/           # Metrics and per-image predictions
│   ├── figures/            # Report-ready comparison figures
│   └── gradcam/            # Grad-CAM explanation figures
├── src/                    # Reproducible PyTorch pipeline
├── requirements.txt        # Dependencies used by this implementation
└── README.md
```

## Environment

The existing virtual environment is one directory above this project:

```bash
cd /Users/hezhu/Desktop/9444/project
source ../.venv/bin/activate
```

For a new environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the pipeline

Run commands from the `project` directory after activating the environment.

### 1. Validate and split the dataset

```bash
python src/prepare_data.py
```

The script groups images by the numeric plot ID in each filename. All three
images for a plot (`N0`, `N75`, and `NFull`) remain in the same split, preventing
plot/genotype leakage.

Split sizes:

| Split | Plot IDs | Images | Images per class |
|---|---:|---:|---:|
| Train | 280 | 840 | 280 |
| Validation | 60 | 180 | 60 |
| Test | 60 | 180 | 60 |

### 2. Check data loading and models

```bash
python src/dataset.py --batch-size 8
python src/model.py --model simple_cnn
python src/model.py --model resnet18 --pretrained
```

### 3. Train the Simple CNN baseline

```bash
python src/train.py \
  --model simple_cnn \
  --epochs 30 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --patience 8
```

### 4. Train ResNet18 in two stages

Stage 1 trains only the classification head:

```bash
python src/train.py \
  --model resnet18 \
  --pretrained \
  --freeze-backbone \
  --epochs 8 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --patience 4 \
  --output-dir models/resnet18_stage1
```

Stage 2 fine-tunes the complete network:

```bash
python src/train.py \
  --model resnet18 \
  --pretrained \
  --initial-checkpoint models/resnet18_stage1/best_model.pt \
  --epochs 25 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --patience 7 \
  --output-dir models/resnet18
```

### 5. Evaluate

Simple CNN:

```bash
python src/evaluate.py
```

ResNet18:

```bash
python src/evaluate.py \
  --checkpoint models/resnet18/best_model.pt \
  --output-dir results/resnet18
```

### 6. Generate figures and Grad-CAM

```bash
python src/analyze_results.py
python src/gradcam.py
```

## Source files

| File | Purpose |
|---|---|
| `prepare_data.py` | Validate filenames and generate grouped splits |
| `dataset.py` | PyTorch dataset, augmentation, and DataLoaders |
| `model.py` | Simple CNN and ResNet18 definitions |
| `train.py` | Training, validation, early stopping, and checkpoints |
| `evaluate.py` | Held-out testing and per-image predictions |
| `analyze_results.py` | Training curves, comparisons, and error examples |
| `gradcam.py` | Grad-CAM explanations for ResNet18 |

## Notes

- The original TensorFlow notebook is retained under `references/` but is not
  used by the final PyTorch pipeline.
- The original `Images.zip` is retained under `archives/`; the working images
  are in `Images/`.
- ImageNet weights are cached under `.torch/` and do not need to be downloaded
  again on this machine.
- Random seed `9444` is used for the dataset split and training setup.
