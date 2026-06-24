# NeuroScan

**Brain MRI Tumor Classification** — A production-style PyTorch deep learning project that classifies brain MRI scans into glioma, meningioma, pituitary tumor, or no tumor, with Grad-CAM explainability and deployable inference APIs.



**Dataset:** [Brain Tumor MRI (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Highlights

| DL component | Implementation |
|--------------|----------------|
| Architecture | Fine-tuned **EfficientNet-B0** (ImageNet pretrained) |
| Dataset | Brain Tumor MRI — 4-class classification (7,200 images) |
| Training | Augmentation, AdamW, LR scheduling, early stopping |
| Evaluation | Accuracy, F1, confusion matrix, per-class metrics |
| Explainability | **Grad-CAM** heatmaps highlighting tumor regions |
| Deployment | Streamlit demo + FastAPI inference API |

## Results (trained on real Kaggle dataset)

| Metric | Score |
|--------|-------|
| **Test accuracy** | **95.75%** |
| **Macro F1** | **0.96** |
| Best validation accuracy | 99.05% |
| Dataset | 5,600 train / 1,600 test MRI images |
| Model | EfficientNet-B0 (ImageNet fine-tuning) |

Per-class test F1: glioma 0.92 · meningioma 0.95 · no tumor 0.96 · pituitary 0.99




---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, torchvision |
| Explainability | Grad-CAM |
| Data & ML | NumPy, scikit-learn, OpenCV |
| Visualization | Matplotlib, Seaborn |
| Demo | Streamlit |
| API | FastAPI, Uvicorn |
| Language | Python 3.10+ |

---

## Project Structure

```text
dlproject/
├── app/
│   ├── streamlit_app.py    # Interactive MRI classifier + Grad-CAM
│   └── api.py              # FastAPI inference service
├── scripts/
│   └── prepare_data.py     # Dataset download / organization
├── src/
│   ├── config.py           # Config loader + ImageNet normalization
│   ├── data.py             # MRI dataloaders & augmentation
│   ├── trainer.py          # Training loop with early stopping
│   ├── gradcam.py          # Grad-CAM heatmap generation
│   ├── utils.py            # Metrics, checkpoints, plots
│   └── models/
│       ├── efficientnet.py
│       └── resnet.py
├── tests/
├── config.yaml
├── train.py
├── evaluate.py
├── predict.py
├── run.ps1                 # Windows helper (train, demo, api, test)
├── start_demo.bat          # One-click Streamlit launcher (Windows)
├── start_demo.ps1
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
cd dlproject
pip install -r requirements.txt
```

### 2. Download dataset

```bash
python scripts/prepare_data.py --kaggle-download
```

Expected layout:

```text
data/brain_tumor_mri/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    ...
```

### 3. Train, evaluate, and predict

```bash
# Default: EfficientNet-B0 (config.yaml)
python train.py

# Explicit options
python train.py --model efficientnet_b0 --epochs 15

python evaluate.py
python predict.py data/brain_tumor_mri/Testing/glioma/glioma_00.jpg --gradcam
```

If you hit memory errors on CPU, lower `batch_size` in `config.yaml` (e.g. `8`).

### 4. Launch the demo

**Recommended on Windows** — double-click or run:

```powershell
.\start_demo.bat
```

Or:

```powershell
.\start_demo.ps1
streamlit run app/streamlit_app.py
```

### 5. Start the API

```bash
uvicorn app.api:app --reload --port 8000
```

`POST /predict` with an image file · Docs at `http://localhost:8000/docs`

### Windows shortcuts (`run.ps1`)

```powershell
.\run.ps1 train
.\run.ps1 eval
.\run.ps1 demo
.\run.ps1 api
.\run.ps1 test
```

---

## Configuration

Edit `config.yaml`:

```yaml
model: efficientnet_b0
epochs: 15
batch_size: 16
learning_rate: 0.0003
image_size: 224
patience: 5
```

---

## Outputs

After training, check `outputs/`:

- `checkpoints/best_model.pth` — best model weights
- `training_history.png` — loss & accuracy curves
- `confusion_matrix.png` — per-class performance (after `evaluate.py`)

---


