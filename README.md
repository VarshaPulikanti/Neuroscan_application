# NeuroScan

Brain MRI tumor classification project. I fine-tuned EfficientNet-B0 to classify scans into glioma, meningioma, pituitary tumor, or no tumor, and added Grad-CAM so you can see what the model is looking at.

I also built a small full-stack app around it: React frontend, FastAPI backend, user accounts, and scan history.

**Dataset:** [Brain Tumor MRI (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

> For research / learning only — not for medical diagnosis.

---

## Results

Trained on the Kaggle MRI dataset (5,600 train / 1,600 test):

| Metric | Score |
|--------|-------|
| Test accuracy | 95.75% |
| Macro F1 | 0.96 |
| Best val accuracy | 99.05% |

Per-class F1: glioma 0.92 · meningioma 0.95 · no tumor 0.96 · pituitary 0.99

---

## What’s in the project

| Part | What I used |
|------|-------------|
| Model | EfficientNet-B0 (ImageNet pretrained), optional ResNet-18 |
| Training | Augmentation, AdamW, LR scheduling, early stopping |
| Explainability | Grad-CAM |
| Backend | FastAPI + PyTorch |
| Frontend | React + TypeScript (Vite) |
| Auth / history | JWT + SQLite (local) / PostgreSQL (Render) |

---

## Project structure

```text
├── app/                 # FastAPI (auth, predict, scan history)
├── frontend/            # React UI
├── src/                 # Model, training, Grad-CAM, DB
├── scripts/             # Dataset download
├── train.py
├── evaluate.py
├── predict.py
├── config.yaml
├── render.yaml          # Render (API + Postgres)
├── start_web.ps1        # Run API + frontend locally
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

The trained checkpoint `outputs/checkpoints/best_model.pth` is in the repo, so you can run the web app / API without training first.

Download data only if you want to train or evaluate yourself:

```bash
python scripts/prepare_data.py --kaggle-download
```

Folder layout:

```text
data/brain_tumor_mri/
  Training/{glioma,meningioma,notumor,pituitary}/
  Testing/...
```

---

## Train / eval / CLI predict

```bash
python train.py
python evaluate.py
python predict.py path/to/mri.jpg --gradcam
```

Config is in `config.yaml`. If you run out of memory on CPU, lower `batch_size` (e.g. to `8`).

---

## Run the web app (local)

```powershell
.\start_web.ps1
# or
.\start_web.bat
```

- UI: http://localhost:5173  
- API: http://localhost:8080  
- Docs: http://localhost:8080/docs  

Or separately:

```powershell
.\run.ps1 api
.\run.ps1 frontend
```

Sign up / log in if you want scans saved to history. You can still classify without an account.

Build frontend + serve from API (optional):

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn app.api:app --host 0.0.0.0 --port 8080
```

---

## Deploy

I deploy this as two services (both can be free for a student demo):

| Part | Host | Notes |
|------|------|--------|
| Frontend | Vercel | Root directory: `frontend/` (free hobby) |
| API | Render | Free web service + SQLite (see `render.yaml`) |

**Render (free)**

1. New → **Web Service** (not a paid Blueprint DB)
2. Connect `Neuroscan_application`, runtime Python
3. Instance type: **Free**
4. Build: `pip install -r requirements-api.txt` (lean install for free tier)
5. Start: `uvicorn app.api:app --host 0.0.0.0 --port $PORT --workers 1`
6. Env vars:
   - `DATABASE_URL` = `sqlite:///./data/neuroscan.db`
   - `JWT_SECRET` = any long random string
   - `CHECKPOINT_PATH` = `outputs/checkpoints/best_model.pth`
   - `CORS_ORIGINS` = your Vercel URL (add after frontend is live)
7. Free instances sleep when idle — first request can be slow

**Vercel (free)**

1. Import the repo, root = `frontend`
2. Set `VITE_API_URL` to your Render API URL (no trailing slash)

Env examples: `.env.example`, `frontend/.env.example`

---

## Config

`config.yaml`:

```yaml
model: efficientnet_b0
epochs: 15
batch_size: 16
learning_rate: 0.0003
image_size: 224
patience: 5
```

---

## Outputs after training

- `outputs/checkpoints/best_model.pth` — best weights  
- `outputs/training_history.png` — loss / accuracy curves  
- `outputs/confusion_matrix.png` — after `evaluate.py`  
