# FlameVision: Wildfire Image Classification

Binary wildfire image classification (fire / nofire) using three pre-trained deep learning architectures — **ResNet-50**, **ViT-B/16**, and **ResMLP-12** — with Optuna-based hyperparameter tuning.

---

## Results

| Model | Accuracy | Macro Recall | Macro F1 |
|---|---|---|---|
| ResNet-50 | 90.91% | 0.9271 | 0.9053 |
| ViT-B/16 | 91.39% | 0.9299 | 0.9101 |
| ResMLP-12 | 91.31% | 0.9285 | 0.9092 |

Evaluated on 2,474 test images (1,587 fire, 887 nofire).

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
optuna>=3.0.0
scikit-learn>=1.2.0
numpy>=1.23.0
pandas>=1.5.0
tqdm>=4.64.0
```

Install with:

```bash
pip install -r requirement.txt
```

> **GPU note:** bfloat16 AMP support (required for ResMLP-12) needs an Ampere-class GPU (RTX 30xx, A100, etc.) with CUDA 11.8+. On older hardware, set `USE_AMP = False` in the `CFG` dataclass.

---

## Dataset

Download from: https://drive.google.com/drive/folders/1XYG9Wen9vwvCvEUQ_oyUm20tq6W8Wx7b?usp=sharing

Extract and arrange the dataset in the following structure:

```
wildfire/
├── train/
│   └── images/
│       ├── fire/
│       └── nofire/
├── valid/
│   └── images/
│       ├── fire/
│       └── nofire/
└── test/
    ├── fire/
    └── nofire/
```

---

## Pre-trained Model
Due to GitHub's file size limits, the PyTorch model (`.pth`) is hosted externally.

**Download Link:** [Download the three .pth files from Google Drive](https://drive.google.com/drive/u/0/folders/15l3wPMP6yRpR5Ft5k2_i1mIgQeS1UBU2)

Place the downloaded file in the `wildfire_clf/app/weights/` directory:

```text
wildfire_clf/
└── app/
    └── weights/          
        ├── resnet.pth      
        ├── resmlp.pth     
        └── vit.pth  
```

---

## Usage

Run each notebook end-to-end. Each notebook executes the following steps automatically:

1. Loads and validates the dataset splits
2. Runs Optuna hyperparameter search (10 trials, optimising validation macro F1)
3. Retrains the model on train + validation combined using the best hyperparameters
4. Evaluates the final model on the test set and saves results

To execute via command line:

```bash
# ResNet-50
jupyter nbconvert --to notebook --execute resnet50.ipynb

# ViT-B/16
jupyter nbconvert --to notebook --execute vit.ipynb

# ResMLP-12 (requires Ampere GPU for bfloat16)
jupyter nbconvert --to notebook --execute resmlp.ipynb
```

---

## Output Files

Each notebook saves the following files to its `OUT_DIR`:

| File | Description |
|---|---|
| `final_model_<name>.pth` | Trained model weights |
| `test_report_<name>.json` | Full per-class metrics in JSON format |
| `test_report_table_<name>.csv` | Per-class metrics as a CSV table |

---

## Deployment

1. Run Locally 

```bash
py -m venv .venv
.venv\scripts\activate
pip install -r requirements.txt

# start api
uvicorn main:app --host 0.0.0.0 --port 8000 # http://localhost:8000/
```

2. Docker Setup

```bash
docker build -t wildfire-clf:latest .
docker run -p 8000:8000 wildfire-clf
```

3. Kubernetes (Minikube Setup)

```bash
minikube start --driver=docker
kubectl get nodes
minikube docker-env | Invoke-Expression
docker build -t wildfire-clf:latest .

# apply deployment
kubectl apply -f deployment.yaml
kubectl get pods
kubectl apply -f service.yaml

# expose service
minikube tunnel 
kubectl get svc # http://127.0.0.1:8000/
```

---