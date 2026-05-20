from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from typing import Annotated, List
from PIL import Image
import time
import uuid
import io
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from collections import deque
from model_utils import load_model

app = FastAPI()
# allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load ALL models
model_paths = [
    "./weights/resnet.pth",
    "./weights/resmlp.pth",
    "./weights/vit.pth"
]

model_names = ["resnet", "resmlp", "vit"]

models = {}
class_names = None

for path, model_name in zip(model_paths, model_names):

    model, classes, loaded_name = load_model(path, device, model_name)
    model.to(device)
    model.eval()
    models[loaded_name] = model

    if class_names is None:
        class_names = classes


tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

metrics = {
    "total": 0,
    "fire_count": 0,
    "conf_sum": 0.0
}

history = deque(maxlen=200)
conf_history = deque(maxlen=200)

@app.get("/")
async def index():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(files: Annotated[List[UploadFile], File(...)]):
    start_time = time.time()
    images = []
    filenames = []

    # -------------------------
    # 1. Load images (CPU)
    # -------------------------
    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x = tfms(image)
        images.append(x)
        filenames.append(file.filename)

    # -------------------------
    # 2. Create batch (CPU → GPU)
    # -------------------------
    batch = torch.stack(images).to(device)

    # -------------------------
    # 3. Inference (GPU)
    # -------------------------
    output = {}

    with torch.inference_mode():
        for model_name, model in models.items():

            logits = model(batch)
            probs = torch.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)
            confs = probs.max(dim=1).values

            for i, fname in enumerate(filenames):
                label = class_names[preds[i].item()]
                prob = float(confs[i].item())

                output.setdefault(fname, {})
                output[fname][model_name] = {
                    "prediction": label,
                    "class_id": int(preds[i].item()),
                    "confidence": prob
                }

    num_new_images = len(filenames)
    metrics["total"] += num_new_images

    first_model = list(models.keys())[0]
    for fname in filenames:
        m_res = output[fname][first_model]
        conf_history.append(m_res["confidence"])
        history.append(m_res["prediction"])
        metrics["conf_sum"] += m_res["confidence"]
        if m_res["prediction"] == "fire":
            metrics["fire_count"] += 1
    latency = time.time() - start_time

    return {"latency_sec": round(latency, 3), "results": output}

@app.post("/predict_csv")
async def predict_csv(files: List[UploadFile] = File(...)):

    images = []
    filenames = []

    # -------------------------
    # 1. Load images
    # -------------------------
    for file in files:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        images.append(tfms(img))
        filenames.append(file.filename)

    # -------------------------
    # 2. Batch
    # -------------------------
    batch = torch.stack(images).to(device)

    rows = []

    # -------------------------
    # 3. Inference (same as /predict)
    # -------------------------
    with torch.inference_mode():
        for model_name, model in models.items():

            logits = model(batch)
            probs = torch.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)
            confs = probs.max(dim=1).values

            for i, fname in enumerate(filenames):

                rows.append({
                    "file": fname,
                    "model": model_name,
                    "prediction": class_names[preds[i].item()],
                    "class_id": int(preds[i].item()),
                    "confidence": float(confs[i].item())
                })

    # -------------------------
    # 4. Save CSV
    # -------------------------
    df = pd.DataFrame(rows)
    # Pivot to get models as columns
    pivot_df = df.pivot(index="file", columns="model", values="prediction")
    
    # Save to a StringIO buffer instead of a physical file
    stream = io.StringIO()
    pivot_df.to_csv(stream)
    
    # Seek to the start of the stream so the response can read it
    stream.seek(0)
    
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=wildfire_results.csv"}
    )

BASELINE = {
    "fire_ratio": 0.51,  # Based on ~51% fire in training
    "avg_conf": 0.92,    # Based on your high precision scores
    "conf_std": 0.05     # Standard deviation of confidence in validation
}

# In-memory storage for rolling window
history = []       # stores "fire" or "nofire"
conf_history = []  # stores float confidence values
WINDOW_SIZE = 100

@app.get("/metrics")
def get_metrics():
    if len(history) < 5:
        return {"message": "Insufficient data for drift analysis"}

    # 1. Calculate Current State
    current_fire_ratio = sum(1 for x in history if x == "fire") / len(history)
    current_avg_conf = np.mean(conf_history)
    current_conf_std = np.std(conf_history)

    # 2. Calculate the Three Deltas (Shift Magnitude)
    # Range [0, 1] - 0 is stable, 1 is total shift
    delta_label = abs(current_fire_ratio - BASELINE["fire_ratio"])
    delta_conf = abs(current_avg_conf - BASELINE["avg_conf"])
    delta_stability = abs(current_conf_std - BASELINE["conf_std"])

    if metrics["total"] == 0:
        return {"raw_counts": {"total_processed": 0}}

    return {
        "status": "CRITICAL" if any([delta_label > 0.4, delta_conf > 0.3]) else "STABLE",
        "deltas": {
            "label_drift_delta": round(delta_label, 4),
            "confidence_drift_delta": round(delta_conf, 4),
            "stability_drift_delta": round(delta_stability, 4)
        },
        "raw_counts": {
            "total_processed": metrics["total"],
            "window_fire_ratio": current_fire_ratio,
            "window_avg_conf": current_avg_conf
        }
    }