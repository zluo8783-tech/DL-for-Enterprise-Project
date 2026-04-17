from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Annotated, List
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

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

# 🔥 load ALL models
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

@app.get("/")
async def index():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(files: Annotated[List[UploadFile], File(...)]):

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

                output.setdefault(fname, {})

                output[fname][model_name] = {
                    "prediction": class_names[preds[i].item()],
                    "class_id": int(preds[i].item()),
                    "confidence": float(confs[i].item())
                }

    return {"results": output}