import os
import torch
from PIL import Image
import torchvision.transforms as transforms

from model_utils import load_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 ALL YOUR MODELS HERE
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

    # 🔹 transforms
    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    image_dir = "test"
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    batch_size = 16

    # 🔥 batch loop
    for i in range(0, len(image_files), batch_size):

        batch_files = image_files[i:i + batch_size]

        images = []
        for fname in batch_files:
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            images.append(tfms(img))

        batch = torch.stack(images).to(device)

        # 🔥 run all models
        for model_name, model in models.items():
            with torch.no_grad():
                probs = torch.softmax(model(batch), dim=1)

            print(f"\n=== {model_name.upper()} ===")

            for j, fname in enumerate(batch_files):
                pred = probs[j].argmax().item()
                prob = probs[j][pred].item()

                print(f"{fname} → {class_names[pred]} ({prob:.3f})")


if __name__ == "__main__":
    main()