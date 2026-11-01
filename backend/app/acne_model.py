"""ResNet50 inference for Skinora's ACNE04 acne-severity model.

The model is deliberately optional: the API reports `not_trained` until a
locally trained checkpoint exists. It never substitutes a heuristic prediction.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "acne_resnet50.pth"
CLASS_NAMES = ["mild", "moderate", "severe", "very_severe"]
IMAGE_SIZE = 224

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def model_status() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {
            "available": False,
            "status": "not_trained",
            "model": "ResNet50",
            "dataset": "ACNE04",
            "message": "Train the model with backend/training/train_acne_model.py before enabling image predictions.",
        }
    return {"available": True, "status": "ready", "model": "ResNet50", "dataset": "ACNE04", "message": "Local acne-severity model checkpoint found."}


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        return None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint.get("classes", CLASS_NAMES)
    net = models.resnet50(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net, classes, checkpoint.get("validation_accuracy")


def predict_acne(image_bytes: bytes) -> dict[str, Any]:
    loaded = _load_model()
    if loaded is None:
        return {"available": False, "status": "not_trained", "message": "No trained ResNet50 checkpoint is installed. The photo was not classified."}
    try:
        from io import BytesIO
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return {"available": False, "status": "invalid_image", "message": "Could not read this image for model inference."}
    net, classes, val_acc = loaded
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0)
    with torch.inference_mode():
        probs = torch.softmax(net(tensor), dim=1)[0]
    index = int(torch.argmax(probs).item())
    confidence = float(probs[index].item())
    return {
        "available": True,
        "status": "prediction_available",
        "model": "ResNet50 fine-tuned on ACNE04",
        "class": classes[index],
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "validation_accuracy": val_acc,
        "warning": "Research prototype only. ACNE04 is a small academic-use dataset; this prediction is not a diagnosis and may not generalize to selfies, different lighting, or all skin tones. Seek a dermatologist for medical concerns.",
    }
