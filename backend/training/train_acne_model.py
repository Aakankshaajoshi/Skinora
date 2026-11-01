
"""Fine-tune ImageNet-pretrained ResNet50 on ACNE04 acne severity.

Optional compatible images:
extra-data/{mild,moderate,severe,very_severe}/image.jpg

Only add images whose labels map reliably to the same four classes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models" / "acne_resnet50.pth"
CLASSES = ["mild", "moderate", "severe", "very_severe"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(SIZE, scale=(0.82, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.12, contrast=0.12, saturation=0.08
    ),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class HFDataset(Dataset):
    """Decode only the image requested by the DataLoader."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.dataset[self.indices[index]]
        image = row["image"].convert("RGB")
        label = int(row["level"])
        return self.transform(image), label


class FolderDataset(Dataset):
    """Load optional images from four class-named directories."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image), label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_extra_images(folder: Path | None):
    items = []

    if folder is None:
        return items

    for class_name, label in CLASS_TO_ID.items():
        class_dir = folder / class_name

        if not class_dir.exists():
            continue

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp"
            }:
                items.append((path, label))

    if items:
        print(f"Found {len(items)} optional extra images in {folder}")

    return items


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    accuracy = correct / max(total, 1)
    average_loss = total_loss / max(total, 1)

    return accuracy, average_loss


def make_loader(dataset, batch_size, shuffle, workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--extra-data-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print("Loading ACNE04 from Hugging Face...")

    # Keep the images in the Hugging Face dataset rather than creating
    # Python lists containing every decoded image.
    ds = load_dataset("Layered-Labs/ACNE04", split="train")

    required_columns = {"image", "level", "split"}
    if not required_columns.issubset(set(ds.column_names)):
        raise RuntimeError(
            f"Unexpected ACNE04 columns: {ds.column_names}. "
            f"Expected at least {sorted(required_columns)}."
        )

    # Read metadata columns only; this does not decode all image files.
    split_values = ds["split"]
    all_labels = np.asarray(ds["level"], dtype=np.int64)

    if not set(np.unique(all_labels)).issubset({0, 1, 2, 3}):
        raise RuntimeError(
            "Expected integer labels 0, 1, 2, 3 in the level column. "
            "Inspect the dataset label mapping before training."
        )

    train_indices = np.asarray([
        i for i, value in enumerate(split_values)
        if str(value).lower() == "train"
    ])

    test_indices = np.asarray([
        i for i, value in enumerate(split_values)
        if str(value).lower() == "test"
    ])

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise RuntimeError(
            "No train/test rows found in the split column. "
            "Inspect ds features and the ACNE04 dataset card."
        )

    train_labels = all_labels[train_indices]

    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.15,
        random_state=args.seed,
        stratify=train_labels,
    )

    # These lists contain integer indices, not image objects.
    train_base = train_idx.tolist()
    val_rows = val_idx.tolist()
    test_rows = test_indices.tolist()

    extra = load_extra_images(args.extra_data_dir)
    extra_train, extra_val = [], []

    if extra:
        extra_labels = np.asarray([label for _, label in extra])
        extra_indices = np.arange(len(extra))

        try:
            e_train, e_val = train_test_split(
                extra_indices,
                test_size=0.15,
                random_state=args.seed,
                stratify=extra_labels,
            )
        except ValueError:
            e_train, e_val = train_test_split(
                extra_indices,
                test_size=0.15,
                random_state=args.seed,
            )

        extra_train = [extra[i] for i in e_train]
        extra_val = [extra[i] for i in e_val]

    train_ds = torch.utils.data.ConcatDataset([
        HFDataset(ds, train_base, TRAIN_TRANSFORM),
        *(
            [FolderDataset(extra_train, TRAIN_TRANSFORM)]
            if extra_train else []
        ),
    ])

    val_ds = torch.utils.data.ConcatDataset([
        HFDataset(ds, val_rows, EVAL_TRANSFORM),
        *(
            [FolderDataset(extra_val, EVAL_TRANSFORM)]
            if extra_val else []
        ),
    ])

    test_ds = HFDataset(ds, test_rows, EVAL_TRANSFORM)

    # Balance the training classes through weighted sampling.
    all_train_labels = (
        all_labels[train_base].tolist()
        + [label for _, label in extra_train]
    )
    counts = np.bincount(all_train_labels, minlength=len(CLASSES))

    sample_weights = [
        1.0 / max(int(counts[int(label)]), 1)
        for label in all_labels[train_base]
    ]
    sample_weights.extend(
        1.0 / max(int(counts[label]), 1)
        for _, label in extra_train
    )

    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False, args.workers, device
    )
    test_loader = make_loader(
        test_ds, args.batch_size, False, args.workers, device
    )

    # ImageNet-pretrained ResNet50 transfer learning.
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # Compatible with standard PyTorch versions with torch.amp support.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    patience = 3
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        val_acc, val_loss = evaluate(model, val_loader, device)
        train_loss = running_loss / max(seen, 1)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0

            checkpoint = {
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "classes": CLASSES,
                "validation_accuracy": round(best_acc, 4),
                "model": "ResNet50",
                "primary_dataset": "ACNE04",
                "pretraining": "ImageNet-1K",
                "image_size": SIZE,
            }
            torch.save(checkpoint, OUTPUT)
        else:
            no_improve += 1

            if no_improve >= patience:
                print("Early stopping.")
                break

    # Load the best validation checkpoint for final test evaluation.
    checkpoint = torch.load(
        OUTPUT, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_acc, test_loss = evaluate(model, test_loader, device)

    metrics = {
        "model": "ResNet50",
        "primary_dataset": "ACNE04",
        "pretraining": "ImageNet-1K",
        "classes": CLASSES,
        "acne04_train_images": len(train_base),
        "acne04_validation_images": len(val_rows),
        "acne04_test_images": len(test_rows),
        "extra_compatible_train_images": len(extra_train),
        "extra_compatible_validation_images": len(extra_val),
        "best_validation_accuracy": round(best_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "checkpoint": str(OUTPUT),
    }

    metrics_path = OUTPUT.parent / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nTraining complete. Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nCheckpoint:", OUTPUT)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
