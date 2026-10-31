"""Fine-tune the DeepFaune species classifier on manually reviewed labels.

Uses the `manual_label` column written by review_labels.py. Only the classifier head
and the last few transformer blocks are trained; the rest of the DINOv2 ViT-L backbone
stays frozen, which suits the small number of manual labels.

Each run writes to models/deepfaune_finetuned_<date>_<git hash>/:
- weights.pt          state dict plus class names, git hash, date and hyperparameters
- metadata.json       the same metadata without the weights, plus training history
- training_data/      the exact crops used for training and validation, with labels.csv
- confusion_matrix.svg  validation counts for the best model, with class labels
- finetune_classifier.py and git_diff.patch (if the work tree was dirty)

Validation predictions and rendered images are written to
labels/labels_<run>/validation_predictions/.

Usage:
    uv run src/labelling/finetune_classifier.py
"""

import csv
import json
import random
import shutil
import subprocess
from ast import literal_eval
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from matplotlib import rc_context
from matplotlib.figure import Figure
from PIL import Image, ImageDraw
from PytorchWildlife.models.classification import DeepfauneClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

try:
    from . import label_data
    from .organize_images import image_paths, nfc
except ImportError:
    import label_data
    from organize_images import image_paths, nfc

REPO_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = REPO_DIR / "labels"
CSV_PATH = LABELS_DIR / "human_labels.csv"
MODELS_DIR = REPO_DIR / "models"

# The classifier only sees MegaDetector "animal" crops, so these labels are not trained
EXCLUDED_LABELS = {"", "none", "unknown", "human", "vehicle"}
MIN_SAMPLES_PER_CLASS = 5

NUM_UNFROZEN_BLOCKS = 2
EPOCHS = 15
BATCH_SIZE = 16
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
VAL_FOLDS = 5  # one fold (~20 %) is held out for validation
SEQUENCE_GAP_SECONDS = (
    600  # cameras retrigger every ~3 min; one visit stays in one split
)
SEED = 42

IMAGE_SIZE = DeepfauneClassifier.IMAGE_SIZE
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Warm-start new head rows from the closest DeepFaune class
DEEPFAUNE_EQUIVALENT = {
    "badger": "badger",
    "fox": "fox",
    "wild boar": "wild boar",
    "roe deer": "roe deer",
    "dog": "dog",
    "cat": "cat",
    "squirrel": "squirrel",
    "hare": "lagomorph",
    "marten": "mustelid",
    "racoon": "raccoon",
    "raccoon": "raccoon",
    "pigeon": "bird",
    "jay": "bird",
    "buzzard": "bird",
    "crow": "bird",
    "owl": "bird",
    "song_bird": "bird",
    "woodpecker": "bird",
}

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            (IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.7, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        # Night images from the cameras are infrared greyscale
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]
)

# Matches DeepfauneClassifier's inference transform
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]
)


class CropDataset(Dataset):
    def __init__(self, paths, targets, transform):
        self.paths = list(paths)
        self.targets = list(targets)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB")), self.targets[idx]


def git_info():
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(REPO_DIR), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    return {
        "git_hash": run("rev-parse", "HEAD").strip(),
        "git_dirty": bool(run("status", "--porcelain").strip()),
        "git_diff": run("diff", "HEAD"),
    }


def load_labelled_rows():
    df = (
        pd.DataFrame(label_data.load_rows(CSV_PATH))
        if Path(CSV_PATH).name == "human_labels.csv"
        else pd.read_csv(CSV_PATH, keep_default_na=False)
    )
    df["manual_label"] = df["manual_label"].str.strip()
    df = df[(df["box"] != "") & ~df["manual_label"].isin(EXCLUDED_LABELS)].copy()

    counts = df["manual_label"].value_counts()
    rare = counts[counts < MIN_SAMPLES_PER_CLASS]
    if len(rare):
        print(
            f"Skipping classes with < {MIN_SAMPLES_PER_CLASS} samples: {rare.to_dict()}"
        )
    df = df[df["manual_label"].isin(counts[counts >= MIN_SAMPLES_PER_CLASS].index)]

    indexed_paths = image_paths()
    paths = [
        indexed_paths.get((nfc(loc), name))
        for loc, name in zip(df["location_id"], df["image_file"])
    ]
    exists = np.array([p is not None for p in paths])
    if not exists.all():
        print(f"Skipping {(~exists).sum()} rows whose image file is missing")
    df = df.assign(image_path=[str(p) for p in paths])
    return df[exists].reset_index(drop=True)


def assign_sequence_groups(df):
    """Group bursts of images so near-duplicates never straddle train and val."""
    time = pd.to_datetime(
        df["timestamp_exif"].where(df["timestamp_exif"] != "", df["timestamp"]),
        errors="coerce",
    )
    order = np.lexsort(
        (
            time.values.astype("datetime64[ns]").astype(np.int64),
            df["location_id"].values,
        )
    )
    groups = np.empty(len(df), dtype=int)
    group = -1
    prev_loc, prev_time = None, None
    for i in order:
        loc, t = df.at[i, "location_id"], time.iat[i]
        new_sequence = (
            loc != prev_loc
            or pd.isna(t)
            or pd.isna(prev_time)
            or (t - prev_time).total_seconds() > SEQUENCE_GAP_SECONDS
        )
        if new_sequence:
            group += 1
        groups[i] = group
        prev_loc, prev_time = loc, t
    return groups


def crop_box(img, box):
    """Crop a YOLO-format [x_center, y_center, w, h] box, as label_images.py does."""
    xc, yc, w, h = box
    img_w, img_h = img.size
    x1 = max(0, int((xc - w / 2) * img_w))
    y1 = max(0, int((yc - h / 2) * img_h))
    x2 = min(img_w, int((xc + w / 2) * img_w))
    y2 = min(img_h, int((yc + h / 2) * img_h))
    return img.crop((x1, y1, x2, y2))


def export_training_data(df, data_dir):
    crop_paths = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Exporting crops"):
        label_dir = data_dir / row.split / row.manual_label.replace(" ", "_")
        label_dir.mkdir(parents=True, exist_ok=True)
        crop_path = label_dir / f"{row.location_id}_{Path(row.image_file).stem}.jpg"
        with Image.open(row.image_path) as img:
            crop_box(img.convert("RGB"), literal_eval(row.box)).save(
                crop_path, quality=95
            )
        crop_paths.append(crop_path)
    df = df.assign(crop_path=[str(p.relative_to(data_dir)) for p in crop_paths])
    df.to_csv(data_dir / "labels.csv", index=False)
    return crop_paths


def save_validation_predictions(rows, targets, predictions, class_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    for row, target, prediction in tqdm(
        zip(rows.itertuples(), targets, predictions, strict=True),
        total=len(rows),
        desc="Saving validation images",
    ):
        is_correct = target == prediction
        status = "correct" if is_correct else "incorrect"
        color = (46, 125, 50) if is_correct else (198, 40, 40)
        prediction_rows.append(
            {
                "location_id": row.location_id,
                "image_file": row.image_file,
                "true_label": class_names[target],
                "predicted_label": class_names[prediction],
                "status": status,
            }
        )
        label = f"True: {class_names[target]} | Predicted: {class_names[prediction]}"
        image_path = Path(row.image_path)
        output_path = output_dir / status / f"{row.location_id}_{row.image_file}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        label_bounds = draw.textbbox((0, 0), label)
        label_width = int(label_bounds[2] - label_bounds[0])
        label_height = int(label_bounds[3] - label_bounds[1])
        label_left, label_top = 6, 6
        try:
            box = literal_eval(row.box)
            if (
                isinstance(box, (list, tuple))
                and len(box) == 4
                and all(0 <= value <= 1 for value in box)
            ):
                x_center, y_center, width, height = box
                left = max(0, int((x_center - width / 2) * image.width))
                top = max(0, int((y_center - height / 2) * image.height))
                right = min(image.width - 1, int((x_center + width / 2) * image.width))
                bottom = min(
                    image.height - 1, int((y_center + height / 2) * image.height)
                )
                if right > left and bottom > top:
                    draw.rectangle((left, top, right, bottom), outline=color, width=2)
                    label_left = left
                    label_top = max(0, top - label_height - 6)
        except (ValueError, SyntaxError, TypeError):
            pass
        label_left = min(label_left, max(0, image.width - label_width - 6))
        label_top = min(label_top, max(0, image.height - label_height - 6))
        draw.rectangle(
            (
                label_left,
                label_top,
                label_left + label_width + 6,
                label_top + label_height + 6,
            ),
            fill=color,
        )
        draw.text((label_left + 3, label_top + 3), label, fill="white")
        image.save(output_path, quality=95)

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "location_id",
                "image_file",
                "true_label",
                "predicted_label",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)


def save_confusion_matrix(targets, predictions, class_names, output_path):
    size = max(8, len(class_names) * 0.6)
    figure = Figure(figsize=(size, size), layout="constrained")
    axes = figure.subplots()
    ConfusionMatrixDisplay.from_predictions(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        display_labels=class_names,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        ax=axes,
    )
    axes.set_title("Validation confusion matrix (best model)")
    for label in axes.get_xticklabels():
        label.set_horizontalalignment("right")
    with rc_context({"svg.fonttype": "none"}):
        figure.savefig(output_path, format="svg", bbox_inches="tight")


def build_model(class_names, device):
    pretrained = DeepfauneClassifier(device="cpu", class_name_lang="en")
    model = pretrained.predictor
    deepfaune_names = list(pretrained.CLASS_NAMES.values())
    old_head = model.head

    new_head = nn.Linear(old_head.in_features, len(class_names))
    with torch.no_grad():
        for i, name in enumerate(class_names):
            equivalent = DEEPFAUNE_EQUIVALENT.get(name)
            if equivalent in deepfaune_names:
                j = deepfaune_names.index(equivalent)
                new_head.weight[i] = old_head.weight[j]
                new_head.bias[i] = old_head.bias[j]
    model.head = new_head
    model.num_classes = len(class_names)

    for p in model.parameters():
        p.requires_grad = False
    unfrozen = [
        model.blocks[-NUM_UNFROZEN_BLOCKS:],
        model.norm,
        model.fc_norm,
        model.head,
    ]
    for module in unfrozen:
        for p in module.parameters():
            p.requires_grad = True

    return model.to(device)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            total_loss += criterion(logits, y.to(device)).item() * len(y)
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.tolist())
    return total_loss / len(targets), np.array(preds), np.array(targets)


def load_finetuned_classifier(weights_path, device="cpu"):
    """Return (model, class_names, transform) for a weights.pt written by this script."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = timm.create_model(
        checkpoint["backbone"],
        pretrained=False,
        num_classes=len(checkpoint["class_names"]),
        dynamic_img_size=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["class_names"], EVAL_TRANSFORM


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    git = git_info()
    created = datetime.now()
    run_name = f"deepfaune_finetuned_{created:%Y%m%d-%H%M%S}_{git['git_hash'][:8]}"
    if git["git_dirty"]:
        run_name += "-dirty"
        print(
            "⚠️  Work tree has uncommitted changes; saving git_diff.patch alongside the weights"
        )
    out_dir = MODELS_DIR / run_name
    out_dir.mkdir(parents=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    df = load_labelled_rows()
    class_names = sorted(df["manual_label"].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    df["target"] = df["manual_label"].map(class_to_idx)

    groups = assign_sequence_groups(df)
    splitter = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=SEED)
    _, val_idx = next(splitter.split(df, df["target"], groups))
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df["sequence_group"] = groups
    print(pd.crosstab(df["manual_label"], df["split"]))

    crop_paths = export_training_data(df, out_dir / "training_data")
    train_mask = (df["split"] == "train").values
    train_ds = CropDataset(
        np.array(crop_paths)[train_mask], df.loc[train_mask, "target"], TRAIN_TRANSFORM
    )
    val_ds = CropDataset(
        np.array(crop_paths)[~train_mask], df.loc[~train_mask, "target"], EVAL_TRANSFORM
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_model(class_names, device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,}")

    # Inverse square-root frequency weights soften the class imbalance
    counts = np.bincount(df.loc[train_mask, "target"], minlength=len(class_names))
    class_weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=LABEL_SMOOTHING,
    )

    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": LR_HEAD},
            {"params": backbone_params, "lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader)
    )

    trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    best_score, best_epoch, best_state, history = -1.0, None, None, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        val_loss, preds, targets = evaluate(model, val_loader, criterion, device)
        val_acc = float((preds == targets).mean())
        val_bal_acc = float(balanced_accuracy_score(targets, preds))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_bal_acc,
            }
        )
        print(
            f"Epoch {epoch:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} val_bal_acc={val_bal_acc:.3f}"
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation", val_loss, epoch)
        writer.flush()
        if val_bal_acc > best_score:
            best_score, best_epoch = val_bal_acc, epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if k in trainable_keys
            }

    writer.close()
    model.load_state_dict(best_state, strict=False)
    _, preds, targets = evaluate(model, val_loader, criterion, device)
    confusion_matrix_path = out_dir / "confusion_matrix.svg"
    save_confusion_matrix(targets, preds, class_names, confusion_matrix_path)
    print(f"Validation confusion matrix saved to {confusion_matrix_path}")
    val_df = df[~train_mask]
    validation_images_dir = label_data.validation_predictions_dir(REPO_DIR, run_name)
    save_validation_predictions(
        val_df, targets, preds, class_names, validation_images_dir
    )
    print(f"Validation prediction images saved to {validation_images_dir}")
    report = classification_report(
        targets,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    baseline_acc = float((val_df["class"] == val_df["manual_label"]).mean())
    print(f"\nBest epoch: {best_epoch} (val balanced accuracy {best_score:.3f})")
    print(f"Original pipeline accuracy on the same val set: {baseline_acc:.3f}")
    print(
        classification_report(
            targets,
            preds,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    metadata = {
        "created": created.isoformat(timespec="seconds"),
        "git_hash": git["git_hash"],
        "git_dirty": git["git_dirty"],
        "base_model": DeepfauneClassifier.MODEL_NAME,
        "backbone": DeepfauneClassifier.BACKBONE,
        "image_size": IMAGE_SIZE,
        "class_names": class_names,
        "source_csv": str(CSV_PATH.relative_to(REPO_DIR)),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "hyperparameters": {
            "num_unfrozen_blocks": NUM_UNFROZEN_BLOCKS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_head": LR_HEAD,
            "lr_backbone": LR_BACKBONE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
            "sequence_gap_seconds": SEQUENCE_GAP_SECONDS,
            "seed": SEED,
        },
        "best_epoch": best_epoch,
        "val_balanced_accuracy": best_score,
        "val_baseline_accuracy": baseline_acc,
        "val_report": report,
        "history": history,
    }
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            **{k: v for k, v in metadata.items() if k not in ("val_report", "history")},
        },
        out_dir / "weights.pt",
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    shutil.copy2(__file__, out_dir / Path(__file__).name)
    if git["git_dirty"]:
        (out_dir / "git_diff.patch").write_text(git["git_diff"])

    print(f"\n✓ Saved fine-tuned classifier to {out_dir}")


if __name__ == "__main__":
    main()
