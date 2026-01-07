import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_prpd import PRPDDataset, build_transforms
from model_pdresnet import PDResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PRPD classifier")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--input_channels", type=int, default=1, help="1 for grayscale, 2 to append projection channel")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    base_dataset = PRPDDataset(args.data_root, transform=build_transforms(train=False), input_channels=args.input_channels)
    val_size = int(len(base_dataset) * args.val_ratio)
    train_size = len(base_dataset) - val_size
    indices = torch.randperm(len(base_dataset), generator=torch.Generator().manual_seed(args.seed))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = PRPDDataset(args.data_root, transform=build_transforms(train=True), input_channels=args.input_channels)
    val_dataset = PRPDDataset(args.data_root, transform=build_transforms(train=False), input_channels=args.input_channels)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, base_dataset.class_to_idx


def evaluate(model: PDResNet, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    preds_np = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    acc = (preds_np == labels_np).mean().item()
    macro_f1 = f1_score(labels_np, preds_np, average="macro")
    conf_mat = confusion_matrix(labels_np, preds_np)
    return acc, macro_f1, conf_mat


def save_checkpoint(model: PDResNet, class_to_idx: Dict[str, int], args: argparse.Namespace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx, "args": vars(args)}, path)


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_to_idx = prepare_dataloaders(args)
    model = PDResNet(num_classes=len(class_to_idx), input_channels=args.input_channels).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

    history = []
    best_acc = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_acc, val_macro_f1, conf_mat = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "confusion_matrix": conf_mat.tolist(),
            "epoch_time": epoch_time,
        }
        history.append(metrics)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_macro_f1={val_macro_f1:.4f} | "
            f"time={epoch_time:.1f}s"
        )
        print("Confusion matrix:\n", conf_mat)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, class_to_idx, args, out_dir / "best_model.pt")
            print(f"Saved new best model with val_acc={best_acc:.4f}")

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Training completed.")


if __name__ == "__main__":
    train()
