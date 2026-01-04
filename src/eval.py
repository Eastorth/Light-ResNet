import argparse
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset_prpd import PRPDDataset, build_transforms
from model_pdresnet import PDResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PRPD classifier")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--input_channels", type=int, default=1, help="1 or 2 channels")
    return parser.parse_args()


def evaluate(model: PDResNet, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray]:
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    preds_np = np.concatenate(preds_all)
    labels_np = np.concatenate(labels_all)
    acc = (preds_np == labels_np).mean().item()
    macro_f1 = f1_score(labels_np, preds_np, average="macro")
    conf_mat = confusion_matrix(labels_np, preds_np)
    return acc, macro_f1, conf_mat


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)

    dataset = PRPDDataset(args.data_root, transform=build_transforms(train=False), input_channels=args.input_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = PDResNet(num_classes=num_classes, input_channels=args.input_channels)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    acc, macro_f1, conf_mat = evaluate(model, loader, device)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix:\n", conf_mat)


if __name__ == "__main__":
    main()
