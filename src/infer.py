import argparse
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

from model_pdresnet import PDResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single PRPD image")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file")
    parser.add_argument("--input_channels", type=int, default=1, help="1 or 2 channels")
    return parser.parse_args()


def add_projection_channel(tensor_img: torch.Tensor) -> torch.Tensor:
    if tensor_img.dim() != 3 or tensor_img.shape[0] != 1:
        raise ValueError("Expected grayscale tensor of shape (1,H,W)")
    _, h, w = tensor_img.shape
    column_sum = tensor_img.sum(dim=1, keepdim=True)
    max_val = torch.clamp(column_sum.max(), min=1e-6)
    projection = (column_sum / max_val).repeat(1, h, 1)
    return torch.cat([tensor_img, projection], dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = PDResNet(num_classes=len(class_to_idx), input_channels=args.input_channels)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    img_path = Path(args.image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    transform = transforms.ToTensor()
    with Image.open(img_path) as img:
        img = img.convert("L")
    tensor_img = transform(img)
    if args.input_channels == 2:
        tensor_img = add_projection_channel(tensor_img)
    tensor_img = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        pred_idx = outputs.argmax(dim=1).item()
    pred_class = idx_to_class.get(pred_idx, str(pred_idx))
    print(f"Predicted class: {pred_class}")


if __name__ == "__main__":
    main()
