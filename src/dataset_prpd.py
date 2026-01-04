from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PRPDDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        input_channels: int = 1,
    ) -> None:
        self.root = Path(root)
        if input_channels not in (1, 2):
            raise ValueError("input_channels must be 1 or 2")
        self.input_channels = input_channels
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Tuple[Path, int]] = []
        self._scan()

    def _scan(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root {self.root} does not exist")
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        for cls_name in classes:
            cls_dir = self.root / cls_name
            for fname in sorted(cls_dir.iterdir()):
                if fname.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue
                self.samples.append((fname, self.class_to_idx[cls_name]))
        if not self.samples:
            raise RuntimeError(f"No image files found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _add_projection_channel(self, tensor_img: torch.Tensor) -> torch.Tensor:
        if tensor_img.dim() != 3:
            raise ValueError("Expected image tensor shape (C,H,W)")
        if tensor_img.shape[0] != 1:
            raise ValueError("Projection channel expects a single-channel input before concatenation")
        _, h, w = tensor_img.shape
        column_sum = tensor_img.sum(dim=1, keepdim=True)  # (1, W)
        max_val = torch.clamp(column_sum.max(), min=1e-6)
        projection = (column_sum / max_val).repeat(1, h, 1)  # (1, H, W)
        return torch.cat([tensor_img, projection], dim=0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("L")
        tensor_img = self.transform(img)
        if self.input_channels == 2:
            tensor_img = self._add_projection_channel(tensor_img)
        return tensor_img, label


def build_transforms(train: bool = False) -> Callable:
    if train:
        return transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
                transforms.ToTensor(),
            ]
        )
    return transforms.ToTensor()
