import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

CLASSES = ["internal_void", "surface", "corona", "floating"]


def make_gaussian_grid(size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(coords, coords)
    return xx, yy


def gaussian_blob(xx: np.ndarray, yy: np.ndarray, center: Tuple[float, float], sigma: float) -> np.ndarray:
    cx, cy = center
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def generate_pattern(class_name: str, size: int = 128) -> np.ndarray:
    xx, yy = make_gaussian_grid(size)
    image = np.zeros_like(xx)

    rng = np.random.default_rng()

    if class_name == "internal_void":
        centers = [(-0.2, -0.2), (0.25, 0.25)]
        sigmas = [0.18, 0.22]
        weights = [1.0, 0.8]
    elif class_name == "surface":
        centers = [(-0.6, 0.0), (-0.2, 0.0), (0.2, 0.0), (0.6, 0.0)]
        sigmas = [0.08, 0.12, 0.12, 0.08]
        weights = [0.8, 1.0, 1.0, 0.8]
    elif class_name == "corona":
        ring = [(0.5 * np.cos(theta), 0.5 * np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        centers = [(0.0, 0.0)] + ring
        sigmas = [0.12] + [0.08] * len(ring)
        weights = [1.2] + [0.6] * len(ring)
    elif class_name == "floating":
        centers = [(rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7)) for _ in range(5)]
        sigmas = list(rng.uniform(0.05, 0.2, size=len(centers)))
        weights = list(rng.uniform(0.6, 1.1, size=len(centers)))
    else:
        centers = [(0.0, 0.0)]
        sigmas = [0.2]
        weights = [1.0]

    for (cx, cy), sigma, w in zip(centers, sigmas, weights):
        image += w * gaussian_blob(xx, yy, (cx, cy), sigma)

    # Random streaks to mimic phase-related discharge bands
    streaks = np.zeros_like(image)
    for _ in range(rng.integers(1, 3)):
        x_pos = rng.uniform(-1.0, 1.0)
        width = rng.uniform(0.05, 0.15)
        streaks += np.exp(-((xx - x_pos) ** 2) / (2 * width ** 2)) * rng.uniform(0.3, 0.7)
    image = image + 0.5 * streaks

    # Noise injection
    noise = rng.normal(0, 0.05, size=image.shape)
    speckle = rng.binomial(1, 0.005, size=image.shape) * rng.uniform(0.3, 0.9)
    image = image + noise + speckle

    # Normalize to 0-1
    image = image - image.min()
    image = image / (image.max() + 1e-8)

    # Slight contrast variation
    contrast = rng.uniform(0.8, 1.2)
    brightness = rng.uniform(-0.05, 0.05)
    image = np.clip(image * contrast + brightness, 0.0, 1.0)

    return image


def save_image(arr: np.ndarray, path: Path) -> None:
    img = (arr * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def generate_dataset(output_root: Path, num_per_class: int = 300, size: int = 128) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        class_dir = output_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(num_per_class):
            arr = generate_pattern(class_name, size=size)
            filename = class_dir / f"{class_name}_{idx:04d}.png"
            save_image(arr, filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic PRPD dataset")
    parser.add_argument("--output_root", type=str, default="datasets/synthetic_prpd", help="Output dataset root directory")
    parser.add_argument("--num_per_class", type=int, default=300, help="Number of images per class")
    parser.add_argument("--size", type=int, default=128, help="Image size (pixels)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    generate_dataset(output_root, num_per_class=args.num_per_class, size=args.size)
    print(f"Synthetic dataset saved to {output_root.resolve()}")


if __name__ == "__main__":
    main()
