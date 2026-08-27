from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def labels_from_row(row: pd.Series, count: int = 40, max_labels: int = 30) -> torch.Tensor:
    labels = torch.zeros(count, dtype=torch.float32)
    for index in range(max_labels):
        value = row.get(f"label_{index}")
        if pd.notna(value) and value != "":
            labels[int(value)] = 1
    return labels

CLASS_NAMES = (
    "tennis", "skate", "football", "swimming", "cemetery", "garage", "golf",
    "roundabout", "parkinglot", "supermarket", "school", "marina", "baseball",
    "fall", "pond", "airport", "beach", "bridge", "religious", "residential",
    "warehouse", "office", "farmland", "university", "forest", "lake",
    "naturereserve", "park", "sand", "soccer", "equestrian", "shooting",
    "icerink", "commercialarea", "garden", "dam", "railroad", "highway",
    "river", "wetland",
)
