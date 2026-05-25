import os

from PIL import Image
from torch.utils.data import Dataset


class ImageNetLTDataset(Dataset):
    """Dataset wrapper for ImageNet-LT split files."""

    def __init__(self, image_root, lt_root, split="val", transform=None, return_path=False):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported ImageNet-LT split: {split}")

        self.image_root = os.path.expanduser(image_root)
        self.lt_root = os.path.expanduser(lt_root)
        self.split = split
        self.transform = transform
        self.return_path = return_path

        split_file = os.path.join(self.lt_root, f"ImageNet_LT_{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"ImageNet-LT split file not found: {split_file}")

        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.split()
                self.samples.append((rel_path, int(label)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel_path, target = self.samples[index]
        img_path = os.path.join(self.image_root, rel_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, target, rel_path
        return image, target
