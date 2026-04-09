import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

import config


def get_dataloaders(transform_fn):
    """
    Downloads CIFAR-10 (if not already downloaded) and returns
    three DataLoaders: train, val, test.

    transform_fn : a function that takes a (3, 32, 32) tensor
                   and returns a preprocessed tensor ready for the model.
                   Each domain passes its own transform here.
    """

    # Download CIFAR-10 — torchvision handles this automatically
    # train=True  → 50,000 images used for train + val
    # train=False → 10,000 images used for test
    train_data = datasets.CIFAR10(root=config.DATA_DIR, train=True,  download=True, transform=ToTensor())
    test_data  = datasets.CIFAR10(root=config.DATA_DIR, train=False, download=True, transform=ToTensor())

    # ── Split the 50k training images into train (70%) and val (15%) ──────────
    # Fixed seed so every domain gets the exact same images in each split
    rng    = np.random.default_rng(config.SEED)
    labels = np.array(train_data.targets)       # integer label for each image
    all_indices = np.arange(len(labels))

    train_idx, val_idx = [], []

    # Split class by class so each class is equally represented in both splits
    for cls in range(config.NUM_CLASSES):
        cls_indices = all_indices[labels == cls]
        rng.shuffle(cls_indices)

        n_train = int(len(cls_indices) * config.TRAIN_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO))
        train_idx.extend(cls_indices[:n_train].tolist())
        val_idx.extend(cls_indices[n_train:].tolist())

    # ── Wrap each split with the domain transform ─────────────────────────────
    train_ds = _TransformDataset(Subset(train_data, train_idx), transform_fn)
    val_ds   = _TransformDataset(Subset(train_data, val_idx),   transform_fn)
    test_ds  = _TransformDataset(test_data,                     transform_fn)

    print(f"Split sizes → train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # ── Create DataLoaders ────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


class _TransformDataset(Dataset):
    """
    Wraps any dataset and applies a transform function on each item.
    This lets every domain inject its own preprocessing without touching the base data.
    """

    def __init__(self, base_dataset, transform_fn):
        self.base_dataset = base_dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]   # image: (3, 32, 32) float tensor in [0, 1]
        image = self.transform_fn(image)         # apply domain-specific preprocessing
        return image, label
