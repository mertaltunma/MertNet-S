from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class StableImageNetDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        # CSV başlıklarında boşluk problemi olmasın
        self.df.columns = [col.strip() for col in self.df.columns]

        self.transform = transform

        required_columns = {"filepath", "label", "class_name", "split"}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"CSV dosyasında eksik sütunlar var: {missing_columns}. "
                f"Mevcut sütunlar: {list(self.df.columns)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["filepath"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(input_size=160):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(176),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def create_dataloaders(
    train_csv,
    val_csv,
    input_size=160,
    batch_size=32,
    num_workers=2
):
    train_transform, val_transform = get_transforms(input_size=input_size)

    train_dataset = StableImageNetDataset(
        csv_path=train_csv,
        transform=train_transform
    )

    val_dataset = StableImageNetDataset(
        csv_path=val_csv,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        train_csv="data_splits/stable_imagenet_train.csv",
        val_csv="data_splits/stable_imagenet_val.csv",
        input_size=160,
        batch_size=32,
        num_workers=2
    )

    print("=" * 70)
    print("Stable ImageNet-1K DataLoader Kontrolü")
    print("=" * 70)
    print(f"Train veri sayısı: {len(train_dataset)}")
    print(f"Validation veri sayısı: {len(val_dataset)}")
    print(f"Train batch sayısı: {len(train_loader)}")
    print(f"Validation batch sayısı: {len(val_loader)}")

    images, labels = next(iter(train_loader))

    print(f"Bir train batch görüntü boyutu: {tuple(images.shape)}")
    print(f"Bir train batch label boyutu: {tuple(labels.shape)}")
    print(f"İlk 10 label: {labels[:10].tolist()}")