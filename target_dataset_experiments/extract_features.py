from pathlib import Path
import time
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append("source_training")

from model_architecture import MertNetS


MODEL_PATH = Path("exported_model/mertnet_s_best.pth")

TRAIN_CSV = Path("results/intel_dataset/intel_train.csv")
TEST_CSV = Path("results/intel_dataset/intel_test.csv")

RESULTS_DIR = Path("results/intel_features")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SIZE = 160
BATCH_SIZE = 32
NUM_CLASSES_SOURCE = 1000
FEATURE_DIM = 384
SEED = 42


class IntelFeatureDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df.columns = [col.strip() for col in self.df.columns]
        self.transform = transform

        self.class_names = sorted(self.df["class_name"].unique())
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        self.df["label"] = self.df["class_name"].map(self.class_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["filepath"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])
        class_name = row["class_name"]

        return image, label, class_name


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = MertNetS(num_classes=NUM_CLASSES_SOURCE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        best_top1 = checkpoint.get("best_val_top1", None)
        epoch = checkpoint.get("epoch", None)
    else:
        model.load_state_dict(checkpoint)
        best_top1 = None
        epoch = None

    model.to(device)
    model.eval()

    return model, best_top1, epoch


def get_last_linear_layer(model):
    last_linear = None

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module

    if last_linear is None:
        raise RuntimeError("Model içinde Linear sınıflandırma katmanı bulunamadı.")

    return last_linear


@torch.no_grad()
def extract_split_features(model, dataloader, device, split_name):
    all_features = []
    all_labels = []
    all_class_names = []

    last_linear = get_last_linear_layer(model)

    for images, labels, class_names in tqdm(
        dataloader,
        desc=f"{split_name} özellik çıkarımı"
    ):
        images = images.to(device)

        captured_features = {}

        def hook_fn(module, input, output):
            captured_features["features"] = input[0].detach()

        hook = last_linear.register_forward_hook(hook_fn)

        _ = model(images)

        hook.remove()

        features = captured_features["features"]

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
        all_class_names.extend(list(class_names))

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    class_names_np = np.array(all_class_names)

    return features_np, labels_np, class_names_np


def main():
    start_time = time.time()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("Intel Image Classification Derin Özellik Çıkarımı")
    print("=" * 70)

    device = get_device()

    print(f"Kullanılan cihaz: {device}")
    print(f"Model dosyası: {MODEL_PATH}")
    print(f"Giriş boyutu: {INPUT_SIZE} x {INPUT_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Özellik vektörü boyutu: {FEATURE_DIM}")

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = IntelFeatureDataset(TRAIN_CSV, transform=transform)
    test_dataset = IntelFeatureDataset(TEST_CSV, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\nVeri seti bilgisi:")
    print(f"Train görüntü sayısı: {len(train_dataset)}")
    print(f"Test görüntü sayısı: {len(test_dataset)}")
    print(f"Train sınıf sayısı: {len(train_dataset.class_names)}")
    print(f"Test sınıf sayısı: {len(test_dataset.class_names)}")
    print(f"Sınıflar: {train_dataset.class_names}")

    model, best_top1, epoch = load_model(device)

    print("\nKaynak model bilgisi:")
    if epoch is not None:
        print(f"Checkpoint epoch: {epoch}")
    if best_top1 is not None:
        print(f"Stable ImageNet-1K best validation Top-1: {best_top1:.2f}%")
    print("Model başarıyla yüklendi.")

    print("\nÖzellik çıkarımı başlıyor...")

    train_features, train_labels, train_class_names = extract_split_features(
        model=model,
        dataloader=train_loader,
        device=device,
        split_name="Train"
    )

    test_features, test_labels, test_class_names = extract_split_features(
        model=model,
        dataloader=test_loader,
        device=device,
        split_name="Test"
    )

    np.save(RESULTS_DIR / "intel_train_features.npy", train_features)
    np.save(RESULTS_DIR / "intel_train_labels.npy", train_labels)
    np.save(RESULTS_DIR / "intel_train_class_names.npy", train_class_names)

    np.save(RESULTS_DIR / "intel_test_features.npy", test_features)
    np.save(RESULTS_DIR / "intel_test_labels.npy", test_labels)
    np.save(RESULTS_DIR / "intel_test_class_names.npy", test_class_names)

    class_map = pd.DataFrame({
        "class_name": train_dataset.class_names,
        "label": list(range(len(train_dataset.class_names)))
    })

    class_map.to_csv(RESULTS_DIR / "intel_feature_label_map.csv", index=False)

    elapsed = time.time() - start_time

    summary_path = RESULTS_DIR / "intel_feature_extraction_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Intel Image Classification Derin Özellik Çıkarımı Özeti\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model dosyası: {MODEL_PATH}\n")
        f.write(f"Giriş boyutu: {INPUT_SIZE} x {INPUT_SIZE}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Özellik vektörü boyutu: {FEATURE_DIM}\n")
        f.write(f"Train özellik matrisi: {train_features.shape}\n")
        f.write(f"Test özellik matrisi: {test_features.shape}\n")
        f.write(f"Train label boyutu: {train_labels.shape}\n")
        f.write(f"Test label boyutu: {test_labels.shape}\n")
        f.write(f"Train sınıf sayısı: {len(train_dataset.class_names)}\n")
        f.write(f"Test sınıf sayısı: {len(test_dataset.class_names)}\n")
        f.write(f"Toplam süre: {elapsed / 60:.2f} dakika\n")

    print("\nÖzellik çıkarımı tamamlandı.")
    print(f"Train özellik matrisi: {train_features.shape}")
    print(f"Test özellik matrisi: {test_features.shape}")
    print(f"Train label boyutu: {train_labels.shape}")
    print(f"Test label boyutu: {test_labels.shape}")
    print(f"Label map: {RESULTS_DIR / 'intel_feature_label_map.csv'}")
    print(f"Özet dosyası: {summary_path}")
    print(f"Toplam süre: {elapsed / 60:.2f} dakika")


if __name__ == "__main__":
    main()