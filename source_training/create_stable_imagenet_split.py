from pathlib import Path
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path("data/stable_imagenet_1k/raw/imagenet1k")
SPLIT_DIR = Path("data_splits")
RESULTS_DIR = Path("results/stable_imagenet")

TRAIN_RATIO = 0.80
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_dataset():
    class_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])

    records = []
    label_map = {}

    for label, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        label_map[class_name] = label

        image_paths = sorted([p for p in class_dir.iterdir() if p.is_file() and is_image_file(p)])

        for image_path in image_paths:
            records.append({
                "filepath": str(image_path),
                "label": label,
                "class_name": class_name
            })

    return records, label_map


def create_split(records):
    random.seed(SEED)

    df = pd.DataFrame(records)

    train_records = []
    val_records = []

    for class_name, group in df.groupby("class_name"):
        group_records = group.to_dict("records")
        random.shuffle(group_records)

        train_count = int(len(group_records) * TRAIN_RATIO)

        train_records.extend(group_records[:train_count])
        val_records.extend(group_records[train_count:])

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    train_df["split"] = "train"
    val_df["split"] = "val"

    train_df = train_df.sort_values(["label", "filepath"]).reset_index(drop=True)
    val_df = val_df.sort_values(["label", "filepath"]).reset_index(drop=True)

    return train_df, val_df


def create_summary(train_df, val_df):
    train_counts = train_df.groupby("class_name").size().reset_index(name="train_count")
    val_counts = val_df.groupby("class_name").size().reset_index(name="val_count")

    summary = train_counts.merge(val_counts, on="class_name", how="outer").fillna(0)
    summary["total_count"] = summary["train_count"] + summary["val_count"]

    summary = summary.sort_values("class_name").reset_index(drop=True)

    return summary


def verify_images(df, split_name):
    corrupted = []

    for path in tqdm(df["filepath"], desc=f"{split_name} görüntü doğrulama"):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrupted.append(path)

    return corrupted


def main():
    print("=" * 70)
    print("Stable ImageNet-1K Train/Validation Ayrımı")
    print("=" * 70)

    print(f"Veri klasörü: {DATA_DIR.resolve()}")
    print(f"Train oranı: {TRAIN_RATIO}")
    print(f"Validation oranı: {1 - TRAIN_RATIO:.2f}")
    print(f"Random seed: {SEED}")

    records, label_map = collect_dataset()

    if len(records) == 0:
        print("HATA: Görüntü bulunamadı. DATA_DIR yolunu kontrol et.")
        return

    print(f"Toplam görüntü sayısı: {len(records)}")
    print(f"Toplam sınıf sayısı: {len(label_map)}")

    train_df, val_df = create_split(records)

    train_csv = SPLIT_DIR / "stable_imagenet_train.csv"
    val_csv = SPLIT_DIR / "stable_imagenet_val.csv"
    label_map_csv = SPLIT_DIR / "stable_imagenet_label_map.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    pd.DataFrame(
        [{"class_name": class_name, "label": label} for class_name, label in label_map.items()]
    ).sort_values("label").to_csv(label_map_csv, index=False)

    summary = create_summary(train_df, val_df)
    summary_csv = RESULTS_DIR / "stable_imagenet_split_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\nSplit dosyaları kaydedildi:")
    print(f"Train CSV: {train_csv}")
    print(f"Validation CSV: {val_csv}")
    print(f"Label map CSV: {label_map_csv}")
    print(f"Split özeti: {summary_csv}")

    print("\nSplit özeti:")
    print(f"Train görüntü sayısı: {len(train_df)}")
    print(f"Validation görüntü sayısı: {len(val_df)}")
    print(f"Toplam görüntü sayısı: {len(train_df) + len(val_df)}")
    print(f"Train sınıf sayısı: {train_df['class_name'].nunique()}")
    print(f"Validation sınıf sayısı: {val_df['class_name'].nunique()}")

    print("\nİlk 10 split özeti:")
    print(summary.head(10).to_string(index=False))

    print("\nFarklı toplam görüntü sayısına sahip sınıflar:")
    print(summary[summary["total_count"] != 100].to_string(index=False))

    summary_txt = RESULTS_DIR / "stable_imagenet_split_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Stable ImageNet-1K Train/Validation Split Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Data directory: {DATA_DIR.resolve()}\n")
        f.write(f"Random seed: {SEED}\n")
        f.write(f"Train ratio: {TRAIN_RATIO}\n")
        f.write(f"Validation ratio: {1 - TRAIN_RATIO:.2f}\n")
        f.write(f"Total images: {len(train_df) + len(val_df)}\n")
        f.write(f"Train images: {len(train_df)}\n")
        f.write(f"Validation images: {len(val_df)}\n")
        f.write(f"Number of classes: {train_df['class_name'].nunique()}\n")
        f.write("\nClasses with total_count != 100:\n")
        f.write(summary[summary["total_count"] != 100].to_string(index=False))

    print(f"\nMetin özet dosyası kaydedildi: {summary_txt}")
    print("\nTrain/validation ayrımı başarıyla oluşturuldu.")


if __name__ == "__main__":
    main()