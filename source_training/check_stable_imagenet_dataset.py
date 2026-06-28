from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path("data/stable_imagenet_1k/raw")
RESULTS_DIR = Path("results/stable_imagenet")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_image_files(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def main():
    print("=" * 70)
    print("Stable ImageNet-1K Dataset Kontrolü")
    print("=" * 70)
    print(f"Veri klasörü: {DATA_DIR.resolve()}")

    all_images = find_image_files(DATA_DIR)
    print(f"Toplam görüntü dosyası: {len(all_images)}")

    if len(all_images) == 0:
        print("HATA: Görüntü dosyası bulunamadı. DATA_DIR yolunu kontrol et.")
        return

    class_counts = {}

    for img_path in all_images:
        class_name = img_path.parent.name
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    df = pd.DataFrame(
        [{"class_name": class_name, "image_count": count}
         for class_name, count in sorted(class_counts.items())]
    )

    output_csv = RESULTS_DIR / "stable_imagenet_class_distribution.csv"
    df.to_csv(output_csv, index=False)

    print(f"Sınıf klasörü sayısı: {len(df)}")
    print(f"Minimum sınıf görüntü sayısı: {df['image_count'].min()}")
    print(f"Maksimum sınıf görüntü sayısı: {df['image_count'].max()}")
    print(f"Ortalama sınıf görüntü sayısı: {df['image_count'].mean():.2f}")

    print("\nİlk 10 sınıf:")
    print(df.head(10).to_string(index=False))

    print("\nSon 10 sınıf:")
    print(df.tail(10).to_string(index=False))

    corrupted = []

    for img_path in tqdm(all_images, desc="Bozuk görüntü kontrolü"):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted.append(str(img_path))

    corrupted_csv = RESULTS_DIR / "stable_imagenet_corrupted_images.csv"
    pd.DataFrame({"filepath": corrupted}).to_csv(corrupted_csv, index=False)

    summary_txt = RESULTS_DIR / "stable_imagenet_dataset_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Stable ImageNet-1K Dataset Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Data directory: {DATA_DIR.resolve()}\n")
        f.write(f"Total images: {len(all_images)}\n")
        f.write(f"Number of class folders: {len(df)}\n")
        f.write(f"Minimum images per class: {df['image_count'].min()}\n")
        f.write(f"Maximum images per class: {df['image_count'].max()}\n")
        f.write(f"Average images per class: {df['image_count'].mean():.2f}\n")
        f.write(f"Corrupted images: {len(corrupted)}\n")

    print("\nKontrol tamamlandı.")
    print(f"Sınıf dağılımı kaydedildi: {output_csv}")
    print(f"Bozuk görüntü listesi kaydedildi: {corrupted_csv}")
    print(f"Özet dosyası kaydedildi: {summary_txt}")
    print(f"Bozuk görüntü sayısı: {len(corrupted)}")


if __name__ == "__main__":
    main()