from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


DATA_DIR = Path("data/intel_image_classification/raw")

RESULTS_DIR = Path("results/intel_dataset")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": DATA_DIR / "seg_train" / "seg_train",
    "test": DATA_DIR / "seg_test" / "seg_test"
}


def analyze_split(split_name, split_path):
    rows = []

    class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        image_files = []

        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(class_dir.glob(ext))

        image_files = sorted(image_files)

        for img_path in tqdm(
            image_files,
            desc=f"{split_name} - {class_dir.name}",
            leave=False
        ):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                rows.append({
                    "filepath": str(img_path),
                    "class_name": class_dir.name,
                    "width": width,
                    "height": height,
                    "split": split_name
                })

            except Exception:
                pass

    return pd.DataFrame(rows)


def main():

    print("=" * 60)
    print("Intel Image Classification Dataset Kontrolü")
    print("=" * 60)

    all_df = []

    for split_name, split_path in SPLITS.items():

        print(f"\n{split_name.upper()} klasörü okunuyor...")
        df = analyze_split(split_name, split_path)

        csv_path = RESULTS_DIR / f"intel_{split_name}.csv"
        df.to_csv(csv_path, index=False)

        print(f"Görüntü sayısı : {len(df)}")
        print(f"Sınıf sayısı   : {df['class_name'].nunique()}")

        print("\nSınıf dağılımı:")

        class_counts = (
            df["class_name"]
            .value_counts()
            .sort_index()
        )

        print(class_counts)

        summary = class_counts.reset_index()
        summary.columns = ["class_name", "image_count"]

        summary.to_csv(
            RESULTS_DIR / f"intel_{split_name}_class_distribution.csv",
            index=False
        )

        all_df.append(df)

    full_df = pd.concat(all_df, ignore_index=True)

    print("\n")
    print("=" * 60)
    print("Genel Özet")
    print("=" * 60)

    print(f"Toplam görüntü : {len(full_df)}")
    print(f"Toplam sınıf   : {full_df['class_name'].nunique()}")

    summary_txt = RESULTS_DIR / "intel_dataset_summary.txt"

    with open(summary_txt, "w", encoding="utf-8") as f:

        f.write("Intel Image Classification Dataset Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Toplam görüntü : {len(full_df)}\n")
        f.write(f"Toplam sınıf   : {full_df['class_name'].nunique()}\n\n")

        for split in ["train", "test"]:

            df = full_df[full_df["split"] == split]

            f.write(f"{split.upper()}\n")
            f.write("-" * 20 + "\n")

            counts = (
                df["class_name"]
                .value_counts()
                .sort_index()
            )

            f.write(counts.to_string())
            f.write("\n\n")

    print(f"\nÖzet kaydedildi : {summary_txt}")
    print("Kontrol tamamlandı.")


if __name__ == "__main__":
    main()