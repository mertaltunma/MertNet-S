from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/intel_image_classification/raw/seg_train/seg_train")
RESULTS_DIR = Path("results/intel_dataset")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "intel_sample_grid.png"

SEED = 42
random.seed(SEED)

image_extensions = [".jpg", ".jpeg", ".png"]

class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

sample_items = []

for class_dir in class_dirs:
    images = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    selected = random.sample(images, 2)

    for img_path in selected:
        sample_items.append((img_path, class_dir.name))

random.shuffle(sample_items)

print("Intel Image Classification örnek görüntü gridi")
print(f"Sınıf sayısı: {len(class_dirs)}")
print(f"Toplam seçilen örnek görüntü: {len(sample_items)}")

plt.figure(figsize=(10, 8))

for i, (img_path, class_name) in enumerate(sample_items):
    img = Image.open(img_path).convert("RGB")

    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(class_name, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.close()

print(f"Intel örnek görüntü gridi kaydedildi: {OUTPUT_PATH}")