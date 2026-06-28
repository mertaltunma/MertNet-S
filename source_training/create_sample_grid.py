from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_CSV = Path("data_splits/stable_imagenet_train.csv")
RESULTS_DIR = Path("results/stable_imagenet")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "stable_imagenet_sample_grid.png"
SEED = 42
N_IMAGES = 16

df = pd.read_csv(
    TRAIN_CSV,
    header=0,
    names=["filepath", "label", "class_name", "split"]
)

print("Kullanılan CSV sütunları:", list(df.columns))
print("Toplam train görüntüsü:", len(df))

sample_df = (
    df.groupby("class_name", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=SEED))
    .sample(N_IMAGES, random_state=SEED)
    .reset_index(drop=True)
)

plt.figure(figsize=(10, 10))

for i in range(len(sample_df)):
    filepath = sample_df.iloc[i, 0]
    class_name = sample_df.iloc[i, 2]

    img = Image.open(filepath).convert("RGB")
    plt.subplot(4, 4, i + 1)
    plt.imshow(img)
    plt.title(str(class_name)[:28], fontsize=7)
    plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.close()

print(f"Örnek görüntü grid'i kaydedildi: {OUTPUT_PATH}")
