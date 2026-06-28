from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


FEATURE_DIR = Path("results/intel_features")
RESULTS_DIR = Path("results/intel_feature_selection")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K_FEATURES = 128
SEED = 42


def main():
    start_time = time.time()

    print("=" * 70)
    print("Intel Derin Özellik Seçimi")
    print("=" * 70)

    train_features = np.load(FEATURE_DIR / "intel_train_features.npy")
    test_features = np.load(FEATURE_DIR / "intel_test_features.npy")
    train_labels = np.load(FEATURE_DIR / "intel_train_labels.npy")
    test_labels = np.load(FEATURE_DIR / "intel_test_labels.npy")

    print(f"Train özellik matrisi: {train_features.shape}")
    print(f"Test özellik matrisi: {test_features.shape}")
    print(f"Train label boyutu: {train_labels.shape}")
    print(f"Test label boyutu: {test_labels.shape}")

    print("\nNormalizasyon işlemi uygulanıyor...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    print("Özellik seçimi uygulanıyor...")
    selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
    train_selected = selector.fit_transform(train_scaled, train_labels)
    test_selected = selector.transform(test_scaled)

    selected_indices = selector.get_support(indices=True)
    selected_scores = selector.scores_[selected_indices]

    np.save(RESULTS_DIR / "intel_train_features_selected.npy", train_selected)
    np.save(RESULTS_DIR / "intel_test_features_selected.npy", test_selected)
    np.save(RESULTS_DIR / "intel_train_labels.npy", train_labels)
    np.save(RESULTS_DIR / "intel_test_labels.npy", test_labels)
    np.save(RESULTS_DIR / "selected_feature_indices.npy", selected_indices)

    selected_df = pd.DataFrame({
        "feature_index": selected_indices,
        "anova_f_score": selected_scores
    }).sort_values("anova_f_score", ascending=False)

    selected_df.to_csv(
        RESULTS_DIR / "selected_feature_scores.csv",
        index=False
    )

    elapsed = time.time() - start_time

    summary_path = RESULTS_DIR / "feature_selection_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Intel Derin Özellik Seçimi Özeti\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Orijinal özellik boyutu: {train_features.shape[1]}\n")
        f.write(f"Seçilen özellik sayısı: {K_FEATURES}\n")
        f.write(f"Train seçilmiş özellik matrisi: {train_selected.shape}\n")
        f.write(f"Test seçilmiş özellik matrisi: {test_selected.shape}\n")
        f.write(f"Normalizasyon: StandardScaler\n")
        f.write(f"Özellik seçimi yöntemi: SelectKBest + ANOVA F-score\n")
        f.write(f"Toplam süre: {elapsed:.2f} saniye\n")

    print("\nÖzellik seçimi tamamlandı.")
    print(f"Orijinal özellik boyutu: {train_features.shape[1]}")
    print(f"Seçilen özellik sayısı: {K_FEATURES}")
    print(f"Train seçilmiş özellik matrisi: {train_selected.shape}")
    print(f"Test seçilmiş özellik matrisi: {test_selected.shape}")
    print("Normalizasyon: StandardScaler")
    print("Özellik seçimi yöntemi: SelectKBest + ANOVA F-score")
    print(f"Seçilen feature indeksleri: {RESULTS_DIR / 'selected_feature_indices.npy'}")
    print(f"Feature skorları: {RESULTS_DIR / 'selected_feature_scores.csv'}")
    print(f"Özet dosyası: {summary_path}")
    print(f"Toplam süre: {elapsed:.2f} saniye")


if __name__ == "__main__":
    main()