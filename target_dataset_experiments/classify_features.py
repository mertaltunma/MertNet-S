from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


FEATURE_DIR = Path("results/intel_feature_selection")
LABEL_MAP_PATH = Path("results/intel_features/intel_feature_label_map.csv")

RESULTS_DIR = Path("results/intel_classification")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def evaluate_model(model_name, model, x_train, y_train, x_test, y_test, class_names):
    start_time = time.time()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    elapsed = time.time() - start_time

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    report_path = RESULTS_DIR / f"{model_name}_classification_report.txt"
    cm_path = RESULTS_DIR / f"{model_name}_confusion_matrix.csv"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    ).to_csv(cm_path)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "time_seconds": elapsed,
        "report_path": str(report_path),
        "confusion_matrix_path": str(cm_path)
    }


def plot_best_confusion_matrix(best_model_name, class_names):
    cm_path = RESULTS_DIR / f"{best_model_name}_confusion_matrix.csv"
    cm_df = pd.read_csv(cm_path, index_col=0)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_df.values)
    plt.title(f"{best_model_name} Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, str(cm_df.values[i, j]), ha="center", va="center")

    plt.tight_layout()
    output_path = RESULTS_DIR / f"{best_model_name}_confusion_matrix.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def main():
    print("=" * 70)
    print("Intel Derin Özelliklerle Klasik Sınıflandırma")
    print("=" * 70)

    x_train = np.load(FEATURE_DIR / "intel_train_features_selected.npy")
    x_test = np.load(FEATURE_DIR / "intel_test_features_selected.npy")
    y_train = np.load(FEATURE_DIR / "intel_train_labels.npy")
    y_test = np.load(FEATURE_DIR / "intel_test_labels.npy")

    label_map = pd.read_csv(LABEL_MAP_PATH)
    class_names = label_map.sort_values("label")["class_name"].tolist()

    print(f"Train özellik matrisi: {x_train.shape}")
    print(f"Test özellik matrisi: {x_test.shape}")
    print(f"Train label boyutu: {y_train.shape}")
    print(f"Test label boyutu: {y_test.shape}")
    print(f"Sınıflar: {class_names}")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            n_jobs=-1
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1
        )
    }

    results = []

    print("\nSınıflandırma deneyleri başlıyor...")

    for model_name, model in models.items():
        print(f"\nModel eğitiliyor: {model_name}")

        result = evaluate_model(
            model_name=model_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            class_names=class_names
        )

        results.append(result)

        print(
            f"{model_name} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Precision: {result['precision_macro']:.4f} | "
            f"Recall: {result['recall_macro']:.4f} | "
            f"F1: {result['f1_macro']:.4f} | "
            f"Süre: {result['time_seconds']:.2f} sn"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1_macro", ascending=False)

    results_csv = RESULTS_DIR / "classification_results.csv"
    results_json = RESULTS_DIR / "classification_results.json"
    summary_txt = RESULTS_DIR / "classification_summary.txt"

    results_df.to_csv(results_csv, index=False)

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    best_model_name = results_df.iloc[0]["model"]
    best_cm_png = plot_best_confusion_matrix(best_model_name, class_names)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Intel Derin Özelliklerle Klasik Sınıflandırma Özeti\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train özellik matrisi: {x_train.shape}\n")
        f.write(f"Test özellik matrisi: {x_test.shape}\n")
        f.write("Kullanılan özellik sayısı: 128\n")
        f.write(f"Sınıflar: {class_names}\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        f.write(f"En iyi model: {best_model_name}\n")
        f.write(f"En iyi model confusion matrix görseli: {best_cm_png}\n")

    print("\nSınıflandırma tamamlandı.")
    print("\nSonuç tablosu:")
    print(results_df[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "time_seconds"]])

    print(f"\nSonuç CSV: {results_csv}")
    print(f"Sonuç JSON: {results_json}")
    print(f"Özet dosyası: {summary_txt}")
    print(f"En iyi model: {best_model_name}")
    print(f"Confusion matrix görseli: {best_cm_png}")


if __name__ == "__main__":
    main()