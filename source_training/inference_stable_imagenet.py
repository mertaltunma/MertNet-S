from pathlib import Path
import argparse
import yaml
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms

from model_architecture import MertNetS


DEFAULT_CONFIG_PATH = Path("source_training/config_stable_imagenet.yaml")
DEFAULT_MODEL_PATH = Path("exported_model/mertnet_s_best.pth")
DEFAULT_LABEL_MAP = Path("data_splits/stable_imagenet_label_map.csv")
DEFAULT_IMAGE_PATH = None


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_label_map(label_map_path):
    df = pd.read_csv(label_map_path)
    df.columns = [col.strip() for col in df.columns]

    label_to_class = {}

    for _, row in df.iterrows():
        label_to_class[int(row["label"])] = row["class_name"]

    return label_to_class


def get_inference_transform(input_size):
    return transforms.Compose([
        transforms.Resize(176),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(model_path, config, device):
    model = MertNetS(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"]
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def predict_image(model, image_path, transform, label_to_class, device, topk=5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    top_probs, top_indices = probabilities.topk(topk, dim=1)

    predictions = []

    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_id = int(idx.item())
        class_name = label_to_class.get(class_id, "unknown")

        predictions.append({
            "class_id": class_id,
            "class_name": class_name,
            "probability": float(prob.item())
        })

    features = model.forward_features(image_tensor)

    return predictions, tuple(features.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--label_map", type=str, default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config)
    model_path = Path(args.model)
    label_map_path = Path(args.label_map)
    image_path = Path(args.image)

    config = load_config(config_path)
    device = get_device()

    print("=" * 70)
    print("MertNet-S Stable ImageNet-1K Inference Testi")
    print("=" * 70)
    print(f"Config dosyası: {config_path}")
    print(f"Model dosyası: {model_path}")
    print(f"Label map dosyası: {label_map_path}")
    print(f"Görüntü dosyası: {image_path}")
    print(f"Kullanılan cihaz: {device}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")

    label_to_class = load_label_map(label_map_path)
    transform = get_inference_transform(config["data"]["input_size"])

    model, checkpoint = load_model(model_path, config, device)

    print("\nModel başarıyla yüklendi.")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'bilinmiyor')}")
    print(f"Checkpoint best_val_top1: {checkpoint.get('best_val_top1', 'bilinmiyor')}")

    predictions, feature_shape = predict_image(
        model=model,
        image_path=image_path,
        transform=transform,
        label_to_class=label_to_class,
        device=device,
        topk=args.topk
    )

    print("\nTahmin sonucu:")
    for rank, pred in enumerate(predictions, start=1):
        print(
            f"Top-{rank}: "
            f"class_id={pred['class_id']} | "
            f"class_name={pred['class_name']} | "
            f"probability={pred['probability']:.6f}"
        )

    print(f"\nÇıkarılan özellik vektörü boyutu: {feature_shape}")
    print("Inference testi tamamlandı.")


if __name__ == "__main__":
    main()