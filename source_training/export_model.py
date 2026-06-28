from pathlib import Path
import argparse
import yaml

import torch

from model_architecture import MertNetS


DEFAULT_CONFIG_PATH = Path("source_training/config_stable_imagenet.yaml")
DEFAULT_CHECKPOINT_PATH = Path("exported_model/mertnet_s_best.pth")
DEFAULT_EXPORT_DIR = Path("exported_model")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_model(checkpoint_path, config, device):
    model = MertNetS(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"]
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def export_torchscript(model, export_path, input_size, device):
    example_input = torch.randn(1, 3, input_size, input_size).to(device)

    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(str(export_path))

    return export_path


def export_onnx(model, export_path, input_size, device):
    example_input = torch.randn(1, 3, input_size, input_size).to(device)

    torch.onnx.export(
        model,
        example_input,
        str(export_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=17
    )

    return export_path


def test_torchscript_model(torchscript_path, input_size, device):
    loaded_model = torch.jit.load(str(torchscript_path), map_location=device)
    loaded_model.eval()

    example_input = torch.randn(1, 3, input_size, input_size).to(device)

    with torch.no_grad():
        output = loaded_model(example_input)

    return tuple(output.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--export_dir", type=str, default=str(DEFAULT_EXPORT_DIR))
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    device = get_device()

    input_size = config["data"]["input_size"]

    print("=" * 70)
    print("MertNet-S Model Export İşlemi")
    print("=" * 70)
    print(f"Config dosyası: {config_path}")
    print(f"Checkpoint dosyası: {checkpoint_path}")
    print(f"Export klasörü: {export_dir}")
    print(f"Kullanılan cihaz: {device}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {checkpoint_path}. "
            "Önce modeli eğitip mertnet_s_best.pth dosyasını oluşturmalısın."
        )

    model, checkpoint = load_trained_model(checkpoint_path, config, device)

    print("\nCheckpoint başarıyla yüklendi.")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'bilinmiyor')}")
    print(f"Best validation Top-1: {checkpoint.get('best_val_top1', 'bilinmiyor')}")

    torchscript_path = export_dir / "mertnet_s_torchscript.pt"
    onnx_path = export_dir / "mertnet_s.onnx"

    export_torchscript(
        model=model,
        export_path=torchscript_path,
        input_size=input_size,
        device=device
    )

    print(f"\nTorchScript model export edildi: {torchscript_path}")

    try:
        export_onnx(
            model=model,
            export_path=onnx_path,
            input_size=input_size,
            device=device
        )
        print(f"ONNX model export edildi: {onnx_path}")
    except Exception as e:
        print(f"ONNX export sırasında hata oluştu: {e}")
        print("TorchScript export başarılı olduğu için export adımı devam ediyor.")

    output_shape = test_torchscript_model(
        torchscript_path=torchscript_path,
        input_size=input_size,
        device=device
    )

    output_txt = export_dir / "export_test_output.txt"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("MertNet-S Export Test Output\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint path: {checkpoint_path}\n")
        f.write(f"TorchScript path: {torchscript_path}\n")
        f.write(f"ONNX path: {onnx_path}\n")
        f.write(f"Input size: 1 x 3 x {input_size} x {input_size}\n")
        f.write(f"TorchScript output shape: {output_shape}\n")

    print("\nExport edilen TorchScript model tekrar yüklendi.")
    print(f"Örnek çıktı boyutu: {output_shape}")
    print(f"Export test çıktısı kaydedildi: {output_txt}")
    print("Model export işlemi tamamlandı.")


if __name__ == "__main__":
    main()