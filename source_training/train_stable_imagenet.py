from pathlib import Path
import random
import time
import argparse
import platform
import yaml
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from model_architecture import MertNetS, count_parameters
from dataset_stable_imagenet import create_dataloaders
from utils_metrics import AverageMeter, accuracy_topk


DEFAULT_CONFIG_PATH = Path("source_training/config_stable_imagenet.yaml")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, max_batches=None):
    model.train()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        top1, top5 = accuracy_topk(outputs, labels, topk=(1, 5))

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1.item(), batch_size)
        top5_meter.update(top5.item(), batch_size)

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Train Loss: {loss_meter.avg:.4f} "
                f"Top-1: {top1_meter.avg:.2f}% "
                f"Top-5: {top5_meter.avg:.2f}%"
            )

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, max_batches=None):
    model.eval()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        top1, top5 = accuracy_topk(outputs, labels, topk=(1, 5))

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1.item(), batch_size)
        top5_meter.update(top5.item(), batch_size)

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    print(
        f"Epoch [{epoch}] Validation "
        f"Loss: {loss_meter.avg:.4f} "
        f"Top-1: {top1_meter.avg:.2f}% "
        f"Top-5: {top5_meter.avg:.2f}%"
    )

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_top1, config):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_top1": best_val_top1,
        "config": config
    }
    torch.save(checkpoint, path)


def save_training_curves(log_df, results_dir):
    loss_path = results_dir / "stable_imagenet_loss_curve.png"
    acc_path = results_dir / "stable_imagenet_accuracy_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stable ImageNet-1K Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["train_top1"], label="Train Top-1")
    plt.plot(log_df["epoch"], log_df["val_top1"], label="Validation Top-1")
    plt.plot(log_df["epoch"], log_df["train_top5"], label="Train Top-5")
    plt.plot(log_df["epoch"], log_df["val_top5"], label="Validation Top-5")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Stable ImageNet-1K Top-1 and Top-5 Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    return loss_path, acc_path


def save_system_info(results_dir, device):
    info_path = results_dir / "training_system_info.txt"

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Training System Information\n")
        f.write("=" * 40 + "\n")
        f.write(f"Operating system: {platform.platform()}\n")
        f.write(f"Python version: {platform.python_version()}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"Device: {device}\n")
        f.write(f"MPS available: {torch.backends.mps.is_available()}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")

    return info_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    seed = config["project"]["seed"]
    set_seed(seed)

    device = get_device()

    results_dir = Path(config["outputs"]["results_dir"])
    model_dir = Path(config["outputs"]["model_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / config["outputs"]["best_model_name"]
    last_model_path = model_dir / config["outputs"]["last_model_name"]
    log_csv_path = results_dir / "stable_imagenet_training_log.csv"

    print("=" * 70)
    print("MertNet-S Stable ImageNet-1K Eğitimi")
    print("=" * 70)
    print(f"Config dosyası: {config_path}")
    print(f"Random seed: {seed}")
    print(f"Kullanılan cihaz: {device}")
    print(f"PyTorch sürümü: {torch.__version__}")
    print(f"Smoke test: {args.smoke_test}")

    model = MertNetS(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    print("\nModel bilgisi:")
    print(f"Model adı: {config['model']['name']}")
    print(f"Giriş boyutu: {config['data']['input_size']}")
    print(f"Sınıf sayısı: {config['data']['num_classes']}")
    print(f"Toplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"]
    )

    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        train_csv=config["data"]["train_csv"],
        val_csv=config["data"]["val_csv"],
        input_size=config["data"]["input_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"]
    )

    print("\nDataLoader bilgisi:")
    print(f"Train veri sayısı: {len(train_dataset)}")
    print(f"Validation veri sayısı: {len(val_dataset)}")
    print(f"Train batch sayısı: {len(train_loader)}")
    print(f"Validation batch sayısı: {len(val_loader)}")

    save_system_info(results_dir, device)

    if args.smoke_test:
        epochs = 1
        max_batches = 1
    else:
        epochs = config["training"]["epochs"]
        max_batches = None

    best_val_top1 = 0.0
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    logs = []

    start_time = time.time()

    print("\nEğitim başlıyor...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_top1, train_top5 = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=max_batches
        )

        val_loss, val_top1, val_top5 = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_batches=max_batches
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "train_top5": train_top5,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time
        }

        logs.append(log_row)
        log_df = pd.DataFrame(logs)
        log_df.to_csv(log_csv_path, index=False)

        save_checkpoint(
            path=last_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_top1=best_val_top1,
            config=config
        )

        print(
            f"Epoch [{epoch}/{epochs}] tamamlandı | "
            f"Train Loss: {train_loss:.4f} | Train Top-1: {train_top1:.2f}% | Train Top-5: {train_top5:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}% | "
            f"LR: {current_lr:.6f} | Süre: {epoch_time:.1f} sn"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            patience_counter = 0

            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_top1=best_val_top1,
                config=config
            )

            print(f"Yeni en iyi model kaydedildi: {best_model_path} | Best Val Top-1: {best_val_top1:.2f}%")
        else:
            patience_counter += 1
            print(f"Early stopping sayacı: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping tetiklendi. Eğitim durduruldu.")
            break

    total_time = time.time() - start_time

    log_df = pd.DataFrame(logs)
    log_df.to_csv(log_csv_path, index=False)

    loss_curve_path, acc_curve_path = save_training_curves(log_df, results_dir)

    summary_path = results_dir / "stable_imagenet_training_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stable ImageNet-1K Training Summary\n")
        f.write("=" * 45 + "\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Input size: {config['data']['input_size']}\n")
        f.write(f"Number of classes: {config['data']['num_classes']}\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs completed: {len(log_df)}\n")
        f.write(f"Best validation Top-1: {best_val_top1:.4f}\n")
        f.write(f"Total training time seconds: {total_time:.2f}\n")
        f.write(f"Best model path: {best_model_path}\n")
        f.write(f"Last model path: {last_model_path}\n")
        f.write(f"Training log path: {log_csv_path}\n")
        f.write(f"Loss curve path: {loss_curve_path}\n")
        f.write(f"Accuracy curve path: {acc_curve_path}\n")

    print("\nEğitim tamamlandı.")
    print(f"Toplam süre: {total_time / 60:.2f} dakika")
    print(f"En iyi validation Top-1: {best_val_top1:.2f}%")
    print(f"Training log: {log_csv_path}")
    print(f"Loss grafiği: {loss_curve_path}")
    print(f"Accuracy grafiği: {acc_curve_path}")
    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")
    print(f"Özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()