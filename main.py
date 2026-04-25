import os
import csv
import time
import argparse

import torch
import torch.nn as nn

from utils import set_seed, make_dirs, get_device
from data import get_cifar10_loaders
from models import get_model
from optimizers import get_optimizer, get_scheduler
from engine import train_one_epoch, evaluate, get_current_lr


def train_with_optimizer(args, optimizer_name):
    """
    使用某一种优化器完成完整训练。
    """

    print("=" * 80)
    print(f"Start training with optimizer: {optimizer_name}")
    print("=" * 80)

    # 为了公平，每个优化器重新设置随机种子，并重新初始化模型
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    model = get_model(
        model_name=args.model,
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        lr=None,
        weight_decay=args.weight_decay
    )

    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs
    )

    log_path = os.path.join("results", f"{optimizer_name}.csv")
    checkpoint_path = os.path.join("checkpoints", f"{args.model}_{optimizer_name}_best.pth")

    best_acc = 0.0
    best_epoch = 0
    final_acc = 0.0

    start_time = time.time()

    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "optimizer",
            "epoch",
            "lr",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "epoch_time",
            "total_time"
        ])

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                epochs=args.epochs
            )

            test_loss, test_acc = evaluate(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device
            )

            if scheduler is not None:
                scheduler.step()

            current_lr = get_current_lr(optimizer)

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time

            final_acc = test_acc

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch

                torch.save({
                    "model_name": args.model,
                    "optimizer_name": optimizer_name,
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_names": class_names
                }, checkpoint_path)

            writer.writerow([
                optimizer_name,
                epoch,
                current_lr,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                epoch_time,
                total_time
            ])

            print(
                f"[{optimizer_name}] "
                f"Epoch [{epoch}/{args.epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Best Acc: {best_acc:.2f}%"
            )

    total_time = time.time() - start_time

    result = {
        "optimizer": optimizer_name,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_acc": final_acc,
        "total_time": total_time,
        "checkpoint": checkpoint_path
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"]
    )

    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["SGD", "SGDM", "Adam", "AdamW"],
        help="Example: --optimizers SGD SGDM Adam AdamW RMSprop"
    )

    args = parser.parse_args()

    make_dirs()

    summary_path = os.path.join("results", "summary.csv")

    all_results = []

    for opt_name in args.optimizers:
        result = train_with_optimizer(args, opt_name)
        all_results.append(result)

    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "optimizer",
            "best_acc",
            "best_epoch",
            "final_acc",
            "total_time",
            "checkpoint"
        ])

        for result in all_results:
            writer.writerow([
                result["optimizer"],
                result["best_acc"],
                result["best_epoch"],
                result["final_acc"],
                result["total_time"],
                result["checkpoint"]
            ])

    print("\nTraining finished.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()