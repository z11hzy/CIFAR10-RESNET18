import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def load_logs(results_dir="results"):
    """
    读取每个优化器对应的 csv 日志。
    """
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

    logs = {}

    for file in csv_files:
        filename = os.path.basename(file)

        if filename == "summary.csv":
            continue

        optimizer_name = filename.replace(".csv", "")
        df = pd.read_csv(file)
        logs[optimizer_name] = df

    return logs


def plot_train_loss(logs, save_path="figures/train_loss_curve.png"):
    plt.figure(figsize=(8, 6))

    for optimizer_name, df in logs.items():
        plt.plot(df["epoch"], df["train_loss"], label=optimizer_name)

    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_test_loss(logs, save_path="figures/test_loss_curve.png"):
    plt.figure(figsize=(8, 6))

    for optimizer_name, df in logs.items():
        plt.plot(df["epoch"], df["test_loss"], label=optimizer_name)

    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_train_acc(logs, save_path="figures/train_acc_curve.png"):
    plt.figure(figsize=(8, 6))

    for optimizer_name, df in logs.items():
        plt.plot(df["epoch"], df["train_acc"], label=optimizer_name)

    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy (%)")
    plt.title("Training Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_test_acc(logs, save_path="figures/test_acc_curve.png"):
    plt.figure(figsize=(8, 6))

    for optimizer_name, df in logs.items():
        plt.plot(df["epoch"], df["test_acc"], label=optimizer_name)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_best_acc_bar(logs, save_path="figures/best_acc_bar.png"):
    optimizers = []
    best_accs = []

    for optimizer_name, df in logs.items():
        optimizers.append(optimizer_name)
        best_accs.append(df["test_acc"].max())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(optimizers, best_accs)

    for bar, acc in zip(bars, best_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{acc:.2f}",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Optimizer")
    plt.ylabel("Best Test Accuracy (%)")
    plt.title("Best Test Accuracy Comparison")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_time_bar(logs, save_path="figures/time_bar.png"):
    optimizers = []
    total_times = []

    for optimizer_name, df in logs.items():
        optimizers.append(optimizer_name)
        total_times.append(df["total_time"].iloc[-1])

    plt.figure(figsize=(8, 6))
    bars = plt.bar(optimizers, total_times)

    for bar, t in zip(bars, total_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.1f}s",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Optimizer")
    plt.ylabel("Total Training Time (s)")
    plt.title("Training Time Comparison")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    os.makedirs("figures", exist_ok=True)

    logs = load_logs("results")

    if len(logs) == 0:
        print("No log files found in results/. Please run main.py first.")
        return

    plot_train_loss(logs)
    plot_test_loss(logs)
    plot_train_acc(logs)
    plot_test_acc(logs)
    plot_best_acc_bar(logs)
    plot_time_bar(logs)

    print("All figures saved to figures/.")


if __name__ == "__main__":
    main()