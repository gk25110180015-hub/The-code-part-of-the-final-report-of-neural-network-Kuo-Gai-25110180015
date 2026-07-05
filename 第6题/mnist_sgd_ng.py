"""Question 6: MNIST classification with SGD and Natural Gradient.

The implementation uses only NumPy plus Pillow for optional figures.  It first
tries to read standard MNIST IDX files from ``data/mnist``.  If the files are
not present, it falls back to a deterministic MNIST-like digit dataset so that
the complete training/comparison pipeline can still be executed offline.

Examples
--------
Run the default experiment:
    python mnist_sgd_ng.py

Use local MNIST IDX files:
    python mnist_sgd_ng.py --mnist-dir data/mnist --require-mnist

Reduce runtime:
    python mnist_sgd_ng.py --epochs 8 --train-size 2000 --test-size 500
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - figures are optional
    Image = None
    ImageDraw = None
    ImageFont = None


ArrayDict = dict[str, np.ndarray]


@dataclass
class Dataset:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    note: str


@dataclass
class TrainResult:
    optimizer: str
    model: "MLP"
    metrics: list[dict[str, float | int | str]]
    predictions: np.ndarray
    confusion: np.ndarray


def open_maybe_gzip(path: Path) -> BinaryIO:
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def read_idx_images(path: Path) -> np.ndarray:
    with open_maybe_gzip(path) as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file.")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float64) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    with open_maybe_gzip(path) as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file.")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data.reshape(count).astype(np.int64)


def find_mnist_files(root: Path) -> dict[str, Path] | None:
    candidates = {
        "train_images": ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"),
        "train_labels": ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"),
        "test_images": ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"),
        "test_labels": ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"),
    }
    found: dict[str, Path] = {}
    for key, names in candidates.items():
        for name in names:
            path = root / name
            if path.exists():
                found[key] = path
                break
        if key not in found:
            return None
    return found


def load_mnist_dataset(
    mnist_dir: Path,
    train_size: int,
    test_size: int,
    seed: int,
) -> Dataset | None:
    files = find_mnist_files(mnist_dir)
    if files is None:
        return None

    rng = np.random.default_rng(seed)
    x_train = read_idx_images(files["train_images"])
    y_train = read_idx_labels(files["train_labels"])
    x_test = read_idx_images(files["test_images"])
    y_test = read_idx_labels(files["test_labels"])

    train_idx = rng.permutation(len(y_train))[: min(train_size, len(y_train))]
    test_idx = rng.permutation(len(y_test))[: min(test_size, len(y_test))]

    return Dataset(
        name="MNIST IDX",
        x_train=x_train[train_idx],
        y_train=y_train[train_idx],
        x_test=x_test[test_idx],
        y_test=y_test[test_idx],
        note=f"Loaded standard IDX files from {mnist_dir}.",
    )


SEGMENTS = {
    0: "abcedf",
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgecd",
    7: "abc",
    8: "abcdefg",
    9: "abfgcd",
}


def paint_rect(image: np.ndarray, r0: int, r1: int, c0: int, c1: int, value: float) -> None:
    r0 = max(0, min(28, r0))
    r1 = max(0, min(28, r1))
    c0 = max(0, min(28, c0))
    c1 = max(0, min(28, c1))
    image[r0:r1, c0:c1] = np.maximum(image[r0:r1, c0:c1], value)


def synthetic_digit_image(label: int, rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.float64)
    shift_r = int(rng.integers(-2, 3))
    shift_c = int(rng.integers(-2, 3))
    thickness = int(rng.integers(2, 5))
    value = float(rng.uniform(0.75, 1.0))

    coords = {
        "a": (4, 4 + thickness, 8, 21),
        "b": (5, 14, 20, 20 + thickness),
        "c": (14, 23, 20, 20 + thickness),
        "d": (22, 22 + thickness, 8, 21),
        "e": (14, 23, 6, 6 + thickness),
        "f": (5, 14, 6, 6 + thickness),
        "g": (13, 13 + thickness, 8, 21),
    }

    for segment in SEGMENTS[label]:
        r0, r1, c0, c1 = coords[segment]
        paint_rect(image, r0 + shift_r, r1 + shift_r, c0 + shift_c, c1 + shift_c, value)

    if rng.random() < 0.35:
        ghost = rng.choice(list("abcdefg"))
        if ghost not in SEGMENTS[label]:
            r0, r1, c0, c1 = coords[ghost]
            paint_rect(image, r0 + shift_r, r1 + shift_r, c0 + shift_c, c1 + shift_c, 0.22)

    image += rng.normal(0.0, 0.13, size=(28, 28))
    image = np.clip(image, 0.0, 1.0)

    if rng.random() < 0.5:
        image = np.roll(image, int(rng.integers(-1, 2)), axis=0)
    if rng.random() < 0.5:
        image = np.roll(image, int(rng.integers(-1, 2)), axis=1)

    return image.reshape(-1)


def make_synthetic_dataset(train_size: int, test_size: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)

    def make_split(size: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.arange(size, dtype=np.int64) % 10
        rng.shuffle(labels)
        images = np.stack([synthetic_digit_image(int(label), rng) for label in labels])
        return images, labels

    x_train, y_train = make_split(train_size)
    x_test, y_test = make_split(test_size)
    return Dataset(
        name="MNIST-like offline digits",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        note=(
            "Standard MNIST IDX files were not found, so a deterministic "
            "28x28 digit-like fallback dataset was generated for an offline "
            "pipeline check."
        ),
    )


class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator):
        self.params: ArrayDict = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "W2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim)),
            "b2": np.zeros(output_dim),
        }

    def clone(self) -> "MLP":
        clone = object.__new__(MLP)
        clone.params = {key: value.copy() for key, value in self.params.items()}
        return clone

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.params["W1"] + self.params["b1"]
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ self.params["W2"] + self.params["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs, {"x": x, "z1": z1, "h1": h1, "probs": probs}

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, ArrayDict]:
        n = x.shape[0]
        probs, cache = self.forward(x)
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        loss += 0.5 * l2 * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))

        dz2 = probs.copy()
        dz2[np.arange(n), y] -= 1.0
        dz2 /= n

        dW2 = cache["h1"].T @ dz2 + l2 * self.params["W2"]
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.params["W2"].T
        dz1 = dh1 * (cache["z1"] > 0.0)
        dW1 = cache["x"].T @ dz1 + l2 * self.params["W1"]
        db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return float(loss), grads

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(x), batch_size):
            probs, _ = self.forward(x[start : start + batch_size])
            outputs.append(np.argmax(probs, axis=1))
        return np.concatenate(outputs)


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(x) == y))


def evaluate_loss(model: MLP, x: np.ndarray, y: np.ndarray, l2: float, batch_size: int = 1024) -> float:
    total = 0.0
    seen = 0
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        loss, _ = model.loss_and_grads(xb, yb, l2)
        total += loss * len(xb)
        seen += len(xb)
    return float(total / seen)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: int = 10) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        matrix[int(target), int(pred)] += 1
    return matrix


def train(
    optimizer: str,
    initial_model: MLP,
    dataset: Dataset,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> TrainResult:
    model = initial_model.clone()
    fisher = {key: np.zeros_like(value) for key, value in model.params.items()}
    metrics: list[dict[str, float | int | str]] = []

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(dataset.y_train))
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = dataset.x_train[idx]
            yb = dataset.y_train[idx]
            _, grads = model.loss_and_grads(xb, yb, args.l2)

            if optimizer == "SGD":
                for key in model.params:
                    model.params[key] -= args.sgd_lr * grads[key]
            elif optimizer == "NG":
                for key in model.params:
                    fisher[key] = args.ng_beta * fisher[key] + (1.0 - args.ng_beta) * (
                        grads[key] ** 2
                    )
                    update = args.ng_lr * grads[key] / (fisher[key] + args.ng_damping)
                    norm = float(np.linalg.norm(update))
                    if norm > args.max_update_norm:
                        update *= args.max_update_norm / (norm + 1e-12)
                    model.params[key] -= update
            else:  # pragma: no cover - protected by caller
                raise ValueError(f"Unknown optimizer: {optimizer}")

        record = {
            "optimizer": optimizer,
            "epoch": epoch,
            "train_loss": evaluate_loss(model, dataset.x_train, dataset.y_train, args.l2),
            "test_loss": evaluate_loss(model, dataset.x_test, dataset.y_test, args.l2),
            "train_acc": accuracy(model, dataset.x_train, dataset.y_train),
            "test_acc": accuracy(model, dataset.x_test, dataset.y_test),
        }
        metrics.append(record)
        print(
            f"{optimizer:>3s} epoch {epoch:02d}: "
            f"train_acc={record['train_acc']:.4f}, test_acc={record['test_acc']:.4f}, "
            f"test_loss={record['test_loss']:.4f}"
        )

    predictions = model.predict(dataset.x_test)
    return TrainResult(
        optimizer=optimizer,
        model=model,
        metrics=metrics,
        predictions=predictions,
        confusion=confusion_matrix(dataset.y_test, predictions),
    )


def save_metrics_csv(results: list[TrainResult], path: Path) -> None:
    rows = [record for result in results for record in result.metrics]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["optimizer", "epoch", "train_loss", "test_loss", "train_acc", "test_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_csv(matrix: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred"] + list(range(matrix.shape[1])))
        for i, row in enumerate(matrix):
            writer.writerow([i] + row.tolist())


def save_curve_png(results: list[TrainResult], path: Path) -> None:
    if Image is None:
        return

    width, height = 900, 540
    margin_left, margin_bottom = 72, 64
    margin_top, margin_right = 42, 42
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    draw.rectangle(
        (margin_left, margin_top, width - margin_right, height - margin_bottom),
        outline=(30, 30, 30),
    )
    draw.text((width // 2 - 120, 14), "Test accuracy comparison", fill=(0, 0, 0), font=font)
    draw.text((margin_left - 58, margin_top - 24), "accuracy", fill=(0, 0, 0), font=font)
    draw.text((width // 2 - 22, height - 30), "epoch", fill=(0, 0, 0), font=font)

    for tick in range(0, 11):
        y = margin_top + plot_h - tick * plot_h / 10
        value = tick / 10
        draw.line((margin_left - 4, y, margin_left, y), fill=(30, 30, 30))
        draw.text((margin_left - 45, y - 5), f"{value:.1f}", fill=(0, 0, 0), font=font)

    colors = {"SGD": (220, 80, 50), "NG": (40, 120, 220)}
    max_epoch = max(int(record["epoch"]) for result in results for record in result.metrics)

    for result in results:
        points: list[tuple[float, float]] = []
        for record in result.metrics:
            epoch = int(record["epoch"])
            acc = float(record["test_acc"])
            x = margin_left + (epoch - 1) * plot_w / max(1, max_epoch - 1)
            y = margin_top + plot_h * (1.0 - acc)
            points.append((x, y))

        if len(points) >= 2:
            draw.line(points, fill=colors[result.optimizer], width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[result.optimizer])

    legend_x = width - margin_right - 150
    legend_y = margin_top + 14
    for i, result in enumerate(results):
        color = colors[result.optimizer]
        y = legend_y + i * 22
        draw.line((legend_x, y + 7, legend_x + 28, y + 7), fill=color, width=3)
        draw.text((legend_x + 36, y), result.optimizer, fill=(0, 0, 0), font=font)

    image.save(path)


def save_confusion_png(matrix: np.ndarray, title: str, path: Path) -> None:
    if Image is None:
        return

    cell = 34
    left = 64
    top = 54
    width = left + cell * 10 + 24
    height = top + cell * 10 + 44
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_value = max(1, int(matrix.max()))

    draw.text((left, 18), title, fill=(0, 0, 0), font=font)
    draw.text((left + 130, height - 24), "predicted", fill=(0, 0, 0), font=font)
    draw.text((8, top + 150), "true", fill=(0, 0, 0), font=font)

    for i in range(10):
        draw.text((left + i * cell + 12, top - 20), str(i), fill=(0, 0, 0), font=font)
        draw.text((left - 24, top + i * cell + 10), str(i), fill=(0, 0, 0), font=font)
        for j in range(10):
            value = int(matrix[i, j])
            intensity = int(255 - 210 * value / max_value)
            color = (intensity, intensity, 255)
            x0 = left + j * cell
            y0 = top + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(210, 210, 210))
            if value:
                draw.text((x0 + 8, y0 + 10), str(value), fill=(0, 0, 0), font=font)

    image.save(path)


def save_samples_png(dataset: Dataset, path: Path) -> None:
    if Image is None:
        return
    scale = 3
    cell = 28 * scale
    image = Image.new("RGB", (cell * 10, cell), "white")
    for digit in range(10):
        idx = int(np.where(dataset.y_train == digit)[0][0])
        sample = (dataset.x_train[idx].reshape(28, 28) * 255).astype(np.uint8)
        tile = Image.fromarray(sample, mode="L").resize((cell, cell), resample=Image.Resampling.NEAREST)
        image.paste(Image.merge("RGB", (tile, tile, tile)), (digit * cell, 0))
    image.save(path)


def load_dataset(args: argparse.Namespace) -> Dataset:
    dataset = load_mnist_dataset(args.mnist_dir, args.train_size, args.test_size, args.seed)
    if dataset is not None:
        return dataset
    if args.require_mnist:
        raise FileNotFoundError(
            "MNIST IDX files were not found. Expected train/test image and label files in "
            f"{args.mnist_dir}."
        )
    return make_synthetic_dataset(args.train_size, args.test_size, args.seed)


def write_summary(
    dataset: Dataset,
    results: list[TrainResult],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    summary = {
        "dataset": dataset.name,
        "note": dataset.note,
        "train_size": int(len(dataset.y_train)),
        "test_size": int(len(dataset.y_test)),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "optimizers": {},
    }

    for result in results:
        last = result.metrics[-1]
        best = max(result.metrics, key=lambda record: float(record["test_acc"]))
        summary["optimizers"][result.optimizer] = {
            "final_train_acc": float(last["train_acc"]),
            "final_test_acc": float(last["test_acc"]),
            "final_test_loss": float(last["test_loss"]),
            "best_test_acc": float(best["test_acc"]),
            "best_epoch": int(best["epoch"]),
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args)
    print(dataset.note)
    print(
        f"Dataset: {dataset.name}; train={len(dataset.y_train)}, test={len(dataset.y_test)}, "
        f"input_dim={dataset.x_train.shape[1]}"
    )

    init_rng = np.random.default_rng(args.seed + 17)
    initial_model = MLP(dataset.x_train.shape[1], args.hidden_dim, 10, init_rng)

    sgd = train("SGD", initial_model, dataset, args, np.random.default_rng(args.seed + 101))
    ng = train("NG", initial_model, dataset, args, np.random.default_rng(args.seed + 101))
    results = [sgd, ng]

    save_metrics_csv(results, output_dir / "results.csv")
    for result in results:
        save_confusion_csv(result.confusion, output_dir / f"confusion_{result.optimizer.lower()}.csv")
        save_confusion_png(
            result.confusion,
            f"{result.optimizer} confusion matrix",
            output_dir / f"confusion_{result.optimizer.lower()}.png",
        )
    save_curve_png(results, output_dir / "accuracy_curve.png")
    save_samples_png(dataset, output_dir / "sample_digits.png")
    write_summary(dataset, results, args, output_dir)

    print("\nFinal comparison:")
    for result in results:
        last = result.metrics[-1]
        print(
            f"{result.optimizer}: test_acc={float(last['test_acc']):.4f}, "
            f"test_loss={float(last['test_loss']):.4f}"
        )
    print(f"Artifacts written to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare SGD and natural gradient on MNIST.")
    parser.add_argument("--mnist-dir", type=Path, default=Path("data") / "mnist")
    parser.add_argument("--require-mnist", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--sgd-lr", type=float, default=0.25)
    parser.add_argument("--ng-lr", type=float, default=0.04)
    parser.add_argument("--ng-damping", type=float, default=0.5)
    parser.add_argument("--ng-beta", type=float, default=0.9)
    parser.add_argument("--max-update-norm", type=float, default=1.5)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
