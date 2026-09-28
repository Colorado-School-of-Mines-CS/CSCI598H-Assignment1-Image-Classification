"""Train one of four classifiers on CIFAR-10 and optionally write CSV."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import build_model


RESULT_FIELDS = ["experiment_id", "model", "optimizer", "epoch", "epochs",
                 "batch_size", "learning_rate", "momentum", "num_layers",
                 "seed", "train_loss",
                 "validation_accuracy", "epoch_time_seconds"]


def train_one_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training step and return the loss as a Python float."""
    # TODO: Set gradients to zero, compute logits and loss, backpropagate,
    # update the parameters, and return the loss as a Python float.
    prediction = model(images)
    loss = softmax_cross_entropy(prediction, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # return float(loss)
    return loss.item()




def evaluate(model, data_loader, device):
    # TODO: Evaluate all batches without gradients. Weight each batch accuracy
    # by its number of examples and return the overall accuracy.
    model.eval() 
    with torch.no_grad():
        correct_pred = 0
        total_pred = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            correct_pred += (logits.argmax(dim=1) == labels).sum().item()
            total_pred += labels.size(0)
        return float(correct_pred) / float(total_pred) if total_pred > 0 else 0.0


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the fraction of correct predictions as a Python float."""
    # TODO: Choose the class with the largest logit and compute accuracy.
    predictions = logits.argmax(dim=1)
    accuracy = torch.where(predictions == labels, 1, 0)
    return accuracy.float().mean().item()


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return mean softmax cross-entropy without using F.cross_entropy."""
    # TODO 1: Compute log probabilities using a numerically stable method.
    # TODO 2: Select the log probability of each correct class.
    # TODO 3: Return the mean negative log probability.
    #failed numerical stability
    # prob = torch.softmax(logits, dim=1)
    # log_prob = torch.log(prob)

    log_normalizer = torch.logsumexp(logits, dim=1)
    correct_class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    return (log_normalizer - correct_class_logits).mean()


def build_optimizer(name, parameters, learning_rate, momentum=0.0):
    # Check here for more optimizers: https://docs.pytorch.org/docs/2.14/optim.html
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must satisfy 0 <= momentum < 1")
    if name.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)
    if momentum != 0.0:
        raise ValueError("momentum is only supported by the SGD optimizer")
    if name.lower() == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    raise ValueError(f"Unknown optimizer {name!r}; choose sgd, adam, or adamw.")


def build_lr_scheduler(name, optimizer, epochs, step_size=5, gamma=0.1):
    """Build an optional learning-rate scheduler."""
    name = name.lower()
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    raise ValueError(f"Unknown LR schedule {name!r}; choose none, step, or cosine.")


def make_data_loaders(
        dir_data = "./data",
        batch_SZ=128,
        max_train_samples=None,
        max_val_samples=None,
):
    # TODO 1: load data train / valid sets
    # TODO 2: data augmentation /normalization
    # TODO 3: return dataloaders
    trainig_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))
    ])
    # adding more tricks for VIT perofmance it didnt work
    # trainig_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(brightness=0.2, 
    #                     contrast=0.2,
    #                     saturation=0.2, 
    #                     hue=0.1),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.RandomErasing(p=0.5,
    #                             scale=(0.02, 0.33),
    #                              ratio=(0.3, 3.3),
    #                              value=0),
    #     transforms.Normalize(
    #         (0.5, 0.5, 0.5),
    #         (0.5, 0.5, 0.5))
    # ])
    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])
    train_set = datasets.CIFAR10(dir_data, 
                            train=True, 
                            download=True, 
                            transform=trainig_transform)
    val_set = datasets.CIFAR10(dir_data, 
                            train=False, 
                            download=True, 
                            transform=validation_transform)
    data_Set_sz = len(train_set)
    if max_train_samples is not None:
        train_set = Subset(
            train_set,
            range(min(data_Set_sz, max_train_samples))
        )
    if max_val_samples is not None:
        val_set = Subset(
            val_set, 
            range(min(len(val_set), max_val_samples))
        )

    # train_set, val_set = random_split(
    #     range(data_Set_sz), 
    #     [0.8, 0.2],
    #     generator=torch.Generator().manual_seed(598) 
    # )
    # train_set = Subset(train_set, train_set.indices)
    # val_set = Subset(val_set, val_set.indices)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_SZ,
        shuffle=True)
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_SZ,
        shuffle=False)
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--experiment-id", default="single_run")
    parser.add_argument(
        "--model",
        choices=("linear", "three_layer", "conv", "vit"),
        default="linear",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "step", "cosine"),
        default="none",
        help="Learning-rate schedule to apply after each epoch.",
    )
    parser.add_argument("--lr-step-size", type=int, default=5)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="sgd")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, choices=(2, 5))
    parser.add_argument("--seed", type=int, default=598)
    parser.add_argument("--output")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if args.lr_step_size < 1:
        raise ValueError("LR step size must be positive")
    if not 0.0 < args.lr_gamma <= 1.0:
        raise ValueError("LR gamma must satisfy 0 < gamma <= 1")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: get dataloaders
    train_loader, validation_loader = make_data_loaders(
        dir_data    =args.data_dir,
        batch_SZ=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_validation_samples,
    )

    if args.model in {"conv", "vit"} and args.num_layers is None:
        raise ValueError("--num-layers is required for conv and vit models")
    model = build_model(args.model, num_layers=args.num_layers).to(device)
    optimizer = build_optimizer(
        args.optimizer, model.parameters(), args.learning_rate, args.momentum
    )
    lr_scheduler = build_lr_scheduler(
        args.lr_schedule,
        optimizer,
        epochs=args.epochs,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )
    rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_examples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            count = labels.size(0)
            total_loss += train_one_step(model, images, labels, optimizer) * count
            total_examples += count

        validation_accuracy = evaluate(model, validation_loader, device)
        mean_loss = total_loss / total_examples
        elapsed = time.perf_counter() - start
        current_learning_rate = optimizer.param_groups[0]["lr"]
        rows.append({"experiment_id": args.experiment_id, "model": args.model,
                     "optimizer": args.optimizer, "epoch": epoch, "epochs": args.epochs,
                     "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                     "momentum": args.momentum if args.optimizer == "sgd" else "",
                     "num_layers": args.num_layers if args.num_layers is not None else "",
                     "seed": args.seed, "train_loss": f"{mean_loss:.8f}",
                     "validation_accuracy": f"{validation_accuracy:.8f}",
                     "epoch_time_seconds": f"{elapsed:.4f}"})
        print(
            f"Epoch {epoch:02d} | "
            f"loss={mean_loss:.4f} | validation_accuracy={validation_accuracy:.2%} | "
            f"learning_rate={current_learning_rate:.6g}"
        )
        if lr_scheduler is not None:
            lr_scheduler.step()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
