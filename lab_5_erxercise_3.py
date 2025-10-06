import math
import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


@dataclass
class Config:
    data_dir: str = "./data"
    log_dir: str = "./runs/mnist_mlp"
    ckpt_dir: str = "./checkpoints"
    batch_size: int = 128
    max_epochs: int = 20
    patience: int = 5  # early stopping
    base_lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_sizes: Tuple[int, int, int] = (512, 256, 128)
    num_classes: int = 10
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, hidden_sizes: Tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        layers = []
        in_features = 28 * 28
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.2))
            in_features = hidden
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def lr_range_test(cfg: Config, model: nn.Module, loader: DataLoader, min_lr: float = 1e-6, max_lr: float = 3.0, steps: int = 200) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=min_lr, weight_decay=cfg.weight_decay)
    lr_mult = (max_lr / min_lr) ** (1 / max(steps - 1, 1))

    losses = []
    lrs = []
    running_loss = 0.0

    step = 0
    for images, labels in loader:
        if step >= steps:
            break
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Smooth loss
        running_loss = 0.98 * running_loss + 0.02 * loss.item() if step > 0 else loss.item()
        losses.append(running_loss)
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        # Increase LR exponentially
        optimizer.param_groups[0]["lr"] = lr * lr_mult
        if math.isnan(running_loss) or math.isinf(running_loss):
            break
        step += 1

    # Heuristic: pick LR at which loss is minimal divided by 10
    best_idx = int(torch.tensor(losses).argmin().item())
    suggested_lr = lrs[best_idx] / 10
    return float(max(min(suggested_lr, 1e-2), 1e-4))


def train(cfg: Config) -> None:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    train_loader, test_loader = get_dataloaders(cfg)
    model = MLP(cfg.hidden_sizes, cfg.num_classes).to(cfg.device)

    # Learning rate range test
    lr = lr_range_test(cfg, model, train_loader)
    writer.add_text("lr_finder/suggested_lr", f"{lr:.6f}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    best_acc = 0.0
    best_epoch = -1
    epochs_without_improve = 0

    global_step = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        val_loss, val_acc = evaluate(model, test_loader, cfg.device)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        scheduler.step(val_loss)

        # Checkpointing
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            }, os.path.join(cfg.ckpt_dir, "best.pt"))
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        print(f"Epoch {epoch+1}/{cfg.max_epochs} - val_loss={val_loss:.4f} val_acc={val_acc:.4%} best_acc={best_acc:.4%}")

        # Early stopping
        if epochs_without_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch+1}; best epoch was {best_epoch+1} with accuracy {best_acc:.4%}")
            break

    writer.close()


if __name__ == "__main__":
    train(Config())


