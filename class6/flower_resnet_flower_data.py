from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
try:
    from torchvision import datasets, transforms, models
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "未找到 torchvision。请在当前环境安装与 torch 匹配的 torchvision 后再运行。\n"
        "例如（conda）：conda install -c pytorch torchvision\n"
        "或（pip）：pip install torchvision"
    ) from e

# 与划分、随机增强一致
SEED = 42
TRAIN_RATIO = 0.8


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(
    data_dir: Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    从 DataSets/flower_data 中构建 train/val DataLoader。

    当前仓库中的花卉数据（Oxford 102 花卉子集）实际结构为「根目录下按类别分子文件夹」：
        DataSets/flower_data/
          daisy/
          dandelion/
          roses/
          sunflowers/
          tulips/
          LICENSE.txt

    支持两种用法：
    1）若存在 train/ 与 val/，则分别作为训练集与验证集（每类子文件夹）。
    2）否则在根目录上按类别读取，再按 TRAIN_RATIO（默认 0.8）随机划分 train/val。
    """
    assert data_dir.is_dir(), f"数据目录不存在: {data_dir.resolve()}"

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    # 情况 1：已经有 train/val 子目录，直接读取
    if train_dir.exists() and val_dir.exists():
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
        val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_tfms)
    else:
        # 情况 2：根目录为 ImageFolder（各类别文件夹），按 8:2 划分
        full_ds = datasets.ImageFolder(root=str(data_dir), transform=val_tfms)
        n_total = len(full_ds)
        if n_total == 0:
            raise RuntimeError(
                f"{data_dir} 下未找到图片，请确认存在类别子文件夹（如 daisy/、roses/ 等）。"
            )
        n_train = int(n_total * TRAIN_RATIO)
        n_val = n_total - n_train

        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED),
        )

        # 与 notebook 一致：子集仍按同一索引访问，需用相同 root 重建 Dataset 以换 transform
        root_str = str(data_dir)
        train_ds.dataset = datasets.ImageFolder(root=root_str, transform=train_tfms)
        val_ds.dataset = datasets.ImageFolder(root=root_str, transform=val_tfms)

    num_classes = len(
        train_ds.dataset.classes if hasattr(train_ds, "dataset") else train_ds.classes
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, num_classes


def build_resnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    使用 torchvision 的 ResNet18 作为特征提取器，替换最后的全连接层。
    """
    # torchvision 新版本推荐使用 weights=...，老版本仍支持 pretrained=...
    try:
        from torchvision.models import ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_resnet_on_flower(
    epochs: int = 15,
    img_size: int = 224,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    pretrained: bool = True,
) -> Dict[str, Any]:
    """
    在 DataSets/flower_data (train/val) 上训练 ResNet。

    返回训练过程中的指标以及最佳模型路径。
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "DataSets" / "flower_data"

    device = get_device()
    print("device:", device)

    train_loader, val_loader, num_classes = build_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"数据目录: {data_dir.resolve()}")
    train_sub = data_dir / "train"
    val_sub = data_dir / "val"
    if train_sub.exists() and val_sub.exists():
        print("划分方式: 使用 train/ 与 val/ 子目录")
    else:
        print(
            f"划分方式: 根目录各类别文件夹，随机 {TRAIN_RATIO:.0%} / {1 - TRAIN_RATIO:.0%} "
            f"train/val（seed={SEED}）"
        )
    print(
        f"训练样本数: {len(train_loader.dataset)}  "
        f"验证样本数: {len(val_loader.dataset)}"
    )
    print("num_classes:", num_classes)
    # 兼容 ImageFolder 或 random_split 得到的 Subset
    base_ds = (
        train_loader.dataset.dataset
        if hasattr(train_loader.dataset, "dataset")
        else train_loader.dataset
    )
    print("classes:", base_ds.classes)

    model = build_resnet(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_acc = 0.0
    best_model_path = project_root / "class6" / "best_resnet_flower.pth"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  保存新的最佳模型到: {best_model_path}")

    print(f"最佳验证准确率: {best_val_acc:.4f}")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_model_path": str(best_model_path),
    }


if __name__ == "__main__":
    # 默认设置适合在普通 GPU 上快速实验，
    # 如需更好效果可以适当增大 epochs / batch_size。
    train_resnet_on_flower(
        epochs=15,
        img_size=224,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        num_workers=0,  # Windows 上建议为 0
        pretrained=True,
    )

