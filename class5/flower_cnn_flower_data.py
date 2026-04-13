#!/usr/bin/env python
# coding: utf-8

# # 花卉数据集图像分类（CNN，PyTorch）
# 
# 本 notebook 读取 `DataSets/flower_data` 下的花卉图片数据，完成：
# 
# - 数据读取与预处理（训练/测试划分、数据增强、归一化）
# - 搭建基于卷积神经网络（CNN）的分类模型并训练
# - 输出训练结果并可视化（Loss/Accuracy 曲线、预测效果图）
# 
# > 数据集目录结构（`ImageFolder` 约定）：
# >
# > ```
# > DataSets/flower_data/
# >   daisy/
# >   dandelion/
# >   roses/
# >   sunflowers/
# >   tulips/
# > ```

# In[1]:


import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 让实验可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## 1. 读取数据集与数据预处理
# 
# 我们使用 `torchvision.datasets.ImageFolder` 直接从文件夹读取图片。
# 
# - **数据增强（train）**：随机裁剪/翻转/轻微颜色抖动，提升泛化
# - **测试（test）**：只做 Resize + CenterCrop，保持评估稳定
# - **归一化**：使用 ImageNet 的 mean/std（通用且常用）
# 
# 同时把全量数据集按比例划分为：train/test（不留验证集）。

# In[3]:


# 数据目录（基于脚本位置推导，避免工作目录不同导致路径失效）
DATA_DIR = Path(__file__).resolve().parent.parent / "DataSets" / "flower_data"
assert DATA_DIR.exists(), f"找不到数据目录: {DATA_DIR.resolve()}"

# 超参数（可按需改）
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows 上设为 0 更稳

# ImageNet 归一化
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 先用“无增强”的 transform 构建一个基准 dataset，用于划分与类别读取
base_ds = datasets.ImageFolder(root=str(DATA_DIR), transform=test_tfms)
class_names = base_ds.classes
num_classes = len(class_names)

# 按比例划分：2:8 训练/测试（即 train=0.2, test=0.8）
train_ratio, test_ratio = 2 / (2 + 8), 8 / (2 + 8)
n_total = len(base_ds)
n_train = int(n_total * train_ratio)
n_test = n_total - n_train

train_ds, test_ds = random_split(
    base_ds,
    [n_train, n_test],
    generator=torch.Generator().manual_seed(SEED),
)

# 重要：给 train/test 子集换成对应 transform
train_ds.dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=train_tfms)
test_ds.dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# 阶段性输出：数据规模与划分结果
print(f"[Data] Total={n_total}, Train={len(train_ds)}, Test={len(test_ds)}, Classes={num_classes}")
print(f"[Data] Class names: {class_names}")
print(f"[Data] Using device: {device}")


# ## 2. 可视化：看几张增强后的训练样本
# 
# 为了确认数据读取与增强没问题，我们从 `train_loader` 抽取一个 batch，把归一化后的张量反归一化并显示。

# ## 3. 搭建 CNN 模型
# 
# 这里实现一个轻量 CNN：多层卷积 + BatchNorm + ReLU + 池化，最后用全局平均池化（`AdaptiveAvgPool2d`）接全连接分类。
# 
# 如果你后续希望更高精度，也可以替换成 `torchvision.models` 的预训练网络（如 ResNet18），但本任务先用“从零搭建”的 CNN 更直观。

# 训练函数：训练集用于反向传播；测试集用于阶段性评估输出（不参与参数更新）
def train(
    model,
    train_loader,
    test_loader=None,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    print_every=1,
    eval_every=1,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    best_val_acc = -1.0
    best_val_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += x.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        do_eval = test_loader is not None and (
            (epoch % eval_every) == 0 or epoch == 1 or epoch == epochs
        )

        if do_eval:
            test_stats = evaluate(model, test_loader, criterion)
            history["test_loss"].append(test_stats.loss)
            history["test_acc"].append(test_stats.acc)

            if test_stats.acc > best_val_acc:
                best_val_acc = test_stats.acc
                best_val_epoch = epoch

            if (epoch % print_every) == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"[Train] Epoch {epoch:03d}/{epochs} | "
                    f"train_acc={train_acc:.4f} | val_acc={test_stats.acc:.4f} | "
                    f"train_loss={train_loss:.4f} | val_loss={test_stats.loss:.4f}"
                    f" | max_val_acc={best_val_acc:.4f}(epoch {best_val_epoch:03d})"
                )
        else:
            if (epoch % print_every) == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"[Train] Epoch {epoch:03d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                    f"val_acc=skipped | max_val_acc={best_val_acc:.4f}(epoch {best_val_epoch:03d})"
                )

    if best_val_epoch != -1:
        print(f"[Train] Best val_acc={best_val_acc:.4f} at epoch {best_val_epoch:03d}")

    return history

# In[5]:


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=num_classes).to(device)

# 阶段性输出：模型参数规模
num_params = sum(p.numel() for p in model.parameters())
print(f"[Model] SimpleCNN parameters: {num_params:,}")


# ## 4. 训练
#
# 训练时记录每个 epoch 的：
# - **train loss / train acc**

# In[6]:


from dataclasses import dataclass

@dataclass
class EpochStats:
    loss: float
    acc: float

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return EpochStats(loss=total_loss / total, acc=total_correct / total)


EPOCHS = 100
LR = 1e-3

# 阶段性输出：训练开始
print(f"[Train] Start | epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}")
history = train(model, train_loader, test_loader=test_loader, epochs=EPOCHS, lr=LR, print_every=1, eval_every=1)


# ## 6. 在测试集上评估
# 
# 使用训练完成后的最终模型，在测试集上计算最终准确率。

# In[8]:


criterion = nn.CrossEntropyLoss()
test_stats = evaluate(model, test_loader, criterion)
print(f"[Test] loss={test_stats.loss:.4f} | acc={test_stats.acc:.4f}")

 

