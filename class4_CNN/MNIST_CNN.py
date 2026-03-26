import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ===================== 1. 超参数设置 =====================
BATCH_SIZE = 64    # 批次大小
EPOCHS = 2         # 训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动使用GPU/CPU
DATASET_NAME = "MNIST"
DATASET_ROOT = Path("DataSets") / DATASET_NAME

# ===================== 2. 数据预处理 =====================
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集标准化
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(root=str(DATASET_ROOT), train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=str(DATASET_ROOT), train=False, download=True, transform=transform) 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 3. 构建神经网络模型 =====================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 + 全连接层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)  # 输出10个数字（0-9）

    def forward(self, x):
        # 前向传播
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 320)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(DEVICE)

# ===================== 4. 定义损失函数和优化器 =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===================== 5. 训练模型 =====================
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)   # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        
        # 打印训练进度
        if batch_idx % 100 == 0:
            print(f'训练轮数: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]\t损失: {loss.item():.6f}')

# 打印训练设备（新增代码）
print(f"模型正在使用: {DEVICE} 进行训练\n")

# 开始训练
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)

# ===================== 6. 测试 + 可视化输入输出 =====================
def visualize_predictions(model, device, test_loader, num_images=10):
    model.eval()
    with torch.no_grad():
        # 获取一批测试数据
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        # 转换为numpy用于绘图
        images = data.cpu().numpy()
        labels = target.cpu().numpy()
        preds = torch.argmax(output, dim=1).cpu().numpy()
        
        # 创建画布
        plt.figure(figsize=(12, 4))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            # 显示图片（输入可视化）
            plt.imshow(np.squeeze(images[i]), cmap='gray')
            # 显示预测结果（输出可视化）
            plt.title(f'真实:{labels[i]}\n预测:{preds[i]}', color='green' if labels[i]==preds[i] else 'red')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# 执行可视化预测
print("\n========== 展示测试集预测结果 ==========")
visualize_predictions(model, DEVICE, test_loader)