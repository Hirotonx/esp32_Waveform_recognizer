import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import clear_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

signals_data = clear_data.parse_signals('datas.txt')
# 将信号数据和标签转换为NumPy数组
signals = np.array([signal for signal, label in signals_data])
labels = np.array([label for signal, label in signals_data])

# 转换为PyTorch张量
signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)  # 将unsqueeze维度改为1
labels = torch.tensor(labels, dtype=torch.long)

# 创建数据集和数据加载器
dataset = TensorDataset(signals, labels)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 定义卷积神经网络模型
class SignalCNN(nn.Module):
    def __init__(self):
        super(SignalCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 16, 128)  # Adjust the size accordingly
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16)  # Adjust the size accordingly
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SignalCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for signals, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for signals, labels in val_loader:
            outputs = model(signals)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best.pth')
    torch.save(model.state_dict(), 'last.pth')

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()