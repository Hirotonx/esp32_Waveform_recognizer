import numpy as np
import torch
import torch.nn as nn
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def load_model(model_path):
    model = SignalCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_signal(signal):
    signal = np.array(signal)
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return signal

def predict(model, signal):
    signal = preprocess_signal(signal)
    with torch.no_grad():
        output = model(signal)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def main(signal, model_path='best.pth'):
    model = load_model(model_path)
    prediction = predict(model, signal)
    print(f"Predicted Label: {prediction}")
    return  int(prediction)
if __name__ == "__main__":
    example_signal = []
    res = main(example_signal)
    print(res)
