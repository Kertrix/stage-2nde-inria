import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


def predict(model, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return (probs > 0.5).astype(float)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x.float())
        return logits


class LawDataset(Dataset):
    def __init__(self, X, y):
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)

    def __getitem__(self, idx):
        item = self.X.iloc[idx]
        label = float(self.y.iloc[idx])
        return torch.tensor(item, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )
