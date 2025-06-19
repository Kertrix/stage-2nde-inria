#%%
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from preprocessing import get_train_test_data
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from randomforest import export_model
from utils.graphs import compare, scatter_plot
from utils.neural_utils import LawDataset, NeuralNetwork, predict

# %%
X_train, X_test, y_train, y_test = get_train_test_data("../law_data.csv", "first_pf")

training_data = LawDataset(X_train, y_train)
testing_data = LawDataset(X_test, y_test)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=False)
# %%
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
# %%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
# %%

model = NeuralNetwork(input_size=X_train.shape[1]).to(device)
model
# %%
learning_rate = 1e-2
batch_size = 64
epochs = 10

losses = {}

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device).float().view(-1, 1)
        pred = model(X)
        # print(X, y)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float().view(-1, 1)
            pred = model(X) 
            test_loss += loss_fn(pred, y).item()
            # Pour l'accuracy :
            pred_label = (torch.sigmoid(pred) > 0.5).float()
            correct += (pred_label == y).sum().item()
    test_loss /= num_batches
    correct /= size
    losses[epoch] = test_loss

    scatter_plot(losses, xlabel="Epochs", ylabel="Loss", title="Loss per Epoch")
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %%
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn, t)
print("Done!")


# %%
rf, reg = export_model()

pred_rf = rf.predict(X_test)
pred_reg = reg.predict(X_test)

# Generate predictions for the neural network on the test set
pred_neural = predict(model, X_test)

compare(
    [
        classification_report(y_test, pred_rf, output_dict=True),
        classification_report(y_test, pred_reg, output_dict=True),
        classification_report(y_test, (pred_neural > 0.5).astype(float), output_dict=True)
    ],
    model_names=["Random Forest", "Logistic Regression", "Neural Network"],
    print_output=True,
)
# %%
torch.save(model.state_dict(), "model.pth")