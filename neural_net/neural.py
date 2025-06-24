#%%
import numpy as np
import pandas as pd
import torch
from fairlearn.metrics import demographic_parity_difference
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_net.preprocessing import get_train_test_data
from neural_net.results import *
from utils.graphs import compare, scatter_plot
from utils.neural_utils import LawDataset, NeuralNetwork, predict

# %%
X_train, X_test, y_train, y_test, sf_train, sf_test = get_train_test_data("law_data.csv", "first_pf")

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


# assert len(sf_train) == len(X_train)

#%%
λ = np.linspace(1e-3, 1, 10)
accuracy = []
dem = []

def train_loop(dataloader, model, loss_fn, optimizer, lambda_value):

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device).float().view(-1, 1)
        pred = model(X)
        
        pred_label = (torch.sigmoid(pred) > 0.5).float()
        # print(X, y)
        fairness = demographic_parity_difference(y, pred_label, sensitive_features=sf_train[batch * 64: (batch+1) * 64])

        loss = loss_fn(pred, y) + lambda_value * fairness

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
    dem.append(fairness)

def test_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device).float().view(-1, 1)
            pred = model(X) 
            test_loss += loss_fn(pred, y).item()
            # Pour l'accuracy :
            pred_label = (torch.sigmoid(pred) > 0.5).float()
            correct += (pred_label == y).sum().item()
            # print(demographic_parity_difference(y, pred_label, sensitive_features=sf_train["race"]))

    test_loss /= num_batches
    correct /= size
    losses[epoch] = test_loss

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    accuracy.append(correct)


# %%
loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 20
for i in tqdm(list(λ)):
    print(f"Training with λ = {i}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, i)
        test_loop(test_dataloader, model, loss_fn, t)
print("Done!")
scatter_plot(losses, xlabel="Epochs", ylabel="Loss", title="Loss per Epoch")

#%%

accuracies_mean = np.reshape(accuracies_all, ((10, 20))).mean(axis=1)
accuracies = [accuracies_all[k * 20] for k in range(10)]


plt.bar(list(map(str, list(λ))), accuracies_mean)
plt.ylim(min(accuracies_mean) - 0.001, max(accuracies_mean) + 0.001)  # Zoom in on the top of the bars
plt.ylabel("Accuracy")
plt.xlabel("λ")
plt.title("Accuracy vs λ (zoomed on top)")
plt.show()

#%%
plt.bar(list(map(str, list(λ))), accuracies)
plt.ylim(min(accuracies) - 0.001, max(accuracies) + 0.001)  # Zoom in on the top of the bars
plt.ylabel("Accuracy")
plt.xlabel("λ")
plt.title("Accuracy vs λ (zoomed on top)")
plt.show()

#%%
dems = np.reshape(dems_all, ((10, 20))).mean(axis=1)
plt.bar(list(map(str, list(λ))), accuracies / dems)
# plt.ylim(min(dems) - 0.001, max(dems) + 0.001)  # Zoom in on the top of the bars
plt.ylabel("Demographic Parity Difference")
plt.xlabel("λ")
plt.title("Demographic Parity vs λ (zoomed on top)")
plt.show()

#%%
print(demographic_parity_difference(y_test, predict(model, X_test), sensitive_features=sf_test))
# %%
# rf, reg = export_model()

# pred_rf = rf.predict(X_test)
# pred_reg = reg.predict(X_test)

# # Generate predictions for the neural network on the test set
# pred_neural = predict(model, X_test)

# compare(
#     [
#         classification_report(y_test, pred_rf, output_dict=True),
#         classification_report(y_test, pred_reg, output_dict=True),
#         classification_report(y_test, (pred_neural > 0.5).astype(float), output_dict=True)
#     ],
#     model_names=["Random Forest", "Logistic Regression", "Neural Network"],
#     print_output=True,
# )
# %%
torch.save(model.state_dict(), "model.pth")