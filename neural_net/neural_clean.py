#%%
import torch
from fairlearn.metrics import demographic_parity_difference
from torch import nn
from torch.utils.data import DataLoader

from neural_net.preprocessing import get_train_test_data
from utils.graphs import scatter_plot
from utils.neural_utils import LawDataset, NeuralNetwork

#%%
X_train, X_test, y_train, y_test, sf_train, sf_test = get_train_test_data("law_data.csv", "first_pf")

training_data = LawDataset(X_train, y_train)
testing_data = LawDataset(X_test, y_test)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

#%%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork(input_size=X_train.shape[1]).to(device)

#### Training ####
learning_rate = 1e-2
batch_size = 64
epochs = 10

losses = {}
λ = 0.334

def train_loop(dataloader, model, loss_fn, optimizer, lambda_value = None):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device).float().view(-1, 1)
        pred = model(X)
        
        pred_label = (torch.sigmoid(pred) > 0.5).float()
        # print(X, y)
        fairness = 0
        
        if lambda_value:
            fairness = lambda_value * demographic_parity_difference(y, pred_label, sensitive_features=sf_train[batch * 64: (batch+1) * 64])

        loss = loss_fn(pred, y) + fairness

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

            pred_label = (torch.sigmoid(pred) > 0.5).float()
            correct += (pred_label == y).sum().item()

    test_loss /= num_batches
    correct /= size
    losses[epoch] = test_loss

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#### Main loop ####
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn, i)

scatter_plot(losses, xlabel="Epochs", ylabel="Loss", title="Loss per Epoch")
print("Done!")

#%%
torch.save(model.state_dict(), "neural_net/model.pth")