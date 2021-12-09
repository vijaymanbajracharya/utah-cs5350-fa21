import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch as th
import numpy as np
import torch.nn.functional as F

INIT = "he"


def read_csv():
    # Loading the Concrete dataset
    cols = '''
    variance,
    skewness,
    curtosis,
    entropy,
    label
    '''
    table = []

    for c in cols.split(','):
        if(c.strip()):
            table.append(c.strip())

    train = pd.read_csv("bank-note/train.csv", names=table)
    test = pd.read_csv("bank-note/test.csv", names=table)

    y = train["label"].copy()
    train = train.iloc[:, 0:-1]
    train = train.to_numpy()
    y = y.to_numpy()[:, np.newaxis]

    y_test = test["label"].copy()
    test = test.iloc[:, 0:-1]
    test = test.to_numpy()
    y_test = y_test.to_numpy()[:, np.newaxis]

    return train, test, y, y_test


def init_weights(m):
    if INIT == "he":
      if type(m) == nn.Linear:
          th.nn.init.kaiming_uniform_(m.weight)
          m.bias.data.fill_(0.01)

    if INIT == "xavier":
      if type(m) == nn.Linear:
          th.nn.init.xavier_uniform_(m.weight)
          m.bias.data.fill_(0.01)


class ThreeLayerReluNeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=50):
        super(ThreeLayerReluNeuralNetwork, self).__init__()
        global INIT
        INIT = "he"
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.model.apply(init_weights)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


class FiveLayerReluNeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=50):
        super(FiveLayerReluNeuralNetwork, self).__init__()
        global INIT
        INIT = "he"
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.model.apply(init_weights)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


class NineLayerReluNeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=50):
        super(NineLayerReluNeuralNetwork, self).__init__()
        global INIT
        INIT = "he"
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.model.apply(init_weights)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


class ThreeLayerTanhNeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=50):
        super(ThreeLayerTanhNeuralNetwork, self).__init__()
        global INIT
        INIT = "xavier"
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.model.apply(init_weights)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

class FiveLayerTanhNeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=50):
        super(FiveLayerTanhNeuralNetwork, self).__init__()
        global INIT
        INIT = "xavier"
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.model.apply(init_weights)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

class NineLayerTanhNeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=50):
        super(NineLayerTanhNeuralNetwork, self).__init__()
        global INIT
        INIT = "xavier"
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.model.apply(init_weights)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


def train(model, optimizer, X, y, epochs=100):
  model = model.to(device=device)
  for e in range(epochs):
      model.train()
      scores = model(X)

      loss = F.cross_entropy(scores, y)

      # Zero out all of the gradients
      optimizer.zero_grad()

      # Perform backward pass
      loss.backward()

      # Update parameters with optimizer
      optimizer.step()


def check_accuracy(X, y):
  scores = model(X)
  pred_probab = nn.Softmax(dim=1)(scores)
  y_pred = pred_probab.argmax(1)

  error = 0
  for i in range(len(X)):
    if y_pred[i] != y[i]:
      error += 1

  return error / len(X)


if __name__ == "__main__":

  if th.cuda.is_available():
    device = th.device('cuda:0')
  else:
    device = th.device('cpu')

  train_x, test_x, train_y, test_y = read_csv()

  train_x = th.tensor(train_x).to(device=device, dtype=th.float32)
  test_x = th.tensor(test_x).to(device=device, dtype=th.float32)
  train_y = th.tensor(train_y).to(device=device, dtype=th.long).squeeze()
  test_y = th.tensor(test_y).to(device=device, dtype=th.long) .squeeze()

  hidden_dims = [5, 10, 25, 50, 100]

  for dim in hidden_dims:
    model = ThreeLayerReluNeuralNetwork(hidden_dim=dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Relu Activation, Depth 3, Hidden Dimension {dim}")
    train(model, optimizer, train_x, train_y)
    train_error = check_accuracy(train_x, train_y) * 100
    print(f"Train Error: {train_error}%")
    test_error = check_accuracy(test_x, test_y) * 100
    print(f"Test Error: {test_error}%")
    print(f"\r\n")

  for dim in hidden_dims:
    model = FiveLayerReluNeuralNetwork(hidden_dim=dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Relu Activation, Depth 5, Hidden Dimension {dim}")
    train(model, optimizer, train_x, train_y)
    train_error = check_accuracy(train_x, train_y) * 100
    print(f"Train Error: {train_error}%")
    test_error = check_accuracy(test_x, test_y) * 100
    print(f"Test Error: {test_error}%")
    print(f"\r\n")

  for dim in hidden_dims:
    model = NineLayerReluNeuralNetwork(hidden_dim=dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Relu Activation, Depth 9, Hidden Dimension {dim}")
    train(model, optimizer, train_x, train_y)
    train_error = check_accuracy(train_x, train_y) * 100
    print(f"Train Error: {train_error}%")
    test_error = check_accuracy(test_x, test_y) * 100
    print(f"Test Error: {test_error}%")
    print(f"\r\n")

  
  for dim in hidden_dims:
    model = ThreeLayerTanhNeuralNetwork(hidden_dim=dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Tanh Activation, Depth 3, Hidden Dimension {dim}")
    train(model, optimizer, train_x, train_y)
    train_error = check_accuracy(train_x, train_y) * 100
    print(f"Train Error: {train_error}%")
    test_error = check_accuracy(test_x, test_y) * 100
    print(f"Test Error: {test_error}%")
    print(f"\r\n")

  for dim in hidden_dims:
    model = FiveLayerTanhNeuralNetwork(hidden_dim=dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Tanh Activation, Depth 5, Hidden Dimension {dim}")
    train(model, optimizer, train_x, train_y)
    train_error = check_accuracy(train_x, train_y) * 100
    print(f"Train Error: {train_error}%")
    test_error = check_accuracy(test_x, test_y) * 100
    print(f"Test Error: {test_error}%")
    print(f"\r\n")
    
  for dim in hidden_dims:
    model = NineLayerTanhNeuralNetwork(hidden_dim=dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Tanh Activation, Depth 9, Hidden Dimension {dim}")
    train(model, optimizer, train_x, train_y)
    train_error = check_accuracy(train_x, train_y) * 100
    print(f"Train Error: {train_error}%")
    test_error = check_accuracy(test_x, test_y) * 100
    print(f"Test Error: {test_error}%")
    print(f"\r\n")
  

  