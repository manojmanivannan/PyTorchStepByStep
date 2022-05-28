import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from PyRegressionGeneric import StepByStep
from chapter3 import *

X, y = make_moons(n_samples=100, noise=0.3, random_state=0)

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2, random_state=13)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_val = sc.transform(X_val)

# fig = figure1(X_train, y_train, X_val, y_val); plt.show()

torch.manual_seed(13)

# Builds tensors from numpy arrays
x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()

x_val_tensor = torch.as_tensor(X_val).float()
y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()

# Builds dataset containing ALL data points
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# Builds a loader of each set
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)

lr = 0.1
torch.manual_seed(42)
model = nn.Sequential()
model.add_module('linear', nn.Linear(2, 1))
# model1.add_module('sigmoid', nn.Sigmoid())
print(model.state_dict())

loss_fn = nn.BCEWithLogitsLoss() # sigmoid is taken care inside this loss
# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

n_epochs = 100

sbs = StepByStep(model,loss_fn,optimizer)
sbs.set_loaders(train_loader,val_loader)
sbs.train(n_epochs)

fig = sbs.plot_losses(); plt.show()

print(model.state_dict())

predictions = sbs.predict(x_train_tensor[:4])

print(predictions)
print('Clearly this is not probabilities\nBecause we are missing the sigmoid')

probabilities = sigmoid(predictions)
print(probabilities)