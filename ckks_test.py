import torch
import tenseal as ts
import pandas as pd
import random
from time import time
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# those are optional and are not necessary for training
import numpy as np
import matplotlib.pyplot as plt

"""
(Doubtful) Output
Using CKKS and PyTorch:

############# Data summary #############
x_train has shape: torch.Size([780, 9])`
y_train has shape: torch.Size([780, 1])
x_test has shape: torch.Size([334, 9])
y_test has shape: torch.Size([334, 1])
#######################################
Accuracy on plain test_set: 0.688622772693634
Fairness Metrics for Group 0:
Accuracy: 0.0
ROC AUC not defined (only one class in this group)
Confusion Matrix:
[[  0 168]
 [  0   0]]

Fairness Metrics for Group 1:
Accuracy: 1.0
ROC AUC not defined (only one class in this group)
Confusion Matrix:
[[166]]

Polynomial curve fitting (Cubic Equations)

"""



np.random.seed(73)
random.seed(73)


def split_train_test(x, y, test_ratio=0.3):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)
    # delimiter between test and train data
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]


def heart_disease_data():
    data = pd.read_csv("./framingham.csv")
    # drop rows with missing values
    data = data.dropna()
    # drop some features
    data = data.drop(
        columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"]
    )
    # balance data
    grouped = data.groupby("TenYearCHD")
    data = grouped.apply(
        lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True)
    )
    # extract labels
    y = torch.from_numpy(data["TenYearCHD"].values).float().unsqueeze(1)
    data = data.drop("TenYearCHD", axis="columns")
    # standardize data
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    return split_train_test(x, y)


def random_data(m=1024, n=2):
    # data separable by the line `y = x`
    x_train = torch.randn(m, n)
    x_test = torch.randn(m // 2, n)
    y_train = (x_train[:, 0] >= x_train[:, 1]).float().unsqueeze(0).t()
    y_test = (x_test[:, 0] >= x_test[:, 1]).float().unsqueeze(0).t()
    return x_train, y_train, x_test, y_test


# You can use whatever data you want without modification to the tutorial
# x_train, y_train, x_test, y_test = random_data()
x_train, y_train, x_test, y_test = heart_disease_data()

print("############# Data summary #############")
print(f"x_train has shape: {x_train.shape}")
print(f"y_train has shape: {y_train.shape}")
print(f"x_test has shape: {x_test.shape}")
print(f"y_test has shape: {y_test.shape}")
print("#######################################")


class LR(torch.nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out


n_features = x_train.shape[1]


model = LR(n_features)
# use gradient descent with a learning_rate=1
optim = torch.optim.SGD(model.parameters(), lr=1)
# use Binary Cross Entropy Loss
criterion = torch.nn.BCELoss()


# define the number of epochs for both plain and encrypted training
EPOCHS = 5


def train(model, optim, criterion, x, y, epochs=EPOCHS):
    for e in range(1, epochs + 1):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        print(f"Loss at epoch {e}: {loss.data}")
    return model


model = train(model, optim, criterion, x_train, y_train)


def accuracy(model, x, y):
    out = model(x)
    correct = torch.abs(y - out) < 0.5
    return correct.float().mean()


plain_accuracy = accuracy(model, x_test, y_test)
print(f"Accuracy on plain test_set: {plain_accuracy}")

# Predict on the test set
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    y_pred_proba = torch.sigmoid(model(x_test)).numpy()

# Convert probability predictions to binary predictions
y_pred = (y_pred_proba >= 0.5).astype(int)

# Flatten the y_test and y_pred arrays to handle 1D arrays
y_test_flat = y_test.squeeze().numpy() if isinstance(y_test, torch.Tensor) else y_test.ravel()
y_pred_flat = y_pred.squeeze() if isinstance(y_pred, torch.Tensor) else y_pred.ravel()

# Calculate confusion matrices
conf_matrix_group0 = confusion_matrix(y_test_flat[y_test_flat == 0], y_pred_flat[y_test_flat == 0], labels=[0, 1])
conf_matrix_group1 = confusion_matrix(y_test_flat[y_test_flat == 1], y_pred_flat[y_test_flat == 1], labels=[0, 1])

# Fairness metrics for different demographic groups
accuracy_group0 = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])
accuracy_group1 = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])

# Calculate ROC AUC for Group 0
unique_labels_group0 = np.unique(y_test[y_test == 0])
if len(unique_labels_group0) > 1:
    roc_auc_group0 = roc_auc_score(y_test[y_test == 0], y_pred_proba[y_test == 0])
else:
    roc_auc_group0 = None

# Calculate ROC AUC for Group 1
unique_labels_group1 = np.unique(y_test[y_test == 1])
if len(unique_labels_group1) > 1:
    roc_auc_group1 = roc_auc_score(y_test[y_test == 1], y_pred_proba[y_test == 1])
else:
    roc_auc_group1 = None

print("Fairness Metrics for Group 0:")
print(f"Accuracy: {accuracy_group0}")
if roc_auc_group0 is not None:
    print(f"ROC AUC: {roc_auc_group0}")
else:
    print("ROC AUC not defined (only one class in this group)")
print("Confusion Matrix:")
print(conf_matrix_group0)

print("\nFairness Metrics for Group 1:")
print(f"Accuracy: {accuracy_group1}")
if roc_auc_group1 is not None:
    print(f"ROC AUC: {roc_auc_group1}")
else:
    print("ROC AUC not defined (only one class in this group)")
print("Confusion Matrix:")
print(conf_matrix_group1)