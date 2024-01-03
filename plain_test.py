import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
import numpy as np

# Optional dependencies for data visualization
# import matplotlib.pyplot as plt

"""
output:
Using Sklearn:

############# Data summary #############
x_train has shape: torch.Size([779, 9])
x_test has shape: torch.Size([335, 9])
y_train has shape: torch.Size([779, 1])
y_test has shape: torch.Size([335, 1])
#######################################
Accuracy on plain test_set: 0.6417910447761194
Fairness Metrics for Group 0:
Accuracy: 0.6549707602339181
ROC AUC not defined (only one class in this group)
Confusion Matrix:
[[112  59]
 [  0   0]]

Fairness Metrics for Group 1:
Accuracy: 0.6280487804878049
ROC AUC not defined (only one class in this group)
Confusion Matrix:
[[  0   0]
 [ 61 103]]
"""


np.random.seed(73)
random.seed(73)


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
    return train_test_split(x, y, test_size=0.3, random_state=73)


def random_data(m=1024, n=2):
    # data separable by the line `y = x`
    x_train = torch.randn(m, n)
    x_test = torch.randn(m // 2, n)
    y_train = (x_train[:, 0] >= x_train[:, 1]).float().unsqueeze(0).t()
    y_test = (x_test[:, 0] >= x_test[:, 1]).float().unsqueeze(0).t()
    return x_train, x_test, y_train, y_test


# You can use whatever data you want without modification to the tutorial
# x_train, x_test, y_train, y_test = random_data()
x_train, x_test, y_train, y_test = heart_disease_data()

print("############# Data summary #############")
print(f"x_train has shape: {x_train.shape}")
print(f"x_test has shape: {x_test.shape}")
print(f"y_train has shape: {y_train.shape}")
print(f"y_test has shape: {y_test.shape}")
print("#######################################")

# Flatten y_train and y_test to match the requirements of sklearn LogisticRegression
y_train = y_train.view(-1).numpy()
y_test = y_test.view(-1).numpy()

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(x_train.numpy(), y_train)

# Predict on the test set
y_pred = model.predict(x_test.numpy())

# Calculate accuracy
plain_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on plain test_set: {plain_accuracy}")

from sklearn.metrics import confusion_matrix, roc_auc_score

# Predict on the test set
y_pred_proba = model.predict_proba(x_test.numpy())[:, 1]

# Calculate fairness metrics
conf_matrix_group0 = confusion_matrix(y_test[y_test == 0], y_pred[y_test == 0])
conf_matrix_group1 = confusion_matrix(y_test[y_test == 1], y_pred[y_test == 1])

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
