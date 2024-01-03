from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

import numpy as np


def calculate_risk_difference(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate the risk difference
    risk_positive_class = np.mean(y_pred)
    risk_difference = risk_positive_class - np.mean(y_true)

    return risk_difference


# Defining column headers since the data file doesn't have them
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

adult = pd.read_csv(
    "adult.data", delimiter=", ", engine="python", header=None, names=columns
)

# Remove the rows with any missing value. Missing value can also be marked with a ? in the data
adult = adult.replace("?", pd.NA)
adult = adult.dropna()

# print(adult)

# Convert the sex column to pure binary
adult["sex"] = adult["sex"].map({"Male": 0, "Female": 1})

# Define the features and target
features = adult[columns[:-1]]  # Exclude the target column
target = adult["income"]

# Define categorical features
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "race",
    "relationship",
    "native-country",
]

# Create a column transformer to apply different transformers to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), features.columns.difference(categorical_features)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Create a pipeline with the column transformer and the logistic regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=2000)),
    ]
)

target_binary = target.map({">50K": 1, "<=50K": 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    features, target_binary, test_size=0.2, random_state=42
)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
# print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1-score:", f1_score(y_test, predictions))

probabilities = model.predict_proba(X_test)[:, 1]

accuracy_values = [accuracy_score(y, model.predict(X)) for (X, y) in [(X_train, y_train), (X_test, y_test)]]
accuracy_mean_std = (np.mean(accuracy_values), np.std(accuracy_values))
print(f"\nAccuracy Mean: {accuracy_mean_std[0]:f}, Std: {accuracy_mean_std[1]:.4f}")

# Calculate and print the risk difference
risk_diff = calculate_risk_difference(y_test, probabilities)
# print("Risk Difference:", risk_diff)

risk_diff_values = [calculate_risk_difference(y, model.predict_proba(X)[:, 1]) for (X, y) in [(X_train, y_train), (X_test, y_test)]]
risk_diff_mean_std = (np.mean(risk_diff_values), np.std(risk_diff_values))
print(f"Risk Difference Mean: {risk_diff_mean_std[0]:f}, Std: {risk_diff_mean_std[1]:.4f}")

