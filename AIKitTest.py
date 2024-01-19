from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.metrics import *

data = fetch_adult()

print(type(data))


X = data.data
y_true = (data.target == ">50K") * 1
sex = X["sex"]

selection_rates = MetricFrame(
    metrics=selection_rate, y_true=y_true, y_pred=y_true, sensitive_features=sex
)

fig = selection_rates.by_group.plot.bar(
    legend=False, rot=0, title="Fraction earning over $50,000"
)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get predictions on the test set
y_pred = model.predict(X_test)

print("Accuracy: ", model.score(X_test, y_test))