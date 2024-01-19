import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# Step 1: Read the data
file_path = 'path/to/boston.txt'
data = pd.read_csv(file_path, sep='\s+', skiprows=22, header=None, names=[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
])

# Step 2: Prepare the data
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Create the 'sklearn/graphs' directory if it doesn't exist
graphs_dir = 'sklearn/graphs'
os.makedirs(graphs_dir, exist_ok=True)

# Step 3 & 4: Perform linear regression and create graphs
for feature_name in X.columns:
    # Extract the feature variable
    feature_variable = X[feature_name].values.reshape(-1, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_variable, y, test_size=0.2, random_state=42)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 5: Create and save the graph
    plt.scatter(X_test, y_test, color='blue', label='Actual values')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
    plt.xlabel(feature_name)
    plt.ylabel('Median Home Price')
    plt.legend()
    plt.title(f'{feature_name} vs Median Home Price')
    
    # Save the graph
    graph_filename = os.path.join(graphs_dir, f'{feature_name}.png')
    plt.savefig(graph_filename)
    plt.close()

print("Graphs saved successfully.")
