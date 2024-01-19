import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import os

def runLinRegSklearn():
    # Step 1: Read the data from bostonPreProcessed.csv, 
    data = pd.read_csv("LinReg/bostonPreProcessed.csv")

    # Step 2: Prepare the data
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']

    # Create the 'sklearn/graphs' directory if it doesn't exist
    graphs_dir = 'LinReg/sklearn'
    os.makedirs(graphs_dir, exist_ok=True)

    # Step 3 & 4: Perform linear regression and create graphs
    results = []
    for feature_name in X.columns:
        # Extract the feature variable
        feature_variable = X[feature_name].values.reshape(-1, 1)

                # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Calculate mean squared error
        mse = mean_squared_error(y_test, y_pred)

        # Get slope and intercept
        slope = model.coef_[0]
        intercept = model.intercept_

        # Step 5: Create and save the graph
        plt.scatter(X_test, y_test, color='white', label='Actual values')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
        plt.xlabel(feature_name)
        plt.ylabel('Median Home Price')
        plt.legend()
        plt.title(f'{feature_name} vs Median Home Price')
        
        # Save the graph
        graph_filename = os.path.join(graphs_dir, f'{feature_name}.png')
        plt.savefig(graph_filename)
        plt.close()

        # Append results to the list
        results.append({
            'Feature': feature_name,
            'Accuracy': model.score(X_test, y_test),
            'MSE': mse,
            'Slope': slope,
            'Intercept': intercept
        })

    # Save the results to sklearnResults.txt
    results_filename = 'LinReg/sklearnResults.txt'
    with open(results_filename, 'w') as file:
        for result in results:
            file.write(f"Feature: {result['Feature']}\n")
            file.write(f"Accuracy: {result['Accuracy']}\n")
            file.write(f"MSE: {result['MSE']}\n")
            file.write(f"Slope: {result['Slope']}\n")
            file.write(f"Intercept: {result['Intercept']}\n")
            file.write("\n")

    print("Graphs saved successfully.")


if __name__ == "__main__":
    runLinRegSklearn()