import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import random
import time
import pandas as pd


class LinRegNormal:
    def __init__(self):
        self.x = []
        self.y = []
        self.slope = 0
        self.intercept = 0

    def initialize_data(self, feature_name):
        # Read the CSV file
        data = pd.read_csv("LinReg/bostonPreProcessed.csv")

        MEDV = data["MEDV"]

        # Store the extracted columns in x and y
        self.x = data[feature_name]
        self.y = MEDV
        # print(self.x)
        # print(self.y)

    # def initialize_data(self, n):
    #     # Generate random data for x and y
    #     self.x = [random.uniform(1, 10) for _ in range(n)]
    #     self.y = [2 * xi + 5 + random.uniform(-2, 2) for xi in self.x]

    #     print(self.x)

        

    def calculate_slope(self):
        numerator = 0
        denominator = 0
        x_mean = self.calculate_mean(self.x)
        y_mean = self.calculate_mean(self.y)

        for i in range(len(self.x)):
            numerator += (self.x[i] - x_mean) * (self.y[i] - y_mean)
            denominator += (self.x[i] - x_mean) ** 2

        return numerator / denominator

    def calculate_mean(self, arr):
        return sum(arr) / len(arr)

    def calculate_intercept(self, slope):
        x_mean = self.calculate_mean(self.x)
        y_mean = self.calculate_mean(self.y)

        return y_mean - slope * x_mean

    def predict(self, x):
        # Predict y values using the linear regression equation
        return [self.slope * xi + self.intercept for xi in x]

    def linear_regression(self):
        self.slope = self.calculate_slope()
        self.intercept = self.calculate_intercept(slope=self.slope)

        return self.slope, self.intercept

    def calculate_mse(self, predicted, actual):
        # Mean Squared Error (MSE) calculation
        n = len(actual)
        mse = sum((predicted[i] - actual[i]) ** 2 for i in range(n)) / n
        return mse

    def calculate_r_squared(self, predicted, actual):
        # R-squared calculation
        mean_actual = sum(actual) / len(actual)
        ss_total = sum((actual[i] - mean_actual) ** 2 for i in range(len(actual)))
        ss_residual = sum((actual[i] - predicted[i]) ** 2 for i in range(len(actual)))
        r_squared = 1 - (ss_residual / ss_total)
        return r_squared

    def make_graph(self, predicted, actual, feature_name):
        graphs_dir = 'LinReg/Plain'
        os.makedirs(graphs_dir, exist_ok=True)

        # Plot the graph
        plt.scatter(self.x, self.y, color='white', label='Actual values')
        plt.plot(self.x, predicted, color='green', linewidth=2, label='Regression line')
        plt.xlabel(feature_name)
        plt.ylabel('Median Home Price')
        plt.legend()
        plt.title(f'{feature_name} vs Median Home Price')

        # Save the graph
        plt.savefig(f'LinReg/Plain/{feature_name}.png')
        plt.close()
        return min(self.x), max(self.x), min(self.y), max(self.y)

# Example usage for single time with fixed data points
# x = [10, 2, 3, 4, 5, 11]
# y = [20, 3, 4, 5, 6, 11]

# regression = LinRegNormal()
# regression.initialize_data(x, y)
# slope, intercept = regression.linear_regression()

def runLinRegPlain():
    feature_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    for i in feature_list:
        regression = LinRegNormal()
        regression.initialize_data(i)
        slope, intercept = regression.linear_regression()

        # Make predictions
        predictions = regression.predict(regression.x)

        # Calculate Mean Squared Error (MSE)
        mse = regression.calculate_mse(predictions, regression.y)

        # Calculate R-squared
        r_squared = regression.calculate_r_squared(predictions, regression.y)
        # Print accuracy as a percentage
        accuracy_percentage = r_squared * 100

        x_min, x_max, y_min, y_max = regression.make_graph(predictions, regression.y, i)

        with open('LinReg/PlainResults.txt', 'a') as file:
            file.write(f"Feature: {i}\n")
            file.write(f"Accuracy: {accuracy_percentage}\n")
            file.write(f"MSE: {mse}\n")
            file.write(f"Slope: {slope}\n")
            file.write(f"Intercept: {intercept}\n")
            file.write(f"x_min: {x_min}\n")
            file.write(f"x_max: {x_max}\n")
            file.write(f"y_min: {y_min}\n")
            file.write(f"y_max: {y_max}\n")
            file.write("\n")

runLinRegPlain()