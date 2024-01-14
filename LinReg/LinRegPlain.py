import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import random
import time


class LinRegNormal:
    def __init__(self):
        self.x = []
        self.y = []
        self.slope = 0
        self.intercept = 0

    def initialize_data(self, n):
        # Generate random data for x and y
        self.x = [random.uniform(1, 10) for _ in range(n)]
        self.y = [2 * xi + 5 + random.uniform(-2, 2) for xi in self.x]

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


# Example usage for single time with fixed data points
# x = [10, 2, 3, 4, 5, 11]
# y = [20, 3, 4, 5, 6, 11]

# regression = LinRegNormal()
# regression.initialize_data(x, y)
# slope, intercept = regression.linear_regression()

start = time.time()

# Define the number of repetitions for the experiment here
ITERATIONS = 20
mse_sum = 0
accuracy_sum = 0

for i in range(ITERATIONS):
    data_points = 10000
    regression = LinRegNormal()
    regression.initialize_data(data_points)
    slope, intercept = regression.linear_regression()

    # Make predictions
    predictions = regression.predict(regression.x)

    # Calculate Mean Squared Error (MSE)
    mse = regression.calculate_mse(predictions, regression.y)
    mse_sum += mse

    # Calculate R-squared
    r_squared = regression.calculate_r_squared(predictions, regression.y)
    # Print accuracy as a percentage
    accuracy_percentage = r_squared * 100
    accuracy_sum += accuracy_percentage

    # print(f"Slope: {slope}, Intercept: {intercept}")
    print(f"In Run {i+1}: Mean Squared Error (MSE): {mse}")
    print(f"In Run {i+1}: Accuracy: {accuracy_percentage:.5f}%")
    print()

    with open("LinReg/PlainLinRegResults.txt", "a") as f:
        f.write(f"In Run {i+1}: Mean Squared Error (MSE): {mse}\n")
        f.write(f"In Run {i+1}: Accuracy: {accuracy_percentage:.5f}%\n\n")

average_mse = mse_sum / ITERATIONS
average_accuracy = accuracy_sum / ITERATIONS

print(f"Average Mean Squared Error (MSE): {average_mse}")
print(f"Average Accuracy: {average_accuracy:.5f}%")
with open("LinReg/PlainLinRegResults.txt", "a") as f:
    f.write(f"Average Mean Squared Error (MSE): {average_mse}\n")
    f.write(f"Average Accuracy: {average_accuracy:.5f}%\n\n")
end = time.time()
print(f"Average Time taken per run: {(end - start)/ITERATIONS} seconds")

