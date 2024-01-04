import numpy as np

# Global variables
x = []
y = []

def initialize_data():
    global x, y
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 4, 5, 6]

def calculate_slope():
    global x, y
    numerator = 0
    denominator = 0
    x_mean = calculate_mean(x)
    y_mean = calculate_mean(y)

    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2

    return numerator / denominator

def calculate_mean(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
    return sum / len(arr)

def calculate_intercept():
    global x, y
    x_mean = calculate_mean(x)
    y_mean = calculate_mean(y)
    slope = calculate_slope()

    return y_mean - slope * x_mean


def linear_regression():
    global x, y
    initialize_data()
    slope = calculate_slope()
    intercept = calculate_intercept()

    return slope, intercept


# Example data
slope, intercept = linear_regression()

print(f"Slope: {slope}, Intercept: {intercept}")
