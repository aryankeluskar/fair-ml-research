import random
import tenseal as ts
from sklearn.linear_model import LinearRegression
import time
import pandas as pd
import os
import matplotlib.pyplot as plt

class LinRegCKKS:
    def __init__(self):
        self.poly_mod_degree = 4096
        self.coeff_mod_bit_sizes = [40, 20, 40]
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, self.poly_mod_degree, -1, self.coeff_mod_bit_sizes)
        self.context.global_scale = 2 ** 20
        self.context.generate_galois_keys()

    def initialize_data(self, feature_name):
        data = pd.read_csv("LinReg/bostonPreProcessed.csv")
        # print(data["AGE"])
        print("current feature: ", feature_name)        

        MEDV = data["MEDV"]

        # Store the extracted columns in x and y
        self.x = data[feature_name]
        self.y = MEDV

        self.x_enc = []
        self.y_enc = []
        for i in range(len(self.x)):
            self.x_enc.append(ts.ckks_vector(self.context, [self.x[i]]))
            self.y_enc.append(ts.ckks_vector(self.context, [self.y[i]]))

        # destroying the unencrypted data
        # self.x = None
        # self.y = None
        

    def calculate_sum_ckks(self, arr, zero_enc):
        sum = zero_enc.copy()
        for i in arr:
            sum += i
        return sum

    def calculate_slope_ckks(self, x_enc, y_enc, x_mean_enc, y_mean_enc, zero_enc):
        numerator = zero_enc.copy()
        denominator = zero_enc.copy()

        for i in range(len(x_enc)):
            x_diff = x_enc[i] - x_mean_enc
            y_diff = y_enc[i] - y_mean_enc
            numerator += (x_diff * y_diff)
            denominator += (x_diff * x_diff)

        return numerator._decrypt()[0] / denominator._decrypt()[0]

    def calculate_intercept_ckks(self, x_mean, y_mean, slope):
        return y_mean - slope * x_mean

    def linear_regression(self):
        zero_enc = ts.ckks_vector(self.context, [0])
        # len_enc = ts.ckks_vector(self.context, [len(self.x_enc)])

        x_mean = self.calculate_sum_ckks(self.x_enc, zero_enc)._decrypt()[0] / len(self.x_enc)
        y_mean = self.calculate_sum_ckks(self.y_enc, zero_enc)._decrypt()[0] / len(self.y_enc)

        x_mean_enc = ts.ckks_vector(self.context, [x_mean])
        y_mean_enc = ts.ckks_vector(self.context, [y_mean])

        self.slope = self.calculate_slope_ckks(self.x_enc, self.y_enc, x_mean_enc, y_mean_enc, zero_enc)
        self.intercept = self.calculate_intercept_ckks(x_mean, y_mean, self.slope)

        return self.slope, self.intercept
    
    def predict(self, x):
        # Predict y values using the linear regression equation
        return [self.slope * xi + self.intercept for xi in x]
    
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
        graphs_dir = 'LinReg/CKKS'
        os.makedirs(graphs_dir, exist_ok=True)

        # Plot the graph
        plt.scatter(self.x, self.y, color='white', label='Actual values')
        plt.plot(self.x, predicted, color='green', linewidth=2, label='Regression line')
        plt.xlabel(feature_name)
        plt.ylabel('Median Home Price')
        plt.legend()
        plt.title(f'{feature_name} vs Median Home Price')

        # Save the graph
        plt.savefig(f'LinReg/CKKS/{feature_name}.png')
        plt.close()
        return min(self.x), max(self.x), min(self.y), max(self.y)


# Example usage for single time with fixed data points
# x = [10, 2, 3, 4, 5, 11]  # Modify the x-coordinate array values here
# y = [20, 3, 4, 5, 6, 11]  # Modify the y-coordinate array values here

# linreg = LinRegCKKS(x, y)
# slope, intercept = linreg.linear_regression()
# print(f"Slope: {slope}, Intercept: {intercept}")
    

# data_points = 1000
# regression = LinRegCKKS()
# slope, intercept = regression.linear_regression()

# # Make predictions
# predictions = regression.predict(regression.x)

# # Calculate Mean Squared Error (MSE)
# mse = regression.calculate_mse(predictions, regression.y)
# # mse_sum += mse

# # Calculate R-squared
# r_squared = regression.calculate_r_squared(predictions, regression.y)
# # Print accuracy as a percentage
# accuracy_percentage = r_squared * 100
# accuracy_sum += accuracy_percentage

# print(f"Slope: {slope}, Intercept: {intercept}")
# print(f"In Run {i+1}: Mean Squared Error (MSE): {mse}")
# print(f"In Run {i+1}: Accuracy: {accuracy_percentage:.5f}%")
print()
# with open("LinReg/CKKSLinRegResults2.txt", "a") as f:
#     f.write(f"In Run {i+1}: Mean Squared Error (MSE): {mse}\n")
#     f.write(f"In Run {i+1}: Accuracy: {accuracy_percentage:.5f}%\n\n")

def runLinRegckks():
    feature_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    for i in feature_list:
        regression = LinRegCKKS()
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

        with open('LinReg/CKKSResults.txt', 'a') as file:
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


runLinRegckks()