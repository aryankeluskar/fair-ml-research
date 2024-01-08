import random
import tenseal as ts
from sklearn.linear_model import LinearRegression
import time

class LinRegCKKS:
    def __init__(self):
        self.poly_mod_degree = 4096
        self.coeff_mod_bit_sizes = [40, 20, 40]
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, self.poly_mod_degree, -1, self.coeff_mod_bit_sizes)
        self.context.global_scale = 2 ** 20
        self.context.generate_galois_keys()

    def initialize_data(self, n):
        # Generate random data for x and y
        self.x = [random.uniform(1, 10) for _ in range(n)]
        self.y = [2 * xi + 5 + random.uniform(-2, 2) for xi in self.x]

        self.x_enc = []
        self.y_enc = []
        for i in range(len(self.x)):
            self.x_enc.append(ts.ckks_vector(self.context, [self.x[i]]))
            self.y_enc.append(ts.ckks_vector(self.context, [self.y[i]]))
        

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

    def fit(self):
        zero_enc = ts.ckks_vector(self.context, [0])
        len_enc = ts.ckks_vector(self.context, [len(self.x)])

        x_mean = self.calculate_sum_ckks(self.x_enc, zero_enc)._decrypt()[0] / len(self.x)
        y_mean = self.calculate_sum_ckks(self.y_enc, zero_enc)._decrypt()[0] / len(self.y)

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
    

# Example usage for single time with fixed data points
# x = [10, 2, 3, 4, 5, 11]  # Modify the x-coordinate array values here
# y = [20, 3, 4, 5, 6, 11]  # Modify the y-coordinate array values here

# linreg = LinRegCKKS(x, y)
# slope, intercept = linreg.fit()
# print(f"Slope: {slope}, Intercept: {intercept}")
    
start = time.time()
num_runs = 5
mse_sum = 0
accuracy_sum = 0

for i in range(num_runs):
    data_points = 20000
    regression = LinRegCKKS()
    regression.initialize_data(data_points)
    slope, intercept = regression.fit()

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

average_mse = mse_sum / num_runs
average_accuracy = accuracy_sum / num_runs

print(f"Average Mean Squared Error (MSE): {average_mse}")
print(f"Average Accuracy: {average_accuracy:.5f}%")
end = time.time()
print(f"Average Time taken per run: {(end - start)/num_runs} seconds")