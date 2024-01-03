import numpy as np
from lightphe import LightPHE


def calculate_slope(x, y, x_mean, y_mean):
    numerator = 0
    denominator = 0

    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2

    return numerator / denominator


def calculate_intercept(x_mean, y_mean, slope):
    return y_mean - slope * x_mean


def linear_regression(x, y, pal):
    sum_X = pal.encrypt(0)
    sum_y = pal.encrypt(0)

    for i in range(len(X)):
        sum_X += Pal_X[i]
        sum_y += Pal_y[i]

    x_mean = pal.decrypt(sum_X) / len(X)
    y_mean = pal.decrypt(sum_y) / len(y)

    slope = calculate_slope(x, y, x_mean, y_mean)
    intercept = calculate_intercept(x_mean, y_mean, slope)

    return slope, intercept


algorithms = [
    "RSA",
    "ElGamal",
    "Exponential-ElGamal",
    "Paillier",
    "Damgard-Jurik",
    "Okamoto-Uchiyama",
    "Benaloh",
    "Naccache-Stern",
    "Goldwasser-Micali",
    "EllipticCurve-ElGamal",
]

pal = LightPHE(algorithm_name="Paillier")
rsa = LightPHE(algorithm_name="RSA")


# Example data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

# encrypt X and y using Paillier and RSA schemes. go one element at a time
Pal_X = []
Pal_y = []
Rsa_X = []
Rsa_y = []

for i in range(len(X)):
    Pal_X.append(pal.encrypt(int(X[i])))
    Rsa_X.append(rsa.encrypt(int(X[i])))

for i in range(len(y)):
    Pal_y.append(pal.encrypt(int(y[i])))
    Rsa_y.append(rsa.encrypt(int(y[i])))

sum_X = pal.encrypt(0)

for i in range(len(X)):
    sum_X += Pal_X[i]

print(pal.decrypt(sum_X) / len(X))

slope, intercept = linear_regression(Pal_X, Pal_y, pal)

print(f"Slope: {slope}, Intercept: {intercept}")


# Perform linear regression
# slope, intercept = linear_regression(Pal_X, Pal_y, Rsa_X, Rsa_y)

# Print the results
# print(f"Slope (m): {slope}")
# print(f"Intercept (b): {intercept}")
