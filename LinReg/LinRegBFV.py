# using bfv to encrypt the arrays x and y. the encrypted arrays and tenseal context are passed to the function linear_regression_bfv
import tenseal as ts
x_enc = []
y_enc = []

def initialize_data():
    global x_enc, y_enc
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 4, 5, 6]
    context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
    x_enc = ts.bfv_vector(context, x)
    y_enc = ts.bfv_vector(context, y)

def linear_regression_bfv():
    global x_enc, y_enc
    slope = calculate_slope_bfv()
    intercept = calculate_intercept_bfv(slope)
    return slope, intercept


def calculate_sum_bfv(arr, zero_enc):
    sum = zero_enc.copy()
    for i in arr:
        # print(i)
        sum += i
    
    return sum 

def calculate_slope_bfv(x_mean_enc, y_mean_enc):
    global x_enc, y_enc
    numerator = zero_enc.copy()
    denominator = zero_enc.copy()

    for i in range(len(x_enc)):
        x_diff = x_enc[i] - x_mean_enc
        y_diff = y_enc[i] - y_mean_enc
        numerator += x_diff * y_diff
        denominator += x_diff * x_diff

    return numerator._decrypt()[0]/ denominator._decrypt()[0]

def calculate_intercept_bfv(x_mean_enc, y_mean_enc, slope):
    return y_mean - slope * x_mean


x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
x_enc = []
y_enc = []
for i in range(len(x)):
    x_enc.append(ts.bfv_vector(context, [x[i]]))
    y_enc.append(ts.bfv_vector(context, [y[i]]))

# print(x_enc)
zero_enc = ts.bfv_vector(context, [0])
one_enc = ts.bfv_vector(context, [1])
len_enc = ts.bfv_vector(context, [len(x)])

x_mean = calculate_sum_bfv(x_enc, zero_enc)._decrypt()[0]/len(x)
y_mean = calculate_sum_bfv(y_enc, zero_enc)._decrypt()[0]/len(y)

print(x_mean)
print(y_mean)

x_mean_enc = ts.bfv_vector(context, [x_mean])
y_mean_enc = ts.bfv_vector(context, [y_mean])
# print(x_mean_enc.decrypt())
slope = calculate_slope_bfv(x_mean_enc, y_mean_enc)
print("slope: ", slope)
print("intercept: ", calculate_intercept_bfv(x_mean, y_mean, slope))