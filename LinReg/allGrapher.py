import matplotlib.pyplot as plt
import numpy as np
import os

def read_results(file_path, feature):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = {}
    slope = None
    intercept = None
    max = None
    min = None
    current_feature = None
    for line in lines:
        if line == "\n":
            continue
        if line.startswith("Feature"):
            # print("Feature: ", line.split(": ")[1].strip())
            current_feature = line.split(": ")[1].strip()
            data[current_feature] = {}
        else:
            key, value = line.split(": ")
            if key != "Accuracy" and key != "MSE":
                if current_feature == feature:
                    # print(key, value)
                    if key == "Slope":
                        slope = float(value)
                    elif key == "Intercept":
                        intercept = float(value)
                    elif key == "x_max":
                        max = float(value)
                    elif key == "x_min":
                        min = float(value)


    return slope, intercept, max, min

feature_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

for feature in feature_list:
    skslope, skintercept, x_max, x_min = read_results("LinReg/sklearnResults.txt", feature)
    plslope, plintercept, x_max, x_min = read_results("LinReg/plainResults.txt", feature)
    ckkslope, ckkintercept, x_max, x_min = read_results("LinReg/CKKSResults.txt", feature)

    x_values = np.linspace(x_min, x_max, 100)
    sk_y_values = skslope * x_values + skintercept
    pl_y_values = plslope * x_values + plintercept
    ckks_y_values = ckkslope * x_values + ckkintercept

    WIDTH = 3

    plt.plot(x_values, sk_y_values, color='red', linewidth=WIDTH, label='sklearn', alpha=0.2)
    plt.plot(x_values, pl_y_values, color='blue', linewidth=WIDTH, label='plain', alpha=0.2)
    # plt.plot(x_values, ckks_y_values, color='green', linewidth=WIDTH, label='CKKS', alpha=0.2)

    plt.xlabel(f"{feature}")
    plt.ylabel('Median Home Price')
    plt.legend()
    plt.title(f'{feature} vs Median Home Price')
    # plt.show()
    os.makedirs("LinReg/sklearn_plain", exist_ok=True)
    plt.savefig(f"LinReg/sklearn_plain/{feature}.png")
    plt.close()