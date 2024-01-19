import csv
# so i have a datset in boston.txt stored as 
#  0.04527   0.00  11.930  0  0.5730  6.1200  76.70  2.2875   1  273.0  21.00  396.90   9.08  20.60
#  0.06076   0.00  11.930  0  0.5730  6.9760  91.00  2.1675   1  273.0  21.00  396.90   5.64  23.90
#  0.10959   0.00  11.930  0  0.5730  6.7940  89.30  2.3889   1  273.0  21.00  393.45   6.48  22.00
#  0.04741   0.00  11.930  0  0.5730  6.0300  80.80  2.5050   1  273.0  21.00  396.90   7.88  11.90

# the vars are 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
# use this info to convert whole of boston.txt to a csv file names bostonPreProcessed.csv

input_file = 'LinReg/boston.txt'
output_file = 'LinReg/bostonPreProcessed.csv'

with open(input_file, 'r') as file:
    lines = file.readlines()

data = [line.strip().split() for line in lines]

header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)
