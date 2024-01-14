import pandas as pd
import numpy as np

# Defining column headers for the German Credit data
columns = [
    "status_checking_account",
    "duration_month",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits_at_this_bank",
    "job",
    "people_liable_maintenance",
    "telephone",
    "foreign_worker",
    "classification",
]

# Read the German Credit data from the provided file
german_credit = pd.read_csv(
    "./german.data", delimiter=" ", engine="python", header=None, names=columns
)

# Convert categorical columns with binary values to pure binary
binary_columns = [
    "status_checking_account",
    "personal_status_sex",
    "other_debtors_guarantors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
    "classification",
]
for column in binary_columns:
    german_credit[column] = german_credit[column].map({"A11": 0, "A12": 1, "A13": 2, "A14": 3, "A30": 0, "A31": 1, "A32": 2, "A33": 3, "A34": 4,
                                                      "A40": 0, "A41": 1, "A42": 2, "A43": 3, "A44": 4, "A45": 5, "A46": 6, "A47": 7, "A48": 8, "A49": 9, "A410": 10,
                                                      "A61": 0, "A62": 1, "A63": 2, "A64": 3, "A65": 4, "A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4,
                                                      "A91": 0, "A92": 1, "A93": 2, "A94": 3, "A95": 4, "A101": 0, "A102": 1, "A103": 2, "A121": 0, "A122": 1, "A123": 2, "A124": 3,
                                                      "A141": 0, "A142": 1, "A143": 2, "A151": 0, "A152": 1, "A153": 2, "A171": 0, "A172": 1, "A173": 2, "A174": 3,
                                                      "A191": 0, "A192": 1, "A201": 0, "A202": 1,
                                                      })

# export German Credit data to csv and save as german_credit_final.csv
german_credit.to_csv('german.csv', index=False)
