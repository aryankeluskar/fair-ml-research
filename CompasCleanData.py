import pandas as pd
import re

compas = pd.read_csv(
    "./compas.csv", delimiter=",", engine="python"
)

# drop columns like Person_ID,AssessmentID,Case_ID,Agency_Text,LastName,FirstName,MiddleName

compas = compas.drop(columns=['Person_ID','AssessmentID','Case_ID','Agency_Text','LastName','FirstName','MiddleName'])


# Remove the rows with any missing value. Missing value can also be marked with a ? in the data
compas = compas.replace("?", pd.NA)
compas = compas.dropna()

# Birthdate is stored in two formats: 12-05-1992 and 09/16/84. so convert all to 12-05-1992 format
compas['DateOfBirth'] = compas['DateOfBirth'].str.replace('/', '-')
compas['DateOfBirth'] = compas['DateOfBirth'].apply(lambda x: re.sub(r'(\d{2})$', r'19\1', x))

# same for Screening_Date
compas['Screening_Date'] = compas['Screening_Date'].str.replace('/', '-')

print(compas['Screening_Date'].head())

# only for ScoreText, high is 2, medium is 1, low is 0
compas['ScoreText'] = compas['ScoreText'].map({'High': 1, 'Medium': 0.5, 'Low': 0})

# today is 2017-11-21, store age as age in 2017 by subtracting DateOfBirth from today and then remove DateOfBirth
compas['DateOfBirth'] = pd.to_datetime(compas['DateOfBirth'])
compas['DateOfBirth'] = compas['DateOfBirth'].dt.year
compas['age'] = 2017 - compas['DateOfBirth']
compas = compas.drop(columns=['DateOfBirth'])

# today is 2017-11-21, store daysSinceScreening as daysSinceScreening by subtracting Screening_Date from today and then remove Screening_Date
compas['Screening_Date'] = pd.to_datetime(compas['Screening_Date'])
compas['daysSinceScreening'] = (pd.to_datetime('today') - compas['Screening_Date']).dt.days
compas = compas.drop(columns=['Screening_Date'])

# export compas to csv and save as compasFinal.csv
compas.to_csv('compasFinal.csv', index=False)

