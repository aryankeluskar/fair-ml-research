import ucimlrepo

# fetch dataset
adult = ucimlrepo.fetch_ucirepo(id=2)

# store data in adult dataframe
adult_data = adult.data.features
adult_target = adult.data.targets
print(adult_data)

# save adult_data locally as csv file
adult_data.to_csv('adult_data.csv', index=False)