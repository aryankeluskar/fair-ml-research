import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load your CSV file into a pandas DataFrame
df = pd.read_csv('compasFinal.csv')

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')  # Drop first category to avoid multicollinearity

# Iterate over each categorical column and one-hot encode it
for col in categorical_cols:
    
    # Extract the column to be one-hot encoded
    column_data = df[[col]]

    # Fit and transform the selected column
    encoded_data = encoder.fit_transform(column_data)

    # Create a DataFrame with the encoded values
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))

    # Concatenate the original DataFrame with the encoded DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    # Drop the original categorical column
    df = df.drop([col], axis=1)

# Now, your DataFrame 'df' contains one-hot encoded values for all categorical columns
print(df.head())
# save df to csv named adultPreprocessed.csv
df.to_csv('compasPreprocessed2.csv', index=False)