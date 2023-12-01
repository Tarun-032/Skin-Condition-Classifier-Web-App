import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the Dermatology Database
file_path = '/home/tarun003/Chatbot/dermatologydata.csv'
column_names = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'family_history', 'class']
dermatology_data = pd.read_csv(file_path, header=None, names=column_names)

# Replace non-numeric values with NaN
dermatology_data.replace('?', pd.NA, inplace=True)

# Convert boolean columns to numeric (True/False to 1/0)
dermatology_data['family_history'] = (dermatology_data['family_history'] == 'True').astype(int)

# Convert the entire DataFrame to numeric type
dermatology_data = dermatology_data.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
dermatology_data.dropna(inplace=True)

# Encode labels
label_encoder = LabelEncoder()
dermatology_data['class'] = label_encoder.fit_transform(dermatology_data['class'])

# Save the processed data as CSV
dermatology_data.to_csv('dermatology_processed.csv', index=False)
