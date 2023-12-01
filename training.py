import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib 
from sklearn.preprocessing import LabelEncoder

# Load preprocessed data
data = pd.read_csv('dermatology_processed.csv')

# Confirm data types
print(data.dtypes)

# Check unique values in the 'erythema' column
print(data['erythema'].unique())

# Print a few rows to inspect the data
print(data.head())

# Explicitly convert 'erythema' column to numeric
data['erythema'] = pd.to_numeric(data['erythema'], errors='coerce')

# Separate features (X) and target variable (y)
X = data.drop('class', axis=1)
y = data['class']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model (optional)
import joblib
joblib.dump(rf_model, 'dermatology_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')