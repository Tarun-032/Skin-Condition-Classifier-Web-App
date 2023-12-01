import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load the Dermatology Database
file_path = '/home/tarun003/Chatbot/dermatologydata.csv'
column_names = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules', 
                'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement', 
                'family_history', 'melanin_incontinence', 'eosinophils_infiltrate', 'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis', 
                'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges', 
                'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis', 
                'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer', 'spongiosis', 'saw-tooth_appearance_of_retes', 
                'follicular_horn_plug', 'perifollicular_parakeratosis', 'inflammatory_mononuclear_inflitrate', 'band-like_infiltrate', 
                'age', 'class']

dermatology_data = pd.read_csv(file_path, header=None, names=column_names)
dermatology_data.replace('?', pd.NA, inplace=True)
dermatology_data = dermatology_data.apply(pd.to_numeric, errors='ignore')
dermatology_data.fillna(dermatology_data.mean(), inplace=True)
categorical_columns = ['family_history']
dermatology_data = pd.get_dummies(dermatology_data, columns=categorical_columns)
label_encoder = LabelEncoder()
dermatology_data['class'] = label_encoder.fit_transform(dermatology_data['class'])



X = dermatology_data.drop('class', axis=1)
y = dermatology_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature Analysis
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['importance'])
sorted_features = feature_importances.sort_values(by='importance', ascending=False)
print("Feature Importances:")
print(sorted_features)

# Select top features (you can adjust this threshold)
# Assuming top_features is a list of column names
top_features = ['clubbing_of_the_rete_ridges', 'fibrosis_of_the_papillary_dermis', 'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis', 'koebner_phenomenon', 'band-like_infiltrate']
top_features = sorted_features[sorted_features['importance'] > 0.02].index
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model_tuned = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model_tuned, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_top, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Get the best model
best_rf_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test_top)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with Top Features: {accuracy}")
