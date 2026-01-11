import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')

    # Selected features
    feature_cols = [
        "Child Age At Placement", "Child Gender", "Child Ethnicity",
        "Carer Age", "Carer Gender Composition", "Carer Ethnicity Or Religion"
    ]

    x = data[feature_cols].copy()

    # Create a dictionary of encoders
    encoders = {}
    categorical_columns = [
        "Child Gender", "Child Ethnicity",
        "Carer Gender Composition", "Carer Ethnicity Or Religion"
    ]

    for col in categorical_columns:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col].astype(str))
        encoders[col] = le

    # Ensure numeric columns are numeric
    x["Child Age At Placement"] = pd.to_numeric(x["Child Age At Placement"], errors='coerce').fillna(0)
    x["Carer Age"] = pd.to_numeric(x["Carer Age"], errors='coerce').fillna(0)

    # Targets
    placement_type_encoder = LabelEncoder()
    y_classification = placement_type_encoder.fit_transform(data["Placement Type"].astype(str))

    y_regression = pd.to_numeric(data["Placement Time Period (days)"], errors='coerce').fillna(0)

    return x, y_classification, y_regression, encoders, placement_type_encoder

# Train the Linear Regression model
def run_linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model

# Train the Random Forest Classifier model
def run_random_forest(x, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x, y)
    return model

# Main function that loads the dataset, trains the models and saves them via joblib to be used in the app
def main():
    x, y_class, y_reg, encoders, placement_encoder = load_data("dataset.csv")
    os.makedirs("models", exist_ok=True) #Create models directory if it doesn't exist

    # Train models
    lr_model = run_linear_regression(x, y_reg)
    rf_model = run_random_forest(x, y_class)

    # Save models
    joblib.dump(lr_model, "models/lr_regressor.pkl")
    joblib.dump(rf_model, "models/rf_classifier.pkl")

    # Save encoders
    joblib.dump(encoders, "models/feature_encoders.pkl")
    joblib.dump(placement_encoder, "models/placement_encoder.pkl")

    print("Models and encoders saved successfully.")

# Execute main function if this script is run directly
if __name__ == "__main__":
    main()

