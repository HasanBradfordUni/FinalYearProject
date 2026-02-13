import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')

    # Selected features INCLUDING placement type
    feature_cols = [
        "Child Age At Placement", "Child Gender", "Child Ethnicity", "Carer Age",
        "Carer Gender Composition", "Carer Ethnicity Or Religion", "Placement Type"
    ]

    x = data[feature_cols].copy()

    # Categorical input columns (safe to assign Unknown)
    input_categorical_columns = [
        "Child Gender", "Child Ethnicity",
        "Carer Gender Composition", "Carer Ethnicity Or Religion"
    ]

    # Target categorical column (DO NOT assign Unknown)
    target_column = "Placement Type"

    # Fill missing categorical values for INPUT features only
    for col in input_categorical_columns:
        x[col] = x[col].astype(str).fillna("Unknown")

    # Numeric columns: convert + fill missing with median
    numeric_columns = ["Child Age At Placement", "Carer Age"]

    for col in numeric_columns:
        x[col] = pd.to_numeric(x[col], errors='coerce')
        median_val = x[col].median()
        x[col] = x[col].fillna(median_val)

    # Create encoders
    encoders = {}

    # Encode input categorical columns
    for col in input_categorical_columns:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col].astype(str))
        encoders[col] = le

    # Encode Placement Type separately (no Unknown allowed)
    placement_le = LabelEncoder()
    x[target_column] = placement_le.fit_transform(x[target_column].astype(str))
    placement_encoder = placement_le

    # Classification target
    y_classification = x[target_column].values

    # Regression target
    y_regression = pd.to_numeric(
        data["Placement Time Period (days)"], errors='coerce'
    ).fillna(0)

    # Remove placement type from X for classification
    x_class = x.drop(columns=[target_column])

    return x, x_class, y_classification, y_regression, encoders, placement_le

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

# Main function that loads the dataset, trains the models and saves them via joblib
def main():
    x_reg, x_class, y_class, y_reg, encoders, placement_encoder = load_data("dataset.csv")

    os.makedirs("models", exist_ok=True)

    # Train models
    lr_model = run_linear_regression(x_reg, y_reg)
    rf_model = run_random_forest(x_class, y_class)

    # Save models and encoders
    joblib.dump(lr_model, "models/lr_regressor.pkl")
    joblib.dump(rf_model, "models/rf_classifier.pkl")
    joblib.dump(encoders, "models/feature_encoders.pkl")
    joblib.dump(placement_encoder, "models/placement_encoder.pkl")

    print("Models and encoders saved successfully.")

if __name__ == "__main__":
    main()