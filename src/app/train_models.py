import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "Child Age At Placement",
    "Child Gender",
    "Child Ethnicity",
    "Child Prior Placements",
    "Returning Child",
    "Missing Episodes",
    "Sibling Group Size",
    "Placed With Siblings",
    "Carer Age",
    "Carer Gender",
    "Carer Ethnicity",
    "EH involvement",
    "YOT involvement",
    "Placement Sequence Number",
]
CATEGORICAL_COLUMNS = ["Child Gender", "Child Ethnicity", "Carer Gender", "Carer Ethnicity"]
BOOLEAN_COLUMNS = ["Returning Child", "Placed With Siblings", "EH involvement", "YOT involvement"]
NUMERIC_COLUMNS = [
    "Child Age At Placement",
    "Child Prior Placements",
    "Missing Episodes",
    "Sibling Group Size",
    "Carer Age",
    "Placement Sequence Number",
]
PLACEMENT_COLUMN = "Placement Type"
REGRESSION_TARGET = "Days Placed"

def _resolve_dataset_path():
    # train_models.py lives in src/app, so dataset is in src/app/static
    dataset_path = Path(__file__).resolve().parent / "static" / "dataset.csv"
    if dataset_path.exists():
        return dataset_path
    raise FileNotFoundError(f"No dataset.csv found at expected path: {dataset_path}")

def _to_bool_int(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
        .fillna(0)
        .astype(int)
    )

def load_data(file_path):
    data = pd.read_csv(file_path, encoding="utf-8")

    required_columns = FEATURE_COLUMNS + [PLACEMENT_COLUMN, REGRESSION_TARGET]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    x = data[FEATURE_COLUMNS + [PLACEMENT_COLUMN]].copy()

    for col in NUMERIC_COLUMNS:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        x[col] = x[col].fillna(x[col].median())

    for col in BOOLEAN_COLUMNS:
        x[col] = _to_bool_int(x[col])

    feature_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        x[col] = x[col].fillna("Unknown").astype(str)
        encoder = LabelEncoder()
        x[col] = encoder.fit_transform(x[col])
        feature_encoders[col] = encoder

    placement_encoder = LabelEncoder()
    x[PLACEMENT_COLUMN] = placement_encoder.fit_transform(x[PLACEMENT_COLUMN].astype(str))

    x_reg = x.copy()
    x_class = x.drop(columns=[PLACEMENT_COLUMN])

    y_classification = x[PLACEMENT_COLUMN].values
    y_regression = pd.to_numeric(data[REGRESSION_TARGET], errors="coerce")
    y_regression = y_regression.fillna(y_regression.median())

    return x_reg, x_class, y_classification, y_regression, feature_encoders, placement_encoder

def run_linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model

def run_random_forest(x, y):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
    )
    model.fit(x, y)
    return model

def _save_artifacts(output_dir, lr_model, rf_model, encoders, placement_encoder, x_reg_cols, x_class_cols):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(output_dir, "lr_regressor.pkl"))
    joblib.dump(rf_model, os.path.join(output_dir, "rf_classifier.pkl"))
    joblib.dump(encoders, os.path.join(output_dir, "feature_encoders.pkl"))
    joblib.dump(placement_encoder, os.path.join(output_dir, "placement_encoder.pkl"))

    rf_importance = pd.DataFrame(
        {
            "feature": x_class_cols,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    rf_importance.to_csv(os.path.join(output_dir, "rf_feature_importance.csv"), index=False)

    lr_coefficients = pd.DataFrame(
        {
            "feature": x_reg_cols,
            "coefficient": lr_model.coef_,
        }
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
    lr_coefficients.to_csv(os.path.join(output_dir, "lr_coefficients.csv"), index=False)

    metadata = {
        "regression_features": x_reg_cols,
        "classification_features": x_class_cols,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "placement_classes": placement_encoder.classes_.tolist(),
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def main():
    dataset_path = _resolve_dataset_path()
    x_reg, x_class, y_class, y_reg, encoders, placement_encoder = load_data(dataset_path)

    lr_model = run_linear_regression(x_reg, y_reg)
    rf_model = run_random_forest(x_class, y_class)

    app_dir = Path(__file__).resolve().parent
    output_dirs = [
        app_dir / "static" / "models",
    ]

    for output_dir in output_dirs:
        _save_artifacts(
            str(output_dir),
            lr_model,
            rf_model,
            encoders,
            placement_encoder,
            x_reg.columns.tolist(),
            x_class.columns.tolist(),
        )

    print(f"Models and explainability artifacts saved to: {', '.join(str(p) for p in output_dirs)}")

if __name__ == "__main__":
    main()