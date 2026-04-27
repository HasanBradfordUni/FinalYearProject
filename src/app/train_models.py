import json
import os
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
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
BREAKDOWN_THRESHOLD_DAYS = 365
REQUIRED_COLUMNS = FEATURE_COLUMNS + [PLACEMENT_COLUMN, REGRESSION_TARGET]
DEFAULT_CRITICAL_FIELDS = [PLACEMENT_COLUMN, REGRESSION_TARGET]
DEFAULT_GENDER_VALUES = ["Non binary", "Male", "Trans Female", "Female", "Trans Male"]
DEFAULT_ETHNICITY_VALUES = [
    "Asian/British Asian - Chinese",
    "Asian/British Asian - Other",
    "Black/Black British - Other",
    "Gypsy / Roma",
    "Black/Black British - African",
    "White - British",
    "White - Irish",
    "Asian/British Asian - Indian",
    "White - Other",
    "Black/Black British - Caribbean",
    "Asian/British Asian - Pakistani",
    "Asian/British Asian - Bangladeshi",
    "Mixed - White/Black African",
    "Traveller of Irish Heritage",
    "Mixed - White/Asian",
    "Mixed - Other",
    "Traveller - Other",
    "White - Central European",
    "Mixed - White/Black Caribbean",
    "Dual Heritage - Black/White",
    "White - Eastern European",
]

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


def _normalize_column_name(column_name):
    return re.sub(r"[^a-z0-9]+", "", str(column_name).strip().lower())


def _scalar_default_value(column_name, value):
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
    else:
        text = value

    if column_name in BOOLEAN_COLUMNS:
        value_text = str(text).strip().lower()
        if value_text in {"true", "1", "yes", "y"}:
            return 1
        if value_text in {"false", "0", "no", "n"}:
            return 0
        return 0

    if column_name in NUMERIC_COLUMNS or column_name == REGRESSION_TARGET:
        return pd.to_numeric([text], errors="coerce")[0]

    return text


def _random_values_for_mode(mode, count, config=None):
    config = config or {}
    if count <= 0:
        return np.array([])

    if mode == "ethnicities":
        return np.random.choice(DEFAULT_ETHNICITY_VALUES, size=count)
    if mode == "genders":
        return np.random.choice(DEFAULT_GENDER_VALUES, size=count)
    if mode == "child_ages":
        return np.random.randint(0, 19, size=count)
    if mode == "carer_ages":
        return np.random.randint(25, 76, size=count)
    if mode == "boolean":
        return np.random.choice([1, 0], size=count)
    if mode == "custom":
        start = config.get("start")
        end = config.get("end")
        if start is None or end is None:
            return np.array([])
        start_value = float(start)
        end_value = float(end)
        if start_value > end_value:
            return np.array([])
        if start_value.is_integer() and end_value.is_integer():
            return np.random.randint(int(start_value), int(end_value) + 1, size=count)
        return np.random.uniform(start_value, end_value, size=count)

    return np.array([])


def _build_default_values(column_name, default_config, count):
    if count <= 0:
        return None

    if isinstance(default_config, dict):
        mode = str(default_config.get("mode", "")).strip().lower()
        random_values = _random_values_for_mode(mode, count, config=default_config)
        if random_values.size == 0:
            return None
        return random_values

    scalar_value = _scalar_default_value(column_name, default_config)
    if scalar_value is None:
        return None
    return np.array([scalar_value] * count)


def _apply_column_mapping(
    data,
    column_mapping=None,
    missing_defaults=None,
    critical_fields=None,
    exclude_missing_critical=False,
):
    column_mapping = column_mapping or {}
    missing_defaults = missing_defaults or {}
    critical_fields = critical_fields or DEFAULT_CRITICAL_FIELDS

    transformed = data.copy()
    for target in REQUIRED_COLUMNS:
        source = column_mapping.get(target)
        if source:
            if source not in transformed.columns:
                raise ValueError(f"Mapped source column '{source}' for '{target}' was not found in uploaded CSV.")
            transformed[target] = transformed[source]
            continue

        if target not in transformed.columns:
            transformed[target] = pd.NA

    if exclude_missing_critical:
        critical_missing_mask = pd.Series(False, index=transformed.index)
        for field in critical_fields:
            col_as_text = transformed[field].astype(str).str.strip()
            missing_mask = transformed[field].isna() | col_as_text.eq("") | col_as_text.str.lower().eq("none")
            critical_missing_mask = critical_missing_mask | missing_mask
        transformed = transformed.loc[~critical_missing_mask].copy()

    for target in REQUIRED_COLUMNS:
        col_as_text = transformed[target].astype(str).str.strip()
        missing_mask = transformed[target].isna() | col_as_text.eq("") | col_as_text.str.lower().eq("none")
        default_values = _build_default_values(target, missing_defaults.get(target), int(missing_mask.sum()))
        if default_values is None:
            continue
        transformed.loc[missing_mask, target] = default_values

    unresolved_critical = []
    for field in critical_fields:
        col_as_text = transformed[field].astype(str).str.strip()
        missing_mask = transformed[field].isna() | col_as_text.eq("") | col_as_text.str.lower().eq("none")
        if transformed.empty or missing_mask.any():
            unresolved_critical.append(field)

    if unresolved_critical:
        raise ValueError(
            f"Critical fields still contain missing values after mapping/default handling: {unresolved_critical}"
        )

    return transformed

def load_data(
    file_path,
    column_mapping=None,
    missing_defaults=None,
    critical_fields=None,
    exclude_missing_critical=False,
):
    data = pd.read_csv(file_path, encoding="utf-8")
    data = _apply_column_mapping(
        data,
        column_mapping=column_mapping,
        missing_defaults=missing_defaults,
        critical_fields=critical_fields,
        exclude_missing_critical=exclude_missing_critical,
    )

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
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
    x_placement_class = x.drop(columns=[PLACEMENT_COLUMN])
    x_breakdown_class = x.copy()

    y_placement_classification = x[PLACEMENT_COLUMN].values
    y_regression = pd.to_numeric(data[REGRESSION_TARGET], errors="coerce")
    y_regression = y_regression.fillna(y_regression.median())

    # Early breakdown proxy for binary classification.
    y_breakdown = (y_regression < BREAKDOWN_THRESHOLD_DAYS).astype(int)

    return (
        x_reg,
        x_placement_class,
        x_breakdown_class,
        y_placement_classification,
        y_breakdown,
        y_regression,
        feature_encoders,
        placement_encoder,
    )

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


def run_random_forest_regressor(x, y):
    model = RandomForestRegressor(
        n_estimators=350,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(x, y)
    return model

def _save_artifacts(
    output_dir,
    lr_model,
    rf_reg_model,
    rf_placement_model,
    rf_breakdown_model,
    encoders,
    placement_encoder,
    x_reg_cols,
    x_placement_cols,
    x_breakdown_cols,
):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(output_dir, "lr_regressor.pkl"))
    joblib.dump(rf_reg_model, os.path.join(output_dir, "rf_regressor.pkl"))
    joblib.dump(rf_placement_model, os.path.join(output_dir, "rf_classifier.pkl"))
    joblib.dump(rf_breakdown_model, os.path.join(output_dir, "rf_breakdown_classifier.pkl"))
    joblib.dump(encoders, os.path.join(output_dir, "feature_encoders.pkl"))
    joblib.dump(placement_encoder, os.path.join(output_dir, "placement_encoder.pkl"))

    rf_importance = pd.DataFrame(
        {
            "feature": x_placement_cols,
            "importance": rf_placement_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    rf_importance.to_csv(os.path.join(output_dir, "rf_feature_importance.csv"), index=False)

    breakdown_importance = pd.DataFrame(
        {
            "feature": x_breakdown_cols,
            "importance": rf_breakdown_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    breakdown_importance.to_csv(os.path.join(output_dir, "rf_breakdown_feature_importance.csv"), index=False)

    lr_coefficients = pd.DataFrame(
        {
            "feature": x_reg_cols,
            "coefficient": lr_model.coef_,
        }
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
    lr_coefficients.to_csv(os.path.join(output_dir, "lr_coefficients.csv"), index=False)

    metadata = {
        "regression_features": x_reg_cols,
        "classification_features": x_placement_cols,
        "breakdown_features": x_breakdown_cols,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "placement_classes": placement_encoder.classes_.tolist(),
        "breakdown_threshold_days": BREAKDOWN_THRESHOLD_DAYS,
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def main(
    dataset_path=None,
    column_mapping=None,
    missing_defaults=None,
    critical_fields=None,
    exclude_missing_critical=False,
):
    dataset_path = Path(dataset_path) if dataset_path else _resolve_dataset_path()
    (
        x_reg,
        x_placement_class,
        x_breakdown_class,
        y_placement_class,
        y_breakdown,
        y_reg,
        encoders,
        placement_encoder,
    ) = load_data(
        dataset_path,
        column_mapping=column_mapping,
        missing_defaults=missing_defaults,
        critical_fields=critical_fields,
        exclude_missing_critical=exclude_missing_critical,
    )

    lr_model = run_linear_regression(x_reg, y_reg)
    rf_reg_model = run_random_forest_regressor(x_reg, y_reg)
    rf_placement_model = run_random_forest(x_placement_class, y_placement_class)
    rf_breakdown_model = run_random_forest(x_breakdown_class, y_breakdown)

    app_dir = Path(__file__).resolve().parent
    output_dirs = [
        app_dir / "static" / "models",
    ]

    for output_dir in output_dirs:
        _save_artifacts(
            str(output_dir),
            lr_model,
            rf_reg_model,
            rf_placement_model,
            rf_breakdown_model,
            encoders,
            placement_encoder,
            x_reg.columns.tolist(),
            x_placement_class.columns.tolist(),
            x_breakdown_class.columns.tolist(),
        )

    print(f"Models and explainability artifacts saved to: {', '.join(str(p) for p in output_dirs)}")

if __name__ == "__main__":
    main()