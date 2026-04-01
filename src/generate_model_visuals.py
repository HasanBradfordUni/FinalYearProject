from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# BCFT palette
BCFT_PRIMARY = "#07375f"
BCFT_SECONDARY = "#ce0f69"
BCFT_TERTIARY_YELLOW = "#ffdd00"
BCFT_TERTIARY_BLUE = "#a2c7e2"

# Defaults for this project structure (inside src/)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "app" / "static" / "dataset.csv"
DEFAULT_MODELS_DIR = BASE_DIR / "app" / "static" / "models"
DEFAULT_OUTPUT_DIR = BASE_DIR / "app" / "static" / "visuals"
DEFAULT_METADATA_PATH = DEFAULT_MODELS_DIR / "model_metadata.json"

# Change these if your target column names differ.
REGRESSION_TARGET = "Days Placed"
# You can rename this to "StabilityClass" when your dataset has that label.
CLASSIFICATION_TARGET = "Placement Type"

ID_LIKE_COLUMNS = {
    "id",
    "record_id",
    "child_id",
    "placement_id",
}


def _find_first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    checked = "\n".join(str(p) for p in paths)
    raise FileNotFoundError(f"None of these files exist:\n{checked}")


def _load_json(path: Path) -> dict:
    import json

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _normalize_boolean_like(series: pd.Series) -> pd.Series:
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
            }
        )
    )
    return mapped


def _infer_feature_columns(data: pd.DataFrame, targets: set[str]) -> list[str]:
    candidates: list[str] = []
    for column in data.columns:
        normalized_name = column.strip().lower().replace(" ", "_")
        is_id_like = normalized_name in ID_LIKE_COLUMNS or normalized_name.endswith("_id")
        if column not in targets and not is_id_like:
            candidates.append(column)
    return candidates


def _encode_with_known_classes(series: pd.Series, classes: np.ndarray) -> pd.Series:
    class_to_index = {label: index for index, label in enumerate(classes)}
    # Unknown values map to -1 so the model still receives numeric input.
    return series.map(class_to_index).fillna(-1).astype(int)


def _prepare_features(
    data: pd.DataFrame,
    feature_columns: list[str],
    feature_encoders: dict | None = None,
) -> pd.DataFrame:
    prepared = data[feature_columns].copy()

    for column in feature_columns:
        series = prepared[column]

        if pd.api.types.is_numeric_dtype(series):
            prepared[column] = pd.to_numeric(series, errors="coerce").fillna(series.median())
            continue

        boolean_mapped = _normalize_boolean_like(series)
        if boolean_mapped.notna().sum() >= max(2, int(len(series) * 0.6)):
            prepared[column] = boolean_mapped.fillna(0).astype(int)
            continue

        if feature_encoders and column in feature_encoders:
            known_classes = np.array(feature_encoders[column].classes_, dtype=object)
            prepared[column] = _encode_with_known_classes(series.fillna("Unknown").astype(str), known_classes)
        else:
            encoder = LabelEncoder()
            prepared[column] = encoder.fit_transform(series.fillna("Unknown").astype(str))

    return prepared


def plot_feature_importance(
    dataset_path: Path,
    rf_model_path: Path,
    output_dir: Path,
    target_column: str = REGRESSION_TARGET,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    feature_encoders_path: Path | None = None,
) -> Path:
    """Generate and save a horizontal feature-importance bar chart from a Random Forest model."""
    data = pd.read_csv(dataset_path)
    model = joblib.load(rf_model_path)
    metadata = _load_json(metadata_path)

    feature_columns = metadata.get("classification_features") or metadata.get("regression_features")
    if not feature_columns:
        feature_columns = _infer_feature_columns(data, targets={target_column, CLASSIFICATION_TARGET})

    feature_encoders = None
    if feature_encoders_path and feature_encoders_path.exists():
        feature_encoders = joblib.load(feature_encoders_path)

    _prepare_features(data, feature_columns, feature_encoders=feature_encoders)

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model at {rf_model_path} does not expose feature_importances_. "
            "Use a Random Forest model for this chart."
        )

    importances = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_importance.png"

    plt.figure(figsize=(10, max(6, len(importances) * 0.4)))
    colors = [BCFT_TERTIARY_BLUE] * len(importances)
    if len(colors) > 0:
        colors[-1] = BCFT_PRIMARY
    plt.barh(importances["Feature"], importances["Importance"], color=colors, edgecolor=BCFT_PRIMARY)
    plt.xlabel("Feature Importance", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.title("Random Forest Feature Importance", fontsize=13, color=BCFT_PRIMARY)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_confusion_matrix(
    dataset_path: Path,
    classifier_model_path: Path,
    output_dir: Path,
    class_target_column: str = CLASSIFICATION_TARGET,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    feature_encoders_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """Generate and save a confusion-matrix heatmap for a classifier model."""
    data = pd.read_csv(dataset_path)
    if class_target_column not in data.columns:
        raise KeyError(
            f"Target column '{class_target_column}' not found in dataset. "
            "Rename class_target_column to match your dataset (e.g., StabilityClass)."
        )

    model = joblib.load(classifier_model_path)
    metadata = _load_json(metadata_path)

    feature_columns = metadata.get("classification_features")
    if not feature_columns:
        feature_columns = _infer_feature_columns(data, targets={class_target_column, REGRESSION_TARGET})

    feature_encoders = None
    if feature_encoders_path and feature_encoders_path.exists():
        feature_encoders = joblib.load(feature_encoders_path)

    x = _prepare_features(data, feature_columns, feature_encoders=feature_encoders)
    y_raw = data[class_target_column].fillna("Unknown").astype(str)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y_raw)
    class_labels = y_encoder.classes_

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # Fit on train split to evaluate in a standard train/test workflow.
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    cm = confusion_matrix(y_test, predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "confusion_matrix.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(BCFT_PRIMARY, as_cmap=True),
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Predicted label", fontsize=11)
    plt.ylabel("True label", fontsize=11)
    plt.title("Confusion Matrix", fontsize=13, color=BCFT_PRIMARY)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_predicted_vs_actual(
    dataset_path: Path,
    regression_model_path: Path,
    output_dir: Path,
    regression_target_column: str = REGRESSION_TARGET,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    feature_encoders_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """Generate and save a predicted-vs-actual scatter chart for a regression model."""
    data = pd.read_csv(dataset_path)
    if regression_target_column not in data.columns:
        raise KeyError(
            f"Target column '{regression_target_column}' not found in dataset. "
            "Rename regression_target_column to match your dataset."
        )

    model = joblib.load(regression_model_path)
    metadata = _load_json(metadata_path)

    feature_columns = metadata.get("regression_features")
    if not feature_columns:
        feature_columns = _infer_feature_columns(data, targets={regression_target_column, CLASSIFICATION_TARGET})

    feature_encoders = None
    if feature_encoders_path and feature_encoders_path.exists():
        feature_encoders = joblib.load(feature_encoders_path)

    x = _prepare_features(data, feature_columns, feature_encoders=feature_encoders)
    y = pd.to_numeric(data[regression_target_column], errors="coerce")
    y = y.fillna(y.median())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Refit so the chart reflects the configured split on this dataset.
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predicted_vs_actual.png"

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.75, color=BCFT_SECONDARY, edgecolor=BCFT_PRIMARY)

    min_value = min(np.min(y_test), np.min(predictions))
    max_value = max(np.max(y_test), np.max(predictions))
    plt.plot([min_value, max_value], [min_value, max_value], color=BCFT_PRIMARY, linewidth=2)

    plt.xlabel("Actual Days Placed", fontsize=11)
    plt.ylabel("Predicted Days Placed", fontsize=11)
    plt.title("Predicted vs Actual (Regression)", fontsize=13, color=BCFT_PRIMARY)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate model-centric visuals for placement prediction models.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to dataset CSV.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory containing model artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save generated visuals.")
    parser.add_argument("--regression-target", default=REGRESSION_TARGET, help="Regression target column name.")
    parser.add_argument("--classification-target", default=CLASSIFICATION_TARGET, help="Classification target column name.")
    parser.add_argument(
        "--rf-importance-model",
        type=Path,
        default=None,
        help="Random Forest model path for feature importance (defaults to first existing rf_model.pkl/rf_classifier.pkl).",
    )
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=None,
        help="Classifier model path for confusion matrix (defaults to rf_classifier.pkl).",
    )
    parser.add_argument(
        "--regressor-model",
        type=Path,
        default=None,
        help="Regressor model path for predicted-vs-actual (defaults to rf_regressor.pkl/lr_model.pkl/lr_regressor.pkl).",
    )
    parser.add_argument(
        "--feature-encoders",
        type=Path,
        default=None,
        help="Path to feature_encoders.pkl (defaults to models-dir/feature_encoders.pkl).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to model metadata JSON (defaults to models-dir/model_metadata.json).",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()

    metadata_path = args.metadata if args.metadata else args.models_dir / "model_metadata.json"
    feature_encoders_path = args.feature_encoders if args.feature_encoders else args.models_dir / "feature_encoders.pkl"

    rf_importance_model_path = args.rf_importance_model or _find_first_existing(
        [
            args.models_dir / "rf_model.pkl",
            args.models_dir / "rf_classifier.pkl",
        ]
    )
    classifier_model_path = args.classifier_model or _find_first_existing([args.models_dir / "rf_classifier.pkl"])
    regressor_model_path = args.regressor_model or _find_first_existing(
        [
            args.models_dir / "rf_regressor.pkl",
            args.models_dir / "lr_model.pkl",
            args.models_dir / "lr_regressor.pkl",
        ]
    )

    feature_importance_path = plot_feature_importance(
        dataset_path=args.dataset,
        rf_model_path=rf_importance_model_path,
        output_dir=args.output_dir,
        target_column=args.regression_target,
        metadata_path=metadata_path,
        feature_encoders_path=feature_encoders_path,
    )

    confusion_matrix_path = plot_confusion_matrix(
        dataset_path=args.dataset,
        classifier_model_path=classifier_model_path,
        output_dir=args.output_dir,
        class_target_column=args.classification_target,
        metadata_path=metadata_path,
        feature_encoders_path=feature_encoders_path,
    )

    predicted_vs_actual_path = plot_predicted_vs_actual(
        dataset_path=args.dataset,
        regression_model_path=regressor_model_path,
        output_dir=args.output_dir,
        regression_target_column=args.regression_target,
        metadata_path=metadata_path,
        feature_encoders_path=feature_encoders_path,
    )

    print("Generated visuals:")
    print(f"- {feature_importance_path}")
    print(f"- {confusion_matrix_path}")
    print(f"- {predicted_vs_actual_path}")


if __name__ == "__main__":
    main()
