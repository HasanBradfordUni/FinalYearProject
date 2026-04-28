from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from src import generate_model_visuals as gmv


def _sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Child Age At Placement": [10, 12, 14, 15, 11, 13],
            "Child Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "Child Ethnicity": ["White - British", "White - Irish", "White - British", "White - Irish", "White - British", "White - Irish"],
            "Child Prior Placements": [0, 1, 2, 1, 0, 2],
            "Returning Child": ["False", "True", "False", "True", "False", "True"],
            "Missing Episodes": [0, 1, 0, 2, 1, 0],
            "Sibling Group Size": [0, 1, 2, 1, 0, 2],
            "Placed With Siblings": ["False", "True", "True", "False", "False", "True"],
            "Carer Age": [35, 45, 40, 50, 37, 47],
            "Carer Gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "Carer Ethnicity": ["White - British", "White - Irish", "White - British", "White - Irish", "White - British", "White - Irish"],
            "EH involvement": ["False", "True", "False", "True", "False", "True"],
            "YOT involvement": ["False", "False", "True", "True", "False", "True"],
            "Placement Sequence Number": [1, 2, 3, 2, 1, 3],
            "Placement Type": ["Kinship", "External Fostering", "In-House Fostering", "Kinship", "External Fostering", "In-House Fostering"],
            "Days Placed": [50, 120, 340, 800, 1500, 200],
        }
    )


def test_find_first_existing_returns_first_match_expected_behavior(tmp_path):
    missing = tmp_path / "missing.pkl"
    present = tmp_path / "present.pkl"
    present.write_text("x", encoding="utf-8")
    assert gmv._find_first_existing([missing, present]) == present


def test_load_json_missing_file_returns_empty_dict_expected_behavior(tmp_path):
    assert gmv._load_json(tmp_path / "none.json") == {}


def test_normalize_boolean_like_maps_values_expected_behavior():
    s = pd.Series(["True", "false", "yes", "0", "unknown"])
    out = gmv._normalize_boolean_like(s)
    assert list(out[:4]) == [1, 0, 1, 0]
    assert np.isnan(out.iloc[4])


def test_infer_feature_columns_excludes_targets_and_ids_expected_behavior():
    df = pd.DataFrame({"id": [1], "feature": [2], "Days Placed": [3]})
    cols = gmv._infer_feature_columns(df, {"Days Placed"})
    assert cols == ["feature"]


def test_encode_with_known_classes_unknown_maps_minus_one_expected_behavior():
    encoded = gmv._encode_with_known_classes(pd.Series(["A", "X"]), np.array(["A", "B"], dtype=object))
    assert list(encoded) == [0, -1]


def test_prepare_features_encodes_mixed_types_expected_behavior():
    df = pd.DataFrame({"num": [1, None], "flag": ["True", "False"], "cat": ["x", "y"]})
    prepared = gmv._prepare_features(df, ["num", "flag", "cat"])
    assert prepared.shape == (2, 3)
    assert set(prepared["flag"].tolist()) <= {0, 1}


def test_plot_functions_create_output_files_expected_behavior(tmp_path):
    dataset = _sample_dataset()
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    feature_cols = [
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
        "Placement Type",
    ]
    
    x = gmv._prepare_features(dataset, feature_cols)
    y_reg = dataset["Days Placed"].to_numpy()

    y_cls_encoder = LabelEncoder()
    y_cls = y_cls_encoder.fit_transform(dataset["Placement Type"].astype(str))

    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=1).fit(x, y_cls)
    rf_regressor = RandomForestRegressor(n_estimators=10, random_state=1).fit(x, y_reg)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(rf_classifier, models_dir / "rf_classifier.pkl")
    joblib.dump(rf_regressor, models_dir / "rf_regressor.pkl")
    joblib.dump(rf_classifier, models_dir / "rf_model.pkl")
    
    # Save metadata as JSON with correct format
    metadata = {
        "classification_features": feature_cols,
        "regression_features": feature_cols,
        "breakdown_features": feature_cols
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    output_dir = tmp_path / "visuals"

    fi_path = gmv.plot_feature_importance(dataset_path, models_dir / "rf_model.pkl", output_dir)
    cm_path = gmv.plot_confusion_matrix(dataset_path, models_dir / "rf_classifier.pkl", output_dir)
    pv_path = gmv.plot_predicted_vs_actual(dataset_path, models_dir / "rf_regressor.pkl", output_dir)

    assert fi_path.exists()
    assert cm_path.exists()
    assert pv_path.exists()


def test_plot_confusion_matrix_raises_for_missing_target_expected_behavior(tmp_path):
    dataset = _sample_dataset().drop(columns=["Placement Type"])
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    x = dataset.drop(columns=["Days Placed"]).select_dtypes(include=["number"]).fillna(0)
    y = np.array([0] * len(dataset))
    clf = RandomForestClassifier(n_estimators=5, random_state=1).fit(x, y)
    model_path = tmp_path / "rf_classifier.pkl"
    joblib.dump(clf, model_path)

    try:
        gmv.plot_confusion_matrix(dataset_path, model_path, tmp_path)
    except KeyError:
        assert True
    else:
        assert False, "Expected KeyError for missing classification target"


def test_argument_parser_and_main_executes_expected_behavior(tmp_path, monkeypatch):
    dataset = _sample_dataset()
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    feature_cols = gmv._infer_feature_columns(dataset, {"Days Placed", "Placement Type"}) + ["Placement Type"]
    x = gmv._prepare_features(dataset, feature_cols)
    y_cls = LabelEncoder().fit_transform(dataset["Placement Type"])
    y_reg = dataset["Days Placed"]
    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=2).fit(x.drop(columns=["Placement Type"]), y_cls)
    rf_regressor = RandomForestRegressor(n_estimators=10, random_state=2).fit(x, y_reg)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(rf_classifier, models_dir / "rf_classifier.pkl")
    joblib.dump(rf_regressor, models_dir / "rf_regressor.pkl")
    joblib.dump(rf_classifier, models_dir / "rf_model.pkl")
    # Save metadata as JSON instead of pickle
    metadata = {
        "classification_features": feature_cols,
        "regression_features": feature_cols,
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_model_visuals.py",
            "--dataset",
            str(dataset_path),
            "--models-dir",
            str(models_dir),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    gmv.main()

    assert (tmp_path / "out" / "feature_importance.png").exists()
