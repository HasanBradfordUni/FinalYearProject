import json
from pathlib import Path

import pandas as pd

from src.app import train_models as tm


def _dataset_frame():
    return pd.DataFrame(
        {
            "Child Age At Placement": [10, 12, 14, 11, 13],
            "Child Gender": ["Male", "Female", "Male", "Female", "Male"],
            "Child Ethnicity": ["White - British", "White - Irish", "White - British", "White - Irish", "White - British"],
            "Child Prior Placements": [0, 1, 2, 1, 0],
            "Returning Child": ["False", "True", "False", "True", "False"],
            "Missing Episodes": [0, 1, 0, 2, 1],
            "Sibling Group Size": [0, 1, 2, 1, 0],
            "Placed With Siblings": ["False", "True", "True", "False", "False"],
            "Carer Age": [35, 45, 40, 50, 37],
            "Carer Gender": ["Female", "Male", "Female", "Male", "Female"],
            "Carer Ethnicity": ["White - British", "White - Irish", "White - British", "White - Irish", "White - British"],
            "EH involvement": ["False", "True", "False", "True", "False"],
            "YOT involvement": ["False", "False", "True", "True", "False"],
            "Placement Sequence Number": [1, 2, 3, 2, 1],
            "Placement Type": ["Kinship", "External Fostering", "In-House Fostering", "Kinship", "External Fostering"],
            "Days Placed": [50, 120, 340, 800, 200],
        }
    )


def test_resolve_dataset_path_returns_existing_file_expected_behavior(tmp_path, monkeypatch):
    fake_script = tmp_path / "train_models.py"
    fake_script.write_text("", encoding="utf-8")
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "dataset.csv").write_text("x", encoding="utf-8")
    monkeypatch.setattr(tm, "__file__", str(fake_script))
    assert tm._resolve_dataset_path() == static_dir / "dataset.csv"


def test_to_bool_int_maps_values_expected_behavior():
    out = tm._to_bool_int(pd.Series(["True", "false", "yes", "0"]))
    assert list(out) == [1, 0, 1, 0]


def test_load_data_returns_expected_structures_expected_behavior(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    _dataset_frame().to_csv(dataset_path, index=False)
    result = tm.load_data(dataset_path)
    assert len(result) == 8
    x_reg, x_place, x_break, y_place, y_break, y_reg, encoders, placement_encoder = result
    assert x_reg.shape[0] == 5
    assert x_place.shape[1] == len(tm.FEATURE_COLUMNS)
    assert len(y_place) == 5
    assert set(y_break.tolist()) <= {0, 1}
    assert len(encoders) == 4
    assert hasattr(placement_encoder, "classes_")


def test_load_data_missing_columns_raises_expected_behavior(tmp_path):
    bad = pd.DataFrame({"foo": [1]})
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    try:
        tm.load_data(path)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError when required columns are missing"


def test_model_fit_helpers_return_trained_models_expected_behavior():
    data = _dataset_frame()
    x = pd.get_dummies(data[tm.FEATURE_COLUMNS + [tm.PLACEMENT_COLUMN]], drop_first=False)
    y_reg = data[tm.REGRESSION_TARGET]
    y_cls = pd.factorize(data[tm.PLACEMENT_COLUMN])[0]

    lr = tm.run_linear_regression(x, y_reg)
    rf_cls = tm.run_random_forest(x, y_cls)
    rf_reg = tm.run_random_forest_regressor(x, y_reg)

    assert hasattr(lr, "coef_")
    assert hasattr(rf_cls, "feature_importances_")
    assert hasattr(rf_reg, "feature_importances_")


def test_save_artifacts_writes_expected_files_expected_behavior(tmp_path):
    data = _dataset_frame()
    x = pd.get_dummies(data[tm.FEATURE_COLUMNS + [tm.PLACEMENT_COLUMN]], drop_first=False)
    y_reg = data[tm.REGRESSION_TARGET]
    y_cls = pd.factorize(data[tm.PLACEMENT_COLUMN])[0]

    lr = tm.run_linear_regression(x, y_reg)
    rf_reg = tm.run_random_forest_regressor(x, y_reg)
    rf_cls = tm.run_random_forest(x.drop(columns=[c for c in x.columns if c.startswith("Placement Type_")][:1], errors="ignore"), y_cls)
    rf_break = tm.run_random_forest(x, (y_reg < tm.BREAKDOWN_THRESHOLD_DAYS).astype(int))

    from sklearn.preprocessing import LabelEncoder

    placement_encoder = LabelEncoder().fit(data[tm.PLACEMENT_COLUMN])
    tm._save_artifacts(
        str(tmp_path),
        lr,
        rf_reg,
        rf_cls,
        rf_break,
        {},
        placement_encoder,
        x.columns.tolist(),
        x.drop(columns=[c for c in x.columns if c.startswith("Placement Type_")][:1], errors="ignore").columns.tolist(),
        x.columns.tolist(),
    )

    assert (tmp_path / "lr_regressor.pkl").exists()
    assert (tmp_path / "model_metadata.json").exists()
    metadata = json.loads((tmp_path / "model_metadata.json").read_text(encoding="utf-8"))
    assert "breakdown_threshold_days" in metadata


def test_main_orchestrates_pipeline_expected_behavior(monkeypatch):
    calls = {"resolve": False, "load": False, "save": 0}

    monkeypatch.setattr(tm, "_resolve_dataset_path", lambda: Path("dummy.csv"))

    def fake_load(_path):
        calls["load"] = True
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", tm.PLACEMENT_COLUMN])
        return df, df[["a"]], df, [0, 1], [1, 0], [10, 20], {}, type("P", (), {"classes_": ["Kinship"]})()

    monkeypatch.setattr(tm, "load_data", fake_load)
    monkeypatch.setattr(tm, "run_linear_regression", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(tm, "run_random_forest_regressor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(tm, "run_random_forest", lambda *_args, **_kwargs: type("RF", (), {"feature_importances_": [1.0]})())
    monkeypatch.setattr(tm, "_save_artifacts", lambda *_args, **_kwargs: calls.__setitem__("save", calls["save"] + 1))

    tm.main()
    assert calls["load"] is True
    assert calls["save"] == 1
