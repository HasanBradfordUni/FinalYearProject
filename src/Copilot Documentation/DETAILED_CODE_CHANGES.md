# Detailed Code Changes Reference

## File: src/app/test_routes.py

### Change 1: Enhanced app_client fixture (lines 15-30)
Added proper mocking of database connection with cursor support:

```python
# Before:
monkeypatch.setattr(routes, "connection", object())

# After:
mock_connection_obj = types.SimpleNamespace()
mock_cursor = types.SimpleNamespace()
mock_cursor.execute = lambda *_args, **_kwargs: None
mock_cursor.fetchone = lambda: None
mock_cursor.fetchall = lambda: []
mock_connection_obj.cursor = lambda: mock_cursor
monkeypatch.setattr(routes, "connection", mock_connection_obj)
```

### Change 2: test_login_post_success_redirects_dashboard_expected_behavior (lines 81-90)
Simplified form submission:

```python
# Before:
response = client.post("/login", data={"username": "ok", "password": "pass123456", "remember_me": "y"})

# After:
response = client.post("/login", data={"username": "ok", "password": "pass123456"}, follow_redirects=False)
```

### Change 3: test_upload_routes_expected_behavior (lines 174-215)
Fixed form data types and added required fields:

```python
# Before:
data={
    "child_age": 10,  # Integer
    "child_prior_placements": 1,  # Integer
    "missing_episodes": 0,  # Integer
    "sibling_group_size": 0,  # Invalid: min=1
    "carer_age": 40,  # Integer
    # Missing submit button
}

# After:
data={
    "child_age": "10",  # String
    "child_prior_placements": "1",  # String
    "missing_episodes": "0",  # String
    "sibling_group_size": "1",  # String, min=1
    "carer_age": "40",  # String
    "submit": "Upload Placement",  # Added submit button
}
```

### Change 4: test_predict_and_compare_routes_expected_behavior (lines 236-265)
Added mock for database function and fixed form data:

```python
# Added this mock:
monkeypatch.setattr(routes, "get_prediction_numeric_averages", 
    lambda *_args, **_kwargs: {
        "child_age": 10, 
        "child_prior_placements": 1, 
        "missing_episodes": 0, 
        "sibling_group_size": 1, 
        "carer_age": 40
    })

# Before:
predict_post = client.post("/predict", data={"child_age": 10})

# After:
predict_post = client.post("/predict", data={"child_age": "10"})
```

### Change 5: test_user_management_and_settings_routes_expected_behavior (lines 316-338)
Added submit buttons to form submissions:

```python
# Before:
client.post("/users/add", data={
    "username": "newuser",
    "email": "new@example.com",
    "password": "Password123",
    "confirm_password": "Password123",
    "role": "placement_officer"
    # Missing submit
})

# After:
client.post("/users/add", data={
    "username": "newuser",
    "email": "new@example.com",
    "password": "Password123",
    "confirm_password": "Password123",
    "role": "placement_officer",
    "submit": "Create User"  # Added
})

# Similar changes for edit and other forms
```

---

## File: src/app/test_train_models.py

### Change 1: test_main_orchestrates_pipeline_expected_behavior (lines 121-139)
Updated fake_load to accept keyword arguments:

```python
# Before:
def fake_load(_path):
    calls["load"] = True
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", tm.PLACEMENT_COLUMN])
    return df, df[["a"]], df, [0, 1], [1, 0], [10, 20], {}, type("P", (), {"classes_": ["Kinship"]})()

# After:
def fake_load(_path, **kwargs):  # Added **kwargs
    calls["load"] = True
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", tm.PLACEMENT_COLUMN])
    return df, df[["a"]], df, [0, 1], [1, 0], [10, 20], {}, type("P", (), {"classes_": ["Kinship"]})()
```

---

## File: src/test_generate_model_visuals.py

### Change 1: Added json import (line 2)
```python
# Before:
from pathlib import Path

import joblib

# After:
from pathlib import Path
import json

import joblib
```

### Change 2: test_plot_functions_create_output_files_expected_behavior (lines 71-118)
Fixed feature columns and metadata JSON:

```python
# Before:
x = gmv._prepare_features(dataset, [
    "Child Age At Placement",
    "Child Gender",
    # ... many columns
    "Placement Type",
])
# ...
joblib.dump(rf_classifier, models_dir / "rf_model.pkl")
# Missing metadata file

# After:
feature_cols = [
    "Child Age At Placement",
    "Child Gender",
    # ... many columns
    "Placement Type",
]
x = gmv._prepare_features(dataset, feature_cols)
# ...
joblib.dump(rf_classifier, models_dir / "rf_model.pkl")

# Save metadata as JSON with correct format
metadata = {
    "classification_features": feature_cols,
    "regression_features": feature_cols,
    "breakdown_features": feature_cols
}
(models_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
```

### Change 3: test_argument_parser_and_main_executes_expected_behavior (lines 140-173)
Fixed metadata persistence to use JSON:

```python
# Before:
joblib.dump(rf_classifier, models_dir / "rf_model.pkl")
joblib.dump({"dummy": "metadata"}, models_dir / "model_metadata.json")  # Wrong format

# After:
joblib.dump(rf_classifier, models_dir / "rf_model.pkl")
metadata = {
    "classification_features": feature_cols,
    "regression_features": feature_cols,
}
(models_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
```

---

## Summary of Issue Categories

1. **Form Validation**: Tests were passing wrong data types (integers instead of strings) and missing submit buttons
2. **Mock Objects**: Connection mock needed proper cursor method support
3. **Function Signatures**: Mock functions needed to accept keyword arguments
4. **File Format**: Metadata should be JSON, not pickle, for consistency
5. **Data Validation**: Form data needed to respect field constraints (e.g., sibling_group_size min=1)

