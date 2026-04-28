# PyTest Fixes Summary

This document outlines all the fixes applied to resolve the failing pytest errors.

## Fixed Issues

### 1. test_login_post_success_redirects_dashboard_expected_behavior
**Problem**: Expected 302 redirect, got 200
**Root Cause**: Form validation issues with the remember_me field

**Fix**: Simplified form data submission and ensured proper redirect behavior by:
- Removing the `remember_me` parameter from form data
- Using `follow_redirects=False` to ensure we get the 302 status code

### 2. test_upload_routes_expected_behavior  
**Problem**: POST request returned 200 instead of expected 302 redirect

**Root Cause**: Form validation failed due to:
- Integer values being passed for form fields that expect strings
- Missing field validation (sibling_group_size was 0, but form requires min=1)
- Missing submit button

**Fix**: 
- Converted all numeric values to strings in form data
- Set sibling_group_size to "1" (minimum required)
- Added "submit": "Upload Placement" to form data

### 3. test_predict_and_compare_routes_expected_behavior
**Problem**: AttributeError: 'object' object has no attribute 'cursor'

**Root Cause**: The mock connection object was too simple and didn't support the `cursor()` method needed by `get_prediction_numeric_averages()`

**Fix**: Enhanced the mock connection object in the fixture to:
```python
mock_connection_obj = types.SimpleNamespace()
mock_cursor = types.SimpleNamespace()
mock_cursor.execute = lambda *_args, **_kwargs: None
mock_cursor.fetchone = lambda: None
mock_cursor.fetchall = lambda: []
mock_connection_obj.cursor = lambda: mock_cursor
```

Also added mock for `get_prediction_numeric_averages` to return proper default values.

### 4. test_user_management_and_settings_routes_expected_behavior
**Problem**: POST to /users/70/edit returned 200 instead of 302

**Root Cause**: Form validation failed due to missing submit button in form data

**Fix**: Added `"submit": "Update User"` to the form data to ensure proper form submission

### 5. test_main_orchestrates_pipeline_expected_behavior
**Problem**: TypeError: fake_load() got an unexpected keyword argument 'column_mapping'

**Root Cause**: The test's mock `fake_load` function didn't accept keyword arguments, but the actual `load_data()` function is called with keyword parameters

**Fix**: Updated function signature from:
```python
def fake_load(_path):
```
to:
```python
def fake_load(_path, **kwargs):
```

### 6. test_plot_functions_create_output_files_expected_behavior
**Problem**: ValueError: All arrays must be of the same length

**Root Cause**: 
- Feature importance arrays didn't match the feature columns
- Metadata was saved as pickle instead of JSON

**Fix**:
- Properly constructed feature_cols list
- Created proper JSON metadata with "classification_features", "regression_features", and "breakdown_features"
- Changed from: `joblib.dump({"dummy": "metadata"}, models_dir / "model_metadata.json")`
- To: `(models_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")`

### 7. test_argument_parser_and_main_executes_expected_behavior
**Problem**: UnicodeDecodeError when loading metadata JSON

**Root Cause**: The test was saving model_metadata as a pickle file using `joblib.dump()` but the code expected JSON format

**Fix**: Changed metadata persistence to use JSON format with proper feature lists

## Changes Made to Files

### src/app/test_routes.py
1. Enhanced fixture to properly mock connection with cursor support
2. Fixed login test to use correct redirect validation
3. Fixed upload test with proper string form data
4. Fixed predict test with proper mocking
5. Fixed user management test with submit buttons

### src/app/test_train_models.py
1. Updated fake_load function to accept keyword arguments

### src/test_generate_model_visuals.py
1. Added json import
2. Fixed both plot tests to use JSON metadata instead of pickle
3. Added proper feature columns structure to metadata

## Testing Recommendation

Run all tests with:
```bash
pytest src/app/test_routes.py -xvs
pytest src/app/test_train_models.py -xvs
pytest src/test_generate_model_visuals.py -xvs
```

All 7 previously failing tests should now pass.

