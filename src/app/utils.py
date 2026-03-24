import pandas as pd
import csv
from io import StringIO

REGRESSION_FEATURE_COLUMNS = [
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

CATEGORICAL_FEATURES = {"Child Gender", "Child Ethnicity", "Carer Gender", "Carer Ethnicity"}

# ============== Prediction Utility Functions ==============

def prepare_prediction_input(form, feature_encoders):
    """
    Prepares user input from form so it matches the exact feature structure
    used during model training.
    """
    def get_value(key, default):
        val = getattr(form, key, None)
        if val and hasattr(val, 'data'):
            val = val.data
        return val if val not in [None, "", "None"] else default

    def parse_bool(value, default=False):
        if value is None:
            return int(default)
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip().lower() in {"true", "1", "yes", "y"})

    def encode_value(encoder, raw_value):
        text_value = str(raw_value)
        classes = set(encoder.classes_.astype(str))
        if text_value in classes:
            return int(encoder.transform([text_value])[0])
        fallback = "Unknown" if "Unknown" in classes else str(encoder.classes_[0])
        return int(encoder.transform([fallback])[0])

    child_prior_placements = int(get_value("child_prior_placements", 0))

    # Setup a dataframe with entered/default values
    df = pd.DataFrame([
        {
            "Child Age At Placement": int(get_value("child_age", 10)),
            "Child Gender": get_value("child_gender", "Unknown"),
            "Child Ethnicity": get_value("child_ethnicity", "Unknown"),
            "Child Prior Placements": child_prior_placements,
            "Returning Child": parse_bool(get_value("returning_child", "False")),
            "Missing Episodes": int(get_value("missing_episodes", 0)),
            "Sibling Group Size": int(get_value("sibling_group_size", 0)),
            "Placed With Siblings": parse_bool(get_value("placed_with_siblings", "False")),
            "Carer Age": int(get_value("carer_age", 45)),
            "Carer Gender": get_value("carer_gender", "Unknown"),
            "Carer Ethnicity": get_value("carer_ethnicity", "Unknown"),
            "EH involvement": parse_bool(get_value("eh_involvement", "False")),
            "YOT involvement": parse_bool(get_value("yot_involvement", "False")),
            "Placement Sequence Number": child_prior_placements + 1,
            "Placement Type": 0,
        }
    ])

    # Apply encoders to known categorical columns only.
    for col, encoder in feature_encoders.items():
        if col in CATEGORICAL_FEATURES and col in df.columns:
            df[col] = df[col].apply(lambda value: encode_value(encoder, value))

    return df[REGRESSION_FEATURE_COLUMNS].values

def generate_predictions_list(input_data, rf_model, lr_model, placement_encoder):
    """
    Generate predictions for all placement types.
    Returns a list of dictionaries with type, duration, and stability score.
    """
    predictions = []

    if rf_model is None or lr_model is None or placement_encoder is None:
        return [{"type": "Error", "duration": 0, "stability": 0, "message": "Models not loaded"}]

    # Get all placement types
    placement_types = placement_encoder.classes_

    for placement_type in placement_types:
        # Encode the placement type
        encoded_placement = placement_encoder.transform([placement_type])[0]

        # Create input with this placement type
        input_with_placement = input_data.copy()
        input_with_placement[0][-1] = encoded_placement  # Last column is Placement Type

        # Predict duration using linear regression
        predicted_duration = lr_model.predict(input_with_placement)[0]

        # Predict stability using random forest (classification for placement type suitability)
        # We'll use the probability as a stability score
        input_without_placement = input_with_placement[:, :-1]  # Remove placement type column

        # Get probability distribution
        try:
            proba = rf_model.predict_proba(input_without_placement)[0]
            rf_classes = list(getattr(rf_model, "classes_", []))
            if encoded_placement in rf_classes:
                stability_score = proba[rf_classes.index(encoded_placement)] * 100
            else:
                stability_score = 50.0
        except:
            stability_score = 50.0

        predictions.append({
            "type": placement_type,
            "duration": max(0, int(predicted_duration)),  # Duration in days
            "stability": round(stability_score, 1)
        })

    # Sort by stability score (descending)
    predictions.sort(key=lambda x: x["stability"], reverse=True)

    return predictions

def extract_profile_from_form(form):
    """Extract profile data from comparison form"""
    return {
        "child_age": form.child_age.data,
        "child_gender": form.child_gender.data,
        "child_ethnicity": form.child_ethnicity.data,
        "child_prior_placements": form.child_prior_placements.data,
        "returning_child": form.returning_child.data,
        "missing_episodes": form.missing_episodes.data,
        "sibling_group_size": form.sibling_group_size.data,
        "placed_with_siblings": form.placed_with_siblings.data,
        "carer_age": form.carer_age.data,
        "carer_gender": form.carer_gender.data,
        "carer_ethnicity": form.carer_ethnicity.data,
        "eh_involvement": form.eh_involvement.data,
        "yot_involvement": form.yot_involvement.data,
    }

def compare_placement_options(profile_data, selected_types, rf_model, lr_model, feature_encoders, placement_encoder):
    """
    Compare multiple placement options for a given profile.
    Returns predictions filtered by selected placement types.
    """
    # Create a mock form-like object with the profile data
    class MockForm:
        pass

    mock_form = MockForm()
    for key, value in profile_data.items():
        setattr(mock_form, key, type('obj', (object,), {'data': value}))

    # Prepare input
    input_data = prepare_prediction_input(mock_form, feature_encoders)

    # Generate all predictions
    all_predictions = generate_predictions_list(input_data, rf_model, lr_model, placement_encoder)

    # Filter by selected types
    filtered_predictions = [p for p in all_predictions if p["type"] in selected_types]

    return filtered_predictions

# ============== Data Upload Utility Functions ==============

def process_bulk_upload(connection, csv_file, user_id):
    """
    Process bulk CSV upload of placement data.
    Returns dict with success/failure counts.
    """
    from .models import add_placement_record

    results = {"success": 0, "failed": 0, "errors": []}

    try:
        # Read CSV file
        csv_content = csv_file.read().decode('utf-8')
        csv_reader = csv.DictReader(StringIO(csv_content))

        for row_num, row in enumerate(csv_reader, start=2):
            try:
                # Map CSV columns to database fields
                placement_data = {
                    'child_age': float(row.get('Child Age At Placement', 0)),
                    'child_gender': row.get('Child Gender', 'Unknown'),
                    'child_ethnicity': row.get('Child Ethnicity', 'Unknown'),
                    'carer_age': float(row.get('Carer Age', 0)),
                    'carer_gender': row.get('Carer Gender', row.get('Carer Gender Composition', 'Unknown')),
                    'carer_ethnicity': row.get('Carer Ethnicity', row.get('Carer Ethnicity Or Religion', 'Unknown')),
                    'placement_type': row.get('Placement Type', 'Unknown'),
                    'uploaded_by': user_id
                }

                # Add to database
                add_placement_record(connection, placement_data)
                results["success"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Row {row_num}: {str(e)}")

    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"File processing error: {str(e)}")

    return results
