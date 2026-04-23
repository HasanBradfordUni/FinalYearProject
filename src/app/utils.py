import pandas as pd
import csv
from io import StringIO

BCFT_UPLOAD_SCHEMA = [
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
    "Placement Type",
    "Placement Start Date",
    "Move Date",
    "Days Placed",
    "Move Reason",
    "Distance From Home",
    "EH involvement",
    "YOT involvement",
    "Placement Sequence Number",
]

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

def prepare_prediction_input(form, feature_encoders, numeric_defaults=None):
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

    def parse_int(value, default=0):
        try:
            if value in [None, "", "None"]:
                return int(default)
            return int(float(value))
        except Exception:
            return int(default)

    numeric_defaults = numeric_defaults or {}
    child_age_default = parse_int(numeric_defaults.get("child_age", 12), 12)
    child_prior_default = parse_int(numeric_defaults.get("child_prior_placements", 1), 1)
    missing_episodes_default = parse_int(numeric_defaults.get("missing_episodes", 1), 1)
    sibling_group_default = max(1, parse_int(numeric_defaults.get("sibling_group_size", 1), 1))
    carer_age_default = parse_int(numeric_defaults.get("carer_age", 45), 45)

    child_prior_placements = parse_int(get_value("child_prior_placements", child_prior_default), child_prior_default)

    # Setup a dataframe with entered/default values
    df = pd.DataFrame([
        {
            "Child Age At Placement": parse_int(get_value("child_age", child_age_default), child_age_default),
            "Child Gender": get_value("child_gender", "Unknown"),
            "Child Ethnicity": get_value("child_ethnicity", "Unknown"),
            "Child Prior Placements": child_prior_placements,
            "Returning Child": parse_bool(get_value("returning_child", "False")),
            "Missing Episodes": parse_int(get_value("missing_episodes", missing_episodes_default), missing_episodes_default),
            "Sibling Group Size": max(1, parse_int(get_value("sibling_group_size", sibling_group_default), sibling_group_default)),
            "Placed With Siblings": parse_bool(get_value("placed_with_siblings", "False")),
            "Carer Age": parse_int(get_value("carer_age", carer_age_default), carer_age_default),
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

def _derive_explanation_factors(feature_row, feature_names, importance_values, max_items=3):
    if feature_names is None or importance_values is None:
        return []

    factors = []
    for idx, feature_name in enumerate(feature_names):
        if idx >= len(importance_values) or idx >= len(feature_row):
            continue
        factors.append((feature_name, float(importance_values[idx]), feature_row[idx]))

    factors.sort(key=lambda item: item[1], reverse=True)
    return [
        f"{name} (importance {importance:.2f}, profile value {value})"
        for name, importance, value in factors[:max_items]
    ]


def generate_predictions_list(
    input_data,
    rf_model,
    lr_model,
    placement_encoder,
    breakdown_model=None,
    placement_feature_names=None,
    breakdown_feature_names=None,
):
    """
    Generate predictions for all placement types.
    Returns a list of dictionaries with type, duration, and stability score.
    """
    predictions = []

    if rf_model is None or lr_model is None or placement_encoder is None:
        return [{
            "type": "Error",
            "duration": 0,
            "stability": 0,
            "breakdown_likelihood": 0,
            "net_stability": 0,
            "explanation_factors": ["Models are not loaded."],
            "message": "Models not loaded",
        }]

    # Get all placement types
    placement_types = placement_encoder.classes_

    for placement_type in placement_types:
        # Encode the placement type
        encoded_placement = placement_encoder.transform([placement_type])[0]

        # Create input with this placement type
        input_with_placement = input_data.copy()
        input_with_placement[0][-1] = encoded_placement  # Last column is Placement Type

        # Predict duration using regression model
        predicted_duration = lr_model.predict(input_with_placement)[0]

        # Predict stability using random forest (classification for placement type suitability)
        # We'll use the probability as a stability score
        input_without_placement = input_with_placement[:, :-1]  # Remove placement type column

        # Placement suitability probability.
        try:
            proba = rf_model.predict_proba(input_without_placement)[0]
            rf_classes = list(getattr(rf_model, "classes_", []))
            if encoded_placement in rf_classes:
                stability_score = proba[rf_classes.index(encoded_placement)] * 100
            else:
                stability_score = 50.0
        except Exception:
            stability_score = 50.0

        breakdown_likelihood = 50.0
        if breakdown_model is not None:
            try:
                breakdown_proba = breakdown_model.predict_proba(input_with_placement)[0]
                breakdown_classes = list(getattr(breakdown_model, "classes_", []))
                if 1 in breakdown_classes:
                    breakdown_likelihood = breakdown_proba[breakdown_classes.index(1)] * 100
                else:
                    breakdown_likelihood = float(max(breakdown_proba) * 100)
            except Exception:
                breakdown_likelihood = 50.0

        explanation = _derive_explanation_factors(
            feature_row=input_without_placement[0],
            feature_names=placement_feature_names,
            importance_values=getattr(rf_model, "feature_importances_", None),
        )
        if not explanation and breakdown_model is not None:
            explanation = _derive_explanation_factors(
                feature_row=input_with_placement[0],
                feature_names=breakdown_feature_names,
                importance_values=getattr(breakdown_model, "feature_importances_", None),
            )

        net_stability = max(0.0, stability_score - breakdown_likelihood)

        predictions.append({
            "type": placement_type,
            "duration": max(0, int(predicted_duration)),  # Duration in days
            "stability": round(stability_score, 1),
            "breakdown_likelihood": round(breakdown_likelihood, 1),
            "net_stability": round(net_stability, 1),
            "explanation_factors": explanation,
        })

    # Rank options by lowest breakdown risk first, then strongest stability, then duration.
    predictions.sort(key=lambda x: (x["breakdown_likelihood"], -x["stability"], -x["duration"]))

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

def compare_placement_options(
    profile_data,
    selected_types,
    rf_model,
    lr_model,
    feature_encoders,
    placement_encoder,
    numeric_defaults=None,
    breakdown_model=None,
    placement_feature_names=None,
    breakdown_feature_names=None,
):
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
    input_data = prepare_prediction_input(mock_form, feature_encoders, numeric_defaults=numeric_defaults)

    # Generate all predictions
    all_predictions = generate_predictions_list(
        input_data,
        rf_model,
        lr_model,
        placement_encoder,
        breakdown_model=breakdown_model,
        placement_feature_names=placement_feature_names,
        breakdown_feature_names=breakdown_feature_names,
    )

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
        csv_headers = csv_reader.fieldnames or []
        missing_columns = [column for column in BCFT_UPLOAD_SCHEMA if column not in csv_headers]
        if missing_columns:
            raise ValueError(f"CSV schema mismatch. Missing columns: {missing_columns}")

        def parse_float(value, default=0.0):
            try:
                if value in [None, "", "None"]:
                    return default
                return float(value)
            except Exception:
                return default

        for row_num, row in enumerate(csv_reader, start=2):
            try:
                # Map CSV columns to database fields
                placement_data = {
                    'child_age': parse_float(row.get('Child Age At Placement'), 10),
                    'child_gender': row.get('Child Gender', 'Unknown'),
                    'child_ethnicity': row.get('Child Ethnicity', 'Unknown'),
                    'child_prior_placements': int(parse_float(row.get('Child Prior Placements'), 0)),
                    'returning_child': int(str(row.get('Returning Child', 'False')).strip().lower() in {'true', '1', 'yes'}),
                    'missing_episodes': int(parse_float(row.get('Missing Episodes'), 0)),
                    'sibling_group_size': max(1, int(parse_float(row.get('Sibling Group Size'), 1))),
                    'placed_with_siblings': int(str(row.get('Placed With Siblings', 'False')).strip().lower() in {'true', '1', 'yes'}),
                    'carer_age': parse_float(row.get('Carer Age'), 45),
                    'carer_gender': row.get('Carer Gender', row.get('Carer Gender Composition', 'Unknown')),
                    'carer_ethnicity': row.get('Carer Ethnicity', row.get('Carer Ethnicity Or Religion', 'Unknown')),
                    'eh_involvement': int(str(row.get('EH involvement', 'False')).strip().lower() in {'true', '1', 'yes'}),
                    'yot_involvement': int(str(row.get('YOT involvement', 'False')).strip().lower() in {'true', '1', 'yes'}),
                    'placement_sequence_number': int(parse_float(row.get('Placement Sequence Number'), 1)),
                    'placement_type': row.get('Placement Type', 'Unknown'),
                    'placement_duration': int(parse_float(row.get('Days Placed'), 0)),
                    'placement_start_date': row.get('Placement Start Date'),
                    'move_date': row.get('Move Date'),
                    'move_reason': row.get('Move Reason', ''),
                    'distance_from_home': parse_float(row.get('Distance From Home'), 0),
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
