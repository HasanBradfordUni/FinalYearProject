import pandas as pd
import csv
from io import StringIO
from pathlib import Path

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
PROFILE_TO_FEATURE = {
    "child_age": "Child Age At Placement",
    "child_prior_placements": "Child Prior Placements",
    "missing_episodes": "Missing Episodes",
    "sibling_group_size": "Sibling Group Size",
    "carer_age": "Carer Age",
}
PROFILE_LABELS = {
    "child_age": "child age",
    "child_prior_placements": "prior placements",
    "missing_episodes": "missing episodes",
    "sibling_group_size": "sibling group size",
    "carer_age": "carer age",
}

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

def _load_training_numeric_averages():
    """Load training-set averages for key numeric profile features."""
    dataset_path = Path(__file__).resolve().parent / "static" / "dataset.csv"
    defaults = {
        "child_age": 12,
        "child_prior_placements": 1,
        "missing_episodes": 1,
        "sibling_group_size": 1,
        "carer_age": 45,
    }
    if not dataset_path.exists():
        return defaults

    try:
        frame = pd.read_csv(dataset_path)
    except Exception:
        return defaults

    averages = defaults.copy()
    for profile_key, feature_name in PROFILE_TO_FEATURE.items():
        if feature_name not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[feature_name], errors="coerce")
        mean_value = numeric.mean()
        if pd.isna(mean_value):
            continue
        averages[profile_key] = float(mean_value)
    return averages


def generate_explainability_summary(
    user_profile,
    predictions,
    feature_names=None,
    feature_importances=None,
    training_averages=None,
):
    """Generate one readable explainability paragraph across all placement options."""
    if not predictions:
        return "The explainability engine could not generate a summary because no placement predictions were produced."

    top_features_text = "child and carer profile variables"
    if feature_names and feature_importances is not None:
        ranked = []
        for idx, name in enumerate(feature_names):
            if idx >= len(feature_importances):
                continue
            ranked.append((name, float(feature_importances[idx])))
        ranked.sort(key=lambda item: item[1], reverse=True)
        top_features = [name for name, _ in ranked[:3]]
        if top_features:
            top_features_text = ", ".join(top_features)

    averages = training_averages or _load_training_numeric_averages()
    profile_diffs = []
    for profile_key, label in PROFILE_LABELS.items():
        user_value = user_profile.get(profile_key)
        avg_value = averages.get(profile_key)
        if user_value in (None, "", "None") or avg_value is None:
            continue
        try:
            user_num = float(user_value)
            avg_num = float(avg_value)
        except Exception:
            continue
        delta = user_num - avg_num
        if abs(delta) < 0.5:
            continue
        direction = "above" if delta > 0 else "below"
        profile_diffs.append(f"{label} is {abs(delta):.1f} {direction} average")

    best_option = min(predictions, key=lambda p: (p.get("breakdown_likelihood", 100), -p.get("stability", 0), -p.get("duration", 0)))
    durations = [float(item.get("duration", 0)) for item in predictions]
    stability_scores = [float(item.get("stability", 0)) for item in predictions]
    breakdown_scores = [float(item.get("breakdown_likelihood", 0)) for item in predictions]

    duration_span = f"{int(min(durations))}-{int(max(durations))} days" if durations else "an uncertain duration range"
    stability_span = f"{min(stability_scores):.1f}% to {max(stability_scores):.1f}%" if stability_scores else "limited stability variation"
    breakdown_span = f"{min(breakdown_scores):.1f}% to {max(breakdown_scores):.1f}%" if breakdown_scores else "limited breakdown variation"

    comparison_sentence = ""
    if profile_diffs:
        comparison_sentence = " For this profile, " + "; ".join(profile_diffs[:3]) + ", which shifts the balance between placement options."

    return (
        f"Across all placement types, the model identifies {top_features_text} as the strongest contributors to estimated duration, "
        f"stability score, and breakdown likelihood. Predicted outcomes vary from {duration_span}, with stability spanning {stability_span} "
        f"and breakdown likelihood spanning {breakdown_span}.{comparison_sentence} "
        f"Overall, {best_option.get('type', 'the leading option')} is preferred for this child-carer profile because it combines lower predicted breakdown risk with stronger expected stability in this context."
    )


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

        predictions.append({
            "type": placement_type,
            "duration": max(0, int(predicted_duration)),  # Duration in days
            "stability": round(stability_score, 1),
            "breakdown_likelihood": round(breakdown_likelihood, 1),
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
