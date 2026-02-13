from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
import os
import joblib
import pandas as pd

try: #Attempt to load models when the app starts
    #Load trained models that are saved in the models directory
    lr_model = joblib.load("models/lr_regressor.pkl")
    rf_model = joblib.load("models/rf_classifier.pkl")
    #Load encoders (also saved in the models directory)
    feature_encoders = joblib.load("models/feature_encoders.pkl")
    placement_encoder = joblib.load("models/placement_encoder.pkl")
except Exception as e:
    print(f"Error loading models or encoders: {e}")
    lr_model = None
    rf_model = None
    feature_encoders = None
    placement_encoder = None

from utils import generate_predictions_list, PlacementProfile, load_profiles, save_profiles

app = Flask(__name__)
ethnicity_options = ["Asian/British Asian - Chinese", "Asian/British Asian - Other", #list of all ethnicity options
                     "Black/Black British - Other", "Gypsy / Roma", "Black/Black British - African",
                     "White - British", "White - Irish", "Asian/British Asian - Indian",
                     "White - Other", "Black/Black British - Caribbean", "Asian/British Asian - Pakistani",
                     "Asian/British Asian - Bangladeshi", "Mixed - White/Black African",
                     "Mixed - White/Asian", "Mixed - Other", "Traveller - Other",
                     "White - Central European", "Mixed - White/Black Caribbean",
                     "Other Ethnic Group", "Dual Heritage - Black/White", "White - Eastern European"]
gender_options = ["Non binary", "Male", "Trans Female", "Female", "Trans Male"] #list of all gender options
placement_profiles = load_profiles("profiles.txt") # list of saved profiles (loaded from text file)
placement_types = ["Fostering - Long Term", "Fostering - Short Term", # list of placement types (checkboxes)
                   "Kinship", "Residential", "Special Guardianship",
                   "Fostering - Emergency", "Fostering - Respite", "Adoption"]
PROFILE_TO_MODEL_MAP = {
    "Child Age At Placement": "childAge",
    "Child Gender": "childGender",
    "Child Ethnicity": "childEthnicity",
    "Carer Age": "carerAge",
    "Carer Gender Composition": "carerGender",
    "Carer Ethnicity Or Religion": "carerEthnicity"
} #Store a mapping of profile attributes to model feature names

@app.route('/')
def index():
    #Render the main dashboard page.
    return render_template('index.html',ethnicity_options=ethnicity_options,
    gender_options=gender_options, placement_types=placement_types, placement_profiles=placement_profiles)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Preprocess input
        input_data = preprocess_input(request.form)

        # Run predictions
        predictions_list = generate_predictions_list(input_data,
                           rf_model,lr_model,placement_encoder)

        return render_template(
            "results.html",
            child_age=request.form["childAge"],
            child_gender=request.form["childGender"],
            child_ethnicity=request.form["childEthnicity"],
            carer_age=request.form["carerAge"],
            carer_gender=request.form["carerGender"],
            carer_ethnicity=request.form["carerEthnicity"],
            predictions=predictions_list  # list of 4 items
        )

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/add_profile', methods=['POST'])
def add_profile():
    profile_name = request.form["profileName"]
    age_child_left_care = "N/A" #Set this variable as may not be captured from form (optional)
    try: #Try to get the age child left care value from the form
        age_child_left_care = int(request.form["ageChildLeftCare"])
    except: #If not provided, leave as N/A
        pass
    profile = {
        "name": request.form["profileName"],
        "childAge": request.form["childAge"],
        "childGender": request.form["childGender"],
        "childEthnicity": request.form["childEthnicity"],
        "carerAge": request.form["carerAge"],
        "carerGender": request.form["carerGender"],
        "carerEthnicity": request.form["carerEthnicity"],
        "childPriorPlacementsNum": request.form["childPriorPlacementsNum"],
        "returningChild": request.form["returningChild"],
        "ageChildLeftCare": age_child_left_care,
        "missingEpisodes": request.form["missingEpisodes"],
        "numberOfCarers": request.form["numberOfCarers"],
        "placedWithSiblings": request.form["placedWithSiblings"],
        "siblingGroupSize": request.form["siblingGroupSize"],
        "ageChildCameIntoCare": request.form["ageChildCameIntoCare"],
        "involvementOfEH": request.form["involvementOfEH"],
        "siblingsInEH": request.form["siblingsInEH"],
        "involvementOfYOT": request.form["involvementOfYOT"]
    }

    placement_profiles.append(PlacementProfile(name=profile_name, attributes_dict=profile))
    save_profiles("profiles.txt", placement_profiles)

    return redirect('/')

@app.route('/compare', methods=['POST'])
def compare():
    try:
        profile_name = request.form.get("profileName")
        selected_types = request.form.getlist("placementTypes")

        if not profile_name:
            return "Error: No profile selected."

        if len(selected_types) < 2 or len(selected_types) > 4:
            return "Error: Please select between 2 and 4 placement types."

        # Find the profile object
        profile_obj = next((p for p in placement_profiles if p.get_name() == profile_name), None)

        if profile_obj is None:
            return f"Error: Profile '{profile_name}' not found."

        profile_dict = profile_obj.to_dict()
        this_profile = profile_dict.copy()

        # Build model input row
        model_row = {}

        for model_col, profile_key in PROFILE_TO_MODEL_MAP.items():
            model_row[model_col] = profile_dict.get(profile_key)

        # Add placeholder for Placement Type
        model_row["Placement Type"] = "TEMP"

        # Convert to DataFrame
        df = pd.DataFrame([model_row])

        # Encode all except Placement Type
        for col, encoder in feature_encoders.items():
            if col != "Placement Type":
                df[col] = encoder.transform(df[col].astype(str))

        # Placeholder for Placement Type (will be replaced)
        df["Placement Type"] = 0

        input_data = df.values

        # Generate predictions
        predictions_list = generate_predictions_list(
            input_data=input_data,
            rf_model=rf_model,
            lr_model=lr_model,
            placement_encoder=placement_encoder
        )

        # Filter predictions to only the selected placement types
        predictions_list = [
            p for p in predictions_list if p["type"] in selected_types
        ]

        return render_template(
            "comparison_results.html",
            profile_name=profile_name,
            profile=this_profile,
            predictions=predictions_list
        )

    except Exception as e:
        return f"Error: {str(e)}"

def preprocess_input(form):
    """
    Prepares user input so it matches the exact feature structure
    used during model training.
    """
    def get_value(key, default):
        val = form.get(key)
        return val if val not in [None, "", "None"] else default

    # Setup a dataframe with entered/default values (if blank) for all features (except placement type)
    df = pd.DataFrame([{
        "Child Age At Placement": float(get_value("childAge", 10)),  # median age
        "Child Gender": get_value("childGender", "Unknown"),
        "Child Ethnicity": get_value("childEthnicity", "Unknown"),
        "Carer Age": float(get_value("carerAge", 45)),  # median carer age
        "Carer Gender Composition": get_value("carerGender", "Unknown"),
        "Carer Ethnicity Or Religion": get_value("carerEthnicity", "Unknown"),
        "Placement Type": None   # placeholder, DO NOT ENCODE HERE
    }])

    # Apply encoders to all columns EXCEPT Placement Type
    for col, encoder in feature_encoders.items():
        if col != "Placement Type":
            # Ensure Unknown is valid for input columns only
            df[col] = df[col].astype(str)
            df[col] = encoder.transform(df[col])

    # For now, set Placement Type to 0 (will be replaced later when predicting via random forest)
    df["Placement Type"] = 0

    return df.values

if __name__ == '__main__':
    app.run(debug=True)