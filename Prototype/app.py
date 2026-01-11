from flask import Flask, render_template, request, jsonify, send_from_directory
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

from utils import generate_predictions_list

app = Flask(__name__)
ethnicity_options = ["Asian/British Asian – Chinese", "Asian/British Asian – Other", #list of all ethnicity options
                     "Black/Black British – Other", "Gypsy / Roma", "Black/Black British – African",
                     "White – British", "White – Irish", "Asian/British Asian – Indian",
                     "White – Other", "Black/Black British – Caribbean", "Asian/British Asian – Pakistani",
                     "Asian/British Asian – Bangladeshi", "Mixed - White/Black African",
                     "Mixed - White/Asian", "Mixed – Other", "Traveller – Other", "Gypsy", "Roma",
                     "White - Central European", "Mixed - White/Black Caribbean",
                     "Other Ethnic Group", "Dual Heritage - Black/White", "White - Eastern European"]
gender_options = ["Non binary", "Male", "Trans Female", "Female", "Trans Male"] #list of all gender options
placement_profiles = [{"name": "Gloriana Unique","childAge": 7,"childGender": "Female","childEthnicity": "White – British",
                       "carerAge": 45,"carerGender": "Male","carerEthnicity": "Black/Black British – Caribbean"
}] # list of saved profiles (including 1 example)
placement_types = ["Fostering - Long Term", "Fostering - Short Term", # list of placement types (checkboxes)
                   "Kinship", "Residential", "Special Guardianship",
                   "Fostering - Emergency", "Fostering - Respite", "Adoption"]

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

def preprocess_input(form):
    # Convert form data into a DataFrame
    df = pd.DataFrame([{
        "Child Age At Placement": float(form["childAge"]),
        "Child Gender": form["childGender"],
        "Child Ethnicity": form["childEthnicity"],
        "Carer Age": float(form["carerAge"]),
        "Carer Gender Composition": form["carerGender"],
        "Carer Ethnicity Or Religion": form["carerEthnicity"]
    }])

    # Apply encoders
    for col, encoder in feature_encoders.items():
        df[col] = encoder.transform(df[col].astype(str))

    return df.values

if __name__ == '__main__':
    app.run(debug=True)