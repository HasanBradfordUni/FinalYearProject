# Utility methods to help with placement prediction
import numpy as np

class PlacementProfile:
    """
    Represents a saved placement profile.
    """

    def __init__(self, name, attributes_dict):
        # Set all attributes to None initially and use the setter method to update them.
        self.__name = name
        self.__childAge = None
        self.__childGender = None
        self.__childEthnicity = None
        self.__childPriorPlacementsNum = None
        self.__returningChild = None
        self.__ageChildLeftCare = None
        self.__carerAge = None
        self.__missingEpisodes = None
        self.__numberOfCarers = None
        self.__carerGender = None
        self.__placedWithSiblings = None
        self.__siblingGroupSize = None
        self.__ageChildCameIntoCare = None
        self.__carerEthnicity = None
        self.__involvementOfEH = None
        self.__siblingsInEH = None
        self.__involvementOfYOT = None
        self.__set_all_attributes(attributes_dict)

    def __set_all_attributes(self, attributes_dict):
        # Sets all attributes from a dictionary.
        for key, value in attributes_dict.items():
            mangled_name = f"_PlacementProfile__{key}"
            if hasattr(self, mangled_name):
                setattr(self, mangled_name, value)

    def to_dict(self):
        #Converts the profile to a dictionary.
        return {
            "name": self.__name,
            "childAge": self.__childAge,
            "childGender": self.__childGender,
            "childEthnicity": self.__childEthnicity,
            "childPriorPlacementsNum": self.__childPriorPlacementsNum,
            "returningChild": self.__returningChild,
            "ageChildLeftCare": self.__ageChildLeftCare,
            "carerAge": self.__carerAge,
            "missingEpisodes": self.__missingEpisodes,
            "numberOfCarers": self.__numberOfCarers,
            "carerGender": self.__carerGender,
            "placedWithSiblings": self.__placedWithSiblings,
            "siblingGroupSize": self.__siblingGroupSize,
            "ageChildCameIntoCare": self.__ageChildCameIntoCare,
            "carerEthnicity": self.__carerEthnicity,
            "involvementOfEH": self.__involvementOfEH,
            "siblingsInEH": self.__siblingsInEH,
            "involvementOfYOT": self.__involvementOfYOT
        }

    def edit_name(self, new_name):
        #Edits the profile name.
        self.__name = new_name

    def edit_attribute(self, attribute, new_value):
        #Edits a specific attribute of the profile.
        mangled_name = f"_PlacementProfile__{attribute}"
        if hasattr(self, mangled_name):
            setattr(self, mangled_name, new_value)
        else:
            raise AttributeError(f"{attribute} is not a valid attribute of Placement Profile.")

    def get_name(self):
        #Returns the profile name.
        return self.__name

    def get_attribute(self, attribute):
        #Returns a specific attribute of the profile.
        mangled_name = f"_PlacementProfile__{attribute}"
        if hasattr(self, mangled_name):
            return getattr(self, mangled_name)
        else:
            raise AttributeError(f"{attribute} is not a valid attribute of Placement Profile.")

def load_profiles(file_path):
    """
    Loads saved placement profiles from a text file.
    Each line in the file represents a profile in JSON format.
    """
    import json

    profiles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                profile = json.loads(line.strip())
                this_profile = PlacementProfile(profile.get("name"), profile)
                profiles.append(this_profile)
    except FileNotFoundError:
        pass  # If file doesn't exist, return empty list

    return profiles

def save_profiles(file_path, profiles):
    """
    Saves placement profiles to a text file.
    Each profile is saved as a JSON string on a new line.
    """
    import json

    with open(file_path, 'w', encoding='utf-8') as f:
        for profile in profiles:
            json.dump(profile.to_dict(), f)
            f.write('\n')

def generate_predictions_list(input_data, rf_model, lr_model, placement_encoder):
    """
    Produces a ranked list of the top 4 most stable placement types
    with predicted durations for each.
    """

    # Ensure input is 2D
    input_data = np.array(input_data).reshape(1, -1)

    # RF uses only the first 6 features (no placement type)
    rf_input = input_data[:, :-1]

    # Get probability distribution for all placement classes
    class_probs = rf_model.predict_proba(rf_input)[0]

    # Top 4 most likely placement types
    top4_indices = class_probs.argsort()[::-1][:4]

    predictions_list = []

    for class_index in top4_indices:

        # Convert class index to label
        placement_type = placement_encoder.inverse_transform([class_index])[0]

        # Copy the full 7-feature input for LR
        modified_input = input_data.copy()

        # Replace placeholder with predicted class index
        modified_input[0, -1] = class_index

        # Predict duration using LR
        predicted_days = lr_model.predict(modified_input)[0]

        predictions_list.append({
            "type": placement_type,
            "days": int(predicted_days)
        })

    # Order the predictions list by days ascending
    predictions_list = sorted(predictions_list, key=lambda x: x["days"])

    return predictions_list