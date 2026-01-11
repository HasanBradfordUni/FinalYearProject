# Utility methods to help with placement prediction
import numpy as np

def generate_predictions_list(input_data, rf_model, lr_model, placement_encoder):
    """
    Produces a ranked list of the top 4 most stable placement types
    with predicted durations for each.
    """
    if rf_model is None or lr_model is None or placement_encoder is None:
        raise ValueError("Models or encoder not loaded properly.")

    # 1. Get probability distribution for all classes
    class_probs = rf_model.predict_proba(input_data)[0]

    # 2. Get indices of top 4 classes
    top4_indices = class_probs.argsort()[::-1][:4]

    predictions_list = []

    # 3. For each predicted class
    for class_index in top4_indices:

        # Convert class index → placement type label
        placement_type = placement_encoder.inverse_transform([class_index])[0]

        # 4. Create a modified copy of the input
        modified_input = input_data.copy()

        # Inject the predicted placement type into the input
        # (Assuming your LR model expects placement type as a feature)
        # If not, remove this line.
        modified_input_with_class = np.append(modified_input, class_index).reshape(1, -1)

        # 5. Predict duration using Linear Regression
        predicted_days = lr_model.predict(modified_input_with_class)[0]

        # 6. Add to results list
        predictions_list.append({
            "type": placement_type,
            "days": int(predicted_days)
        })

    return predictions_list