#Python file to test different AI models with my data
"""The models I'm including (with a separate test for each) are:
1. Linear Regression
2. Random Forest
3. Gaussian Naive Bayes
4. Deep Artificial Neural Networks"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')  # Fix encoding issue

    # Use the selected fields as features (x):
    x = data[["Child Age At Placement", "Child Gender", "Child Ethnicity",
              "Carer Age", "Carer Gender Composition", "Carer Ethnicity Or Religion"]]

    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    categorical_columns = ["Child Gender", "Child Ethnicity",
                          "Carer Gender Composition", "Carer Ethnicity Or Religion"]

    for col in categorical_columns:
        if col in x.columns:
            x[col] = le.fit_transform(x[col].astype(str))

    # Ensure numeric columns are properly typed
    x["Child Age At Placement"] = pd.to_numeric(x["Child Age At Placement"], errors='coerce')
    x["Carer Age"] = pd.to_numeric(x["Carer Age"], errors='coerce')

    # Fill any NaN values that might have been created
    x = x.fillna(0)

    # The variables that will be predicted
    y_classification = le.fit_transform(data["Placement Type"].astype(str))
    y_regression = pd.to_numeric(data["Placement Time Period (days)"], errors='coerce').fillna(0)

    return x, y_classification, y_regression

# Dictionary to store metrics
metrics = {
    'Model': [],
    'Accuracy': [],
    'MSE': [],
    'MAE': [],
    'R2_Score': [],
    'Cross_Val_Mean': [],
    'Cross_Val_Std': []
}

def run_linear_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_percent = (mse / np.mean(y_test)) * 100
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')

    # Print results
    print(f"Linear Regression MSE: {mse}")
    print(f"Linear Regression MSE as percentage of mean actual value: {mse_percent}%")
    print(f"Linear Regression MAE: {mae}")
    print(f"Linear Regression R² Score: {r2}")
    print(f"Cross-Validation MSE (5-fold): {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    return model, y_test, y_pred

def run_random_forest(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')

    # Print results
    print(f"Random Forest Accuracy: {accuracy}")
    print(f"Cross-Validation Accuracy (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    print("\nFeature Importances:")
    feature_names = ["Child Age At Placement", "Child Gender", "Child Ethnicity",
                     "Carer Age", "Carer Gender Composition", "Carer Ethnicity Or Religion"]
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"{name}: {importance:.4f}")

    return model, y_test, y_pred

def run_gaussian_nb(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')

    # Print results
    print(f"Gaussian Naive Bayes Accuracy: {accuracy}")
    print(f"Cross-Validation Accuracy (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, y_test, y_pred

def run_deep_ann(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(x.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train_cat, epochs=50, batch_size=32,
                       validation_split=0.2, verbose=0)

    loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Print results
    print(f"Deep ANN Accuracy: {accuracy}")
    print(f"Deep ANN Loss: {loss}")
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, y_test, y_pred, history


def run_model():
    model_menu = """Select the model you want to test:
1. Linear Regression
2. Random Forest
3. Gaussian Naive Bayes
4. Deep Artificial Neural Networks
Enter the number corresponding to your choice: """
    choice = input(model_menu)
    x, y_classification, y_regression = load_data("dataset.csv")

    if choice == '1':
        model, y_test, y_pred = run_linear_regression(x, y_regression)
    elif choice == '2':
        model, y_test, y_pred = run_random_forest(x, y_classification)
    elif choice == '3':
        model, y_test, y_pred = run_gaussian_nb(x, y_classification)
    elif choice == '4':
        model, y_test, y_pred, history = run_deep_ann(x, y_classification)
    else:
        print("Invalid choice. Please select a valid model number.")

def create_comparison_visualizations(regression_metrics, classification_metrics,
                                     lr_model, rf_reg_model, gb_model, ann_reg_model,
                                     rf_class_model, gnb_model, ann_class_model,
                                     x_test, y_class_test, y_reg_test,
                                     lr_test_pred, rf_reg_test_pred, gb_test_pred, ann_reg_test_pred,
                                     rf_class_pred, gnb_pred,
                                     ann_reg_history, ann_class_history,
                                     duration_predictions, sorted_results):
    """Create visualizations for model comparison."""

    fig = plt.figure(figsize=(20, 15))

    # 1. Regression Model MSE Comparison
    plt.subplot(2, 4, 1)
    plt.bar(regression_metrics['Model'], regression_metrics['MSE'])
    plt.title('Regression Models MSE Comparison')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45, ha='right')

    # 2. Regression Model R² Comparison
    plt.subplot(2, 4, 2)
    plt.bar(regression_metrics['Model'], regression_metrics['R2_Score'])
    plt.title('Regression Models R² Score Comparison')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45, ha='right')

    # 3. Classification Model Accuracy Comparison
    plt.subplot(2, 4, 3)
    plt.bar(classification_metrics['Model'], classification_metrics['Accuracy'])
    plt.title('Classification Models Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)

    # 4. Confusion Matrix - Random Forest Classifier
    plt.subplot(2, 4, 4)
    cm_rf = confusion_matrix(y_class_test, rf_class_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Classifier Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # 5. Confusion Matrix - Gaussian NB
    plt.subplot(2, 4, 5)
    cm_gnb = confusion_matrix(y_class_test, gnb_pred)
    sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Greens')
    plt.title('Gaussian NB Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # 6. Linear Regression: Predicted vs Actual
    plt.subplot(2, 4, 6)
    plt.scatter(y_reg_test, lr_test_pred, alpha=0.5)
    plt.plot([y_reg_test.min(), y_reg_test.max()],
             [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression: Predicted vs Actual')

    # 7. Feature Importance (Random Forest Regressor)
    plt.subplot(2, 4, 7)
    feature_names = ["Child Age", "Child Gender", "Child Ethnicity",
                     "Carer Age", "Carer Gender", "Carer Ethnicity"]
    importances = rf_reg_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.barh(range(len(importances)), importances[indices])
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Random Forest Regressor Feature Importance')

    # 8. Deep ANN Training History
    plt.subplot(2, 4, 8)
    plt.plot(ann_reg_history.history['loss'], label='Training Loss')
    plt.plot(ann_reg_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Deep ANN Regressor Training History')
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'model_comparison_analysis.png'")
    plt.show()

    # 9. Placement Type Predictions by Model
    fig, ax = plt.subplots(figsize=(14, 7))
    placement_names = [p[0] for p in sorted_results]
    models = ['LR', 'RF', 'GB', 'ANN']
    x_pos = np.arange(len(placement_names))
    width = 0.2

    # Get predictions for each model and placement type
    for i, model_name in enumerate(models):
        predictions = []
        for ptype in placement_names:
            pred_value = duration_predictions[ptype][model_name]
            predictions.append(pred_value)

        ax.bar(x_pos + i * width, predictions, width, label=model_name)

    ax.set_xlabel('Placement Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Days', fontsize=11, fontweight='bold')
    ax.set_title('Model Predictions by Placement Type', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(placement_names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('placement_type_predictions.png', dpi=300, bbox_inches='tight')
    print("Placement type predictions saved as 'placement_type_predictions.png'")
    plt.show()

def compare_models():
    """The purpose of this method is to enter between 2 and 4 placement types to compare and
        also some example input data (for x labels) and then order the placement types by most to least stable
        according to the models and by using the average placement time period predicted by the models
        for each placement type chosen by the user."""
    print("First choose the placement types to compare:")
    placement_types = ["Fostering - Long Term", "Fostering - Short Term", "Kinship", "Residential",
                       "Special Guardianship", "Fostering - Emergency", "Fostering - Respite", "Adoption"]
    selected_types = []
    for num in range(1, 5):
        print("Choose placement type", num)
        for i, p_type in enumerate(placement_types, start=1):
            print(f"{i}. {p_type}")
        p_choice = int(input("Enter the number corresponding to your choice (0 to skip selection): "))
        p_chosen = False
        while p_choice < 1 or p_choice > len(placement_types):
            if p_choice == 0 and len(selected_types) > 1:
                p_chosen = True
                break
            elif p_choice == 0 and len(selected_types) < 2:
                print("You must select at least two placement types to compare.")
            p_choice = int(input("Invalid choice. Please enter a valid number: "))
        if p_chosen:
            break
        else:
            selected_type = placement_types[p_choice - 1]
            selected_types.append(selected_type)

    print("Now enter example input data for the features:")
    childAge = input("Enter the Child's Age At Placement: ")
    childGender = input("Enter the Child's Gender: ")
    childEthnicity = input("Enter the Child's Ethnicity: ")
    carerAge = input("Enter the Carer's Age: ")
    carerGender = input("Enter the Carer's Gender: ")
    carerEthnicity = input("Enter the Carer's Ethnicity: ")

    # Load data
    x, y_classification, y_regression = load_data("dataset.csv")

    # Keep your existing encoding logic
    from sklearn.preprocessing import LabelEncoder
    example_df = pd.DataFrame([[childAge, childGender, childEthnicity, carerAge, carerGender, carerEthnicity]],
                              columns=["Child Age At Placement", "Child Gender", "Child Ethnicity",
                                      "Carer Age", "Carer Gender Composition", "Carer Ethnicity Or Religion"])

    le = LabelEncoder()
    categorical_columns = ["Child Gender", "Child Ethnicity", "Carer Gender Composition", "Carer Ethnicity Or Religion"]

    original_data = pd.read_csv("dataset.csv", encoding='latin-1')
    for col in categorical_columns:
        le.fit(original_data[col].astype(str))
        example_df[col] = le.transform(example_df[col].astype(str))

    example_df["Child Age At Placement"] = pd.to_numeric(example_df["Child Age At Placement"], errors='coerce')
    example_df["Carer Age"] = pd.to_numeric(example_df["Carer Age"], errors='coerce')
    example_df = example_df.fillna(0)

    example_data_np = example_df.values

    # Split data for consistent evaluation
    x_train, x_test, y_class_train, y_class_test = train_test_split(
        x, y_classification, test_size=0.2, random_state=42
    )
    _, _, y_reg_train, y_reg_test = train_test_split(
        x, y_regression, test_size=0.2, random_state=42
    )

    print("\n=== Training Models ===")

    # Train REGRESSION models for duration prediction
    print("Training Duration Prediction Models...")

    # Linear Regression for duration
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_reg_train)
    lr_test_pred = lr_model.predict(x_test)
    lr_mse = mean_squared_error(y_reg_test, lr_test_pred)
    lr_mae = mean_absolute_error(y_reg_test, lr_test_pred)
    lr_r2 = r2_score(y_reg_test, lr_test_pred)

    # Random Forest REGRESSOR for duration
    from sklearn.ensemble import RandomForestRegressor
    rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg_model.fit(x_train, y_reg_train)
    rf_reg_test_pred = rf_reg_model.predict(x_test)
    rf_reg_mse = mean_squared_error(y_reg_test, rf_reg_test_pred)
    rf_reg_mae = mean_absolute_error(y_reg_test, rf_reg_test_pred)
    rf_reg_r2 = r2_score(y_reg_test, rf_reg_test_pred)

    # Gradient Boosting for duration
    from sklearn.ensemble import GradientBoostingRegressor
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(x_train, y_reg_train)
    gb_test_pred = gb_model.predict(x_test)
    gb_mse = mean_squared_error(y_reg_test, gb_test_pred)
    gb_mae = mean_absolute_error(y_reg_test, gb_test_pred)
    gb_r2 = r2_score(y_reg_test, gb_test_pred)

    # Deep ANN for duration (regression)
    ann_reg_model = Sequential([
        Dense(64, activation='relu', input_shape=(x.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)  # Single output for regression
    ])
    ann_reg_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    ann_reg_history = ann_reg_model.fit(x_train, y_reg_train, epochs=50, batch_size=32,
                                        validation_data=(x_test, y_reg_test), verbose=0)
    ann_reg_test_pred = ann_reg_model.predict(x_test, verbose=0).flatten()
    ann_reg_mse = mean_squared_error(y_reg_test, ann_reg_test_pred)
    ann_reg_mae = mean_absolute_error(y_reg_test, ann_reg_test_pred)
    ann_reg_r2 = r2_score(y_reg_test, ann_reg_test_pred)

    # Train CLASSIFICATION models for placement type
    print("Training Placement Type Classification Models...")

    # Random Forest Classifier
    rf_class_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_class_model.fit(x_train, y_class_train)
    rf_class_pred = rf_class_model.predict(x_test)
    rf_class_accuracy = accuracy_score(y_class_test, rf_class_pred)

    # Gaussian Naive Bayes
    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_class_train)
    gnb_pred = gnb_model.predict(x_test)
    gnb_accuracy = accuracy_score(y_class_test, gnb_pred)

    # Deep ANN for classification
    num_classes = len(np.unique(y_classification))
    y_train_cat = to_categorical(y_class_train, num_classes)
    y_test_cat = to_categorical(y_class_test, num_classes)

    ann_class_model = Sequential([
        Dense(64, activation='relu', input_shape=(x.shape[1],)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    ann_class_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ann_class_history = ann_class_model.fit(x_train, y_train_cat, epochs=50, batch_size=32,
                                           validation_data=(x_test, y_test_cat), verbose=0)
    _, ann_class_accuracy = ann_class_model.evaluate(x_test, y_test_cat, verbose=0)

    # Create separate metrics DataFrames
    regression_metrics = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest Regressor', 'Gradient Boosting', 'Deep ANN Regressor'],
        'MSE': [lr_mse, rf_reg_mse, gb_mse, ann_reg_mse],
        'MAE': [lr_mae, rf_reg_mae, gb_mae, ann_reg_mae],
        'R2_Score': [lr_r2, rf_reg_r2, gb_r2, ann_reg_r2]
    })

    classification_metrics = pd.DataFrame({
        'Model': ['Random Forest', 'Gaussian NB', 'Deep ANN'],
        'Accuracy': [rf_class_accuracy, gnb_accuracy, ann_class_accuracy]
    })

    print("\n=== Duration Prediction Models (Regression) ===")
    print(regression_metrics.to_string(index=False))
    print("\n=== Placement Type Models (Classification) ===")
    print(classification_metrics.to_string(index=False))

    # Make predictions for selected placement types
    duration_predictions = {}
    placement_predictions = {}

    for ptype in selected_types:
        # Predict DURATION (days) using regression models
        lr_days = lr_model.predict(example_data_np)[0]
        rf_days = rf_reg_model.predict(example_data_np)[0]
        gb_days = gb_model.predict(example_data_np)[0]
        ann_days = ann_reg_model.predict(example_data_np, verbose=0)[0][0]

        duration_predictions[ptype] = {
            'LR': lr_days,
            'RF': rf_days,
            'GB': gb_days,
            'ANN': ann_days,
            'Average': np.mean([lr_days, rf_days, gb_days, ann_days])
        }

        # Predict PLACEMENT TYPE using classification models
        rf_type = rf_class_model.predict(example_data_np)[0]
        gnb_type = gnb_model.predict(example_data_np)[0]
        ann_type = np.argmax(ann_class_model.predict(example_data_np, verbose=0), axis=1)[0]

        placement_predictions[ptype] = {
            'RF': rf_type,
            'GNB': gnb_type,
            'ANN': ann_type
        }

    # Sort by average duration
    sorted_results = sorted(duration_predictions.items(),
                           key=lambda x: x[1]['Average'],
                           reverse=True)

    print("\n=== Placement Types Ordered by Average Predicted Duration ===")
    for ptype, preds in sorted_results:
        print(f"\n{ptype}:")
        print(f"  LR: {preds['LR']:.2f} days")
        print(f"  RF: {preds['RF']:.2f} days")
        print(f"  GB: {preds['GB']:.2f} days")
        print(f"  ANN: {preds['ANN']:.2f} days")
        print(f"  Average: {preds['Average']:.2f} days")

    # Update visualization function call
    create_comparison_visualizations(
        regression_metrics, classification_metrics,
        lr_model, rf_reg_model, gb_model, ann_reg_model,
        rf_class_model, gnb_model, ann_class_model,
        x_test, y_class_test, y_reg_test,
        lr_test_pred, rf_reg_test_pred, gb_test_pred, ann_reg_test_pred,
        rf_class_pred, gnb_pred,
        ann_reg_history, ann_class_history,
        duration_predictions, sorted_results
    )

    return regression_metrics, classification_metrics, duration_predictions, placement_predictions

if __name__ == "__main__":
    run_model()
    compare_models()
