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

def create_comparison_visualizations(metrics_df, rf_model, gnb_model, lr_model,
                                     x_test, y_class_test, y_reg_test,
                                     rf_pred, gnb_pred, lr_pred, ann_history,
                                     model_results, sorted_results):
    """Create visualizations for model comparison."""

    fig = plt.figure(figsize=(20, 15))

    # 1. Accuracy Comparison (Classification Models)
    plt.subplot(2, 4, 1)
    accuracy_data = metrics_df[metrics_df['Accuracy'] != 'N/A']
    plt.bar(accuracy_data['Model'], accuracy_data['Accuracy'])
    plt.title('Classification Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # 2. MSE Comparison (Regression Model)
    plt.subplot(2, 4, 2)
    mse_data = metrics_df[metrics_df['MSE'] != 'N/A']
    plt.bar(mse_data['Model'], mse_data['MSE'])
    plt.title('Linear Regression MSE')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)

    # 3. Cross-Validation Scores
    plt.subplot(2, 4, 3)
    cv_data = metrics_df[metrics_df['Cross_Val_Mean'] != 'N/A']
    x_pos = np.arange(len(cv_data))
    plt.bar(x_pos, cv_data['Cross_Val_Mean'], yerr=cv_data['Cross_Val_Std'],
            capsize=5, alpha=0.7)
    plt.xticks(x_pos, cv_data['Model'], rotation=45)
    plt.title('Cross-Validation Scores (5-Fold)')
    plt.ylabel('Mean Score')

    # 4. Confusion Matrix - Random Forest
    plt.subplot(2, 4, 4)
    cm_rf = confusion_matrix(y_class_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
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
    plt.scatter(y_reg_test, lr_pred, alpha=0.5)
    plt.plot([y_reg_test.min(), y_reg_test.max()],
             [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression: Predicted vs Actual')

    # 7. Feature Importance (Random Forest)
    plt.subplot(2, 4, 7)
    feature_names = ["Child Age", "Child Gender", "Child Ethnicity",
                     "Carer Age", "Carer Gender", "Carer Ethnicity"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.barh(range(len(importances)), importances[indices])
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance')

    # 8. Deep ANN Training History
    plt.subplot(2, 4, 8)
    plt.plot(ann_history.history['accuracy'], label='Training Accuracy')
    plt.plot(ann_history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Deep ANN Training History')
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'model_comparison_analysis.png'")
    plt.show()

    # 9. Placement Type Predictions by Model
    plt.subplot(3, 3, 9)
    placement_names = [p[0] for p in sorted_results]
    models = ['LR', 'RF', 'GNB', 'ANN']
    x_pos = np.arange(len(placement_names))
    width = 0.2

    for i, model_name in enumerate(models):
        predictions = [model_results[ptype][i][1] for ptype in placement_names]
        plt.bar(x_pos + i * width, predictions, width, label=model_name)

    plt.xlabel('Placement Type')
    plt.ylabel('Predicted Days')
    plt.title('Model Predictions by Placement Type')
    plt.xticks(x_pos + width * 1.5, placement_names, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig('placement_type_predictions.png', dpi=300, bbox_inches='tight')
    print("Placement type predictions saved as 'placement_type_predictions.png'")
    plt.show()

def compare_models():
    """Compare models across selected placement types with comprehensive metrics and visualizations."""
    # Keep your existing placement type selection logic
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

    # Keep your existing user input logic
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

    # Train models once and collect metrics
    print("\n=== Training Models ===")

    # Linear Regression
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_reg_train)
    lr_test_pred = lr_model.predict(x_test)
    lr_mse = mean_squared_error(y_reg_test, lr_test_pred)
    lr_mae = mean_absolute_error(y_reg_test, lr_test_pred)
    lr_r2 = r2_score(y_reg_test, lr_test_pred)
    lr_cv = cross_val_score(lr_model, x_train, y_reg_train, cv=5, scoring='neg_mean_squared_error')

    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_class_train)
    rf_test_pred = rf_model.predict(x_test)
    rf_accuracy = accuracy_score(y_class_test, rf_test_pred)
    rf_cv = cross_val_score(rf_model, x_train, y_class_train, cv=5, scoring='accuracy')

    # Gaussian Naive Bayes
    print("Training Gaussian Naive Bayes...")
    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_class_train)
    gnb_test_pred = gnb_model.predict(x_test)
    gnb_accuracy = accuracy_score(y_class_test, gnb_test_pred)
    gnb_cv = cross_val_score(gnb_model, x_train, y_class_train, cv=5, scoring='accuracy')

    # Deep ANN
    print("Training Deep ANN...")
    num_classes = len(np.unique(y_classification))
    y_train_cat = to_categorical(y_class_train, num_classes)
    y_test_cat = to_categorical(y_class_test, num_classes)

    ann_model = Sequential([
        Dense(64, activation='relu', input_shape=(x.shape[1],)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = ann_model.fit(x_train, y_train_cat, epochs=50, batch_size=32,
                           validation_data=(x_test, y_test_cat), verbose=0)
    _, ann_accuracy = ann_model.evaluate(x_test, y_test_cat, verbose=0)
    ann_test_pred = np.argmax(ann_model.predict(x_test, verbose=0), axis=1)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest', 'Gaussian NB', 'Deep ANN'],
        'Accuracy': ['N/A', f'{rf_accuracy:.4f}', f'{gnb_accuracy:.4f}', f'{ann_accuracy:.4f}'],
        'MSE': [f'{lr_mse:.2f}', 'N/A', 'N/A', 'N/A'],
        'MAE': [f'{lr_mae:.2f}', 'N/A', 'N/A', 'N/A'],
        'R2_Score': [f'{lr_r2:.4f}', 'N/A', 'N/A', 'N/A'],
        'Cross_Val_Mean': [f'{-lr_cv.mean():.2f}', f'{rf_cv.mean():.4f}', f'{gnb_cv.mean():.4f}', 'N/A'],
        'Cross_Val_Std': [f'{lr_cv.std():.2f}', f'{rf_cv.std():.4f}', f'{gnb_cv.std():.4f}', 'N/A']
    })

    print("\n=== Model Comparison Statistics ===")
    print(metrics_df.to_string(index=False))

    # Make predictions for selected placement types
    model_results = {ptype: [] for ptype in selected_types}
    for ptype in selected_types:
        modified_data = example_data_np.copy()

        lr_pred = lr_model.predict(modified_data)
        model_results[ptype].append(('Linear Regression', lr_pred[0]))

        rf_pred = rf_model.predict(modified_data)
        model_results[ptype].append(('Random Forest', rf_pred[0]))

        gnb_pred = gnb_model.predict(modified_data)
        model_results[ptype].append(('Gaussian NB', gnb_pred[0]))

        ann_pred = ann_model.predict(modified_data, verbose=0)
        ann_pred_class = np.argmax(ann_pred, axis=1)
        model_results[ptype].append(('Deep ANN', ann_pred_class[0]))

    # Calculate average predictions
    average_results = {}
    for ptype, predictions in model_results.items():
        average_results[ptype] = np.mean([pred[1] for pred in predictions])

    sorted_results = sorted(average_results.items(), key=lambda item: item[1], reverse=True)
    print("\n=== Placement Types Ordered by Predicted Stability ===")
    for ptype, avg_time in sorted_results:
        print(f"{ptype}: {avg_time:.2f} days")

    # Create comprehensive visualizations
    create_comparison_visualizations(
        metrics_df, rf_model, gnb_model, lr_model,
        x_test, y_class_test, y_reg_test,
        rf_test_pred, gnb_test_pred, lr_test_pred,
        history, model_results, sorted_results
    )

    # Save metrics
    metrics_df.to_csv('model_comparison_metrics.csv', index=False)
    print("\nMetrics saved to 'model_comparison_metrics.csv'")

    return metrics_df, model_results, sorted_results

if __name__ == "__main__":
    run_model()
    compare_models()
