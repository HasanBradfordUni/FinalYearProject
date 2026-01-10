from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_data(filepath):
    """Load and prepare the dataset."""
    data = pd.read_csv(filepath, encoding='latin-1')

    # Select relevant columns
    features = ["Child Age At Placement", "Child Gender", "Child Ethnicity",
                "Carer Age", "Carer Gender Composition", "Carer Ethnicity Or Religion"]
    target_regression = "Placement Time Period Days"
    target_classification = "Placement Time Period Class"

    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ["Child Gender", "Child Ethnicity",
                          "Carer Gender Composition", "Carer Ethnicity Or Religion"]

    for col in categorical_columns:
        data[col] = le.fit_transform(data[col].astype(str))

    # Prepare features and targets
    x = data[features].values
    y_regression = data[target_regression].values
    y_classification = le.fit_transform(data[target_classification].astype(str))

    return x, y_regression, y_classification, le

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/run_comparison', methods=['POST'])
def run_comparison():
    """Run model comparison based on user input."""
    try:
        data = request.get_json()

        # Extract form data
        child_age = float(data['childAge'])
        child_gender = data['childGender']
        child_ethnicity = data['childEthnicity']
        carer_age = float(data['carerAge'])
        carer_gender = data['carerGender']
        carer_ethnicity = data['carerEthnicity']
        placement_types = data['placementTypes']

        # Load full dataset
        x, y_regression, y_classification, label_encoder = load_data('dataset.csv')

        # Encode user input
        original_data = pd.read_csv('dataset.csv', encoding='latin-1')
        le = LabelEncoder()

        # Create input DataFrame
        input_df = pd.DataFrame([[child_age, child_gender, child_ethnicity,
                                 carer_age, carer_gender, carer_ethnicity]],
                               columns=["Child Age At Placement", "Child Gender",
                                       "Child Ethnicity", "Carer Age",
                                       "Carer Gender Composition", "Carer Ethnicity Or Religion"])

        # Encode categorical columns
        categorical_columns = ["Child Gender", "Child Ethnicity",
                              "Carer Gender Composition", "Carer Ethnicity Or Religion"]

        for col in categorical_columns:
            le.fit(original_data[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))

        input_data = input_df.values

        # Split data for testing
        x_train, x_test, y_reg_train, y_reg_test = train_test_split(
            x, y_regression, test_size=0.2, random_state=42)
        _, _, y_class_train, y_class_test = train_test_split(
            x, y_classification, test_size=0.2, random_state=42)

        # Train and evaluate models
        model_results = {}
        metrics = []

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(x_train, y_reg_train)
        lr_pred = lr_model.predict(x_test)
        lr_mse = mean_squared_error(y_reg_test, lr_pred)
        lr_user_pred = lr_model.predict(input_data)[0]

        metrics.append({
            'model': 'Linear Regression',
            'accuracy': 'N/A',
            'mse': lr_mse,
            'prediction': lr_user_pred
        })

        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(x_train, y_class_train)
        rf_pred = rf_model.predict(x_test)
        rf_accuracy = accuracy_score(y_class_test, rf_pred)
        rf_user_pred = rf_model.predict(input_data)[0]

        metrics.append({
            'model': 'Random Forest',
            'accuracy': rf_accuracy,
            'mse': 'N/A',
            'prediction': rf_user_pred
        })

        # Gaussian Naive Bayes
        gnb_model = GaussianNB()
        gnb_model.fit(x_train, y_class_train)
        gnb_pred = gnb_model.predict(x_test)
        gnb_accuracy = accuracy_score(y_class_test, gnb_pred)
        gnb_user_pred = gnb_model.predict(input_data)[0]

        metrics.append({
            'model': 'Gaussian Naive Bayes',
            'accuracy': gnb_accuracy,
            'mse': 'N/A',
            'prediction': gnb_user_pred
        })

        # Deep ANN
        num_classes = len(np.unique(y_classification))
        y_class_train_cat = to_categorical(y_class_train, num_classes)
        y_class_test_cat = to_categorical(y_class_test, num_classes)

        ann_model = Sequential([
            Dense(64, activation='relu', input_shape=(x.shape[1],)),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        ann_model.compile(optimizer='adam', loss='categorical_crossentropy',
                         metrics=['accuracy'])
        ann_model.fit(x_train, y_class_train_cat, epochs=50, batch_size=32,
                     verbose=0, validation_split=0.2)

        _, ann_accuracy = ann_model.evaluate(x_test, y_class_test_cat, verbose=0)
        ann_user_pred = np.argmax(ann_model.predict(input_data, verbose=0), axis=1)[0]

        metrics.append({
            'model': 'Deep ANN',
            'accuracy': ann_accuracy,
            'mse': 'N/A',
            'prediction': ann_user_pred
        })

        # Generate placement type predictions
        placement_predictions = {}
        for placement_type in placement_types:
            placement_predictions[placement_type] = {
                'lr': lr_user_pred,
                'rf': rf_user_pred,
                'gnb': gnb_user_pred,
                'ann': ann_user_pred
            }

        # Create visualizations
        create_comparison_viz(metrics, rf_model, lr_pred, y_reg_test)
        create_placement_viz(placement_predictions, placement_types)

        return jsonify({
            'success': True,
            'metrics': metrics,
            'comparison_viz': '/static/images/model_comparison_analysis.png',
            'placement_viz': '/static/images/placement_type_predictions.png',
            'placement_predictions': placement_predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def create_comparison_viz(metrics, rf_model, lr_pred, y_reg_test):
    """Create model comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy comparison
    ax1 = axes[0, 0]
    class_models = [m for m in metrics if m['accuracy'] != 'N/A']
    ax1.bar([m['model'] for m in class_models],
            [m['accuracy'] for m in class_models])
    ax1.set_title('Classification Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # MSE comparison
    ax2 = axes[0, 1]
    reg_models = [m for m in metrics if m['mse'] != 'N/A']
    ax2.bar([m['model'] for m in reg_models],
            [m['mse'] for m in reg_models])
    ax2.set_title('Regression Model MSE', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Squared Error')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Feature importance
    ax3 = axes[1, 0]
    feature_names = ["Child Age", "Child Gender", "Child Ethnicity",
                     "Carer Age", "Carer Gender", "Carer Ethnicity"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    ax3.barh(range(len(importances)), importances[indices])
    ax3.set_yticks(range(len(importances)))
    ax3.set_yticklabels([feature_names[i] for i in indices])
    ax3.set_xlabel('Importance')
    ax3.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')

    # Predicted vs Actual
    ax4 = axes[1, 1]
    ax4.scatter(y_reg_test, lr_pred, alpha=0.5)
    ax4.plot([y_reg_test.min(), y_reg_test.max()],
             [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    ax4.set_xlabel('Actual Values')
    ax4.set_ylabel('Predicted Values')
    ax4.set_title('Linear Regression: Predicted vs Actual', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'model_comparison_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_placement_viz(placement_predictions, placement_types):
    """Create placement type predictions visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = ['lr', 'rf', 'gnb', 'ann']
    model_names = ['Linear Reg', 'Random Forest', 'Gaussian NB', 'Deep ANN']
    x_pos = np.arange(len(placement_types))
    width = 0.2

    for i, (model, name) in enumerate(zip(models, model_names)):
        predictions = [placement_predictions[pt][model] for pt in placement_types]
        ax.bar(x_pos + i * width, predictions, width, label=name)

    ax.set_xlabel('Placement Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Days', fontsize=12, fontweight='bold')
    ax.set_title('Model Predictions by Placement Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(placement_types, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'placement_type_predictions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)